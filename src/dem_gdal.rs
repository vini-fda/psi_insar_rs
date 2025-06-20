use std::path::Path;

use gdal::raster::Buffer;
use ndarray::Array2;

use crate::{egm_2008, geodesy::geodetic_to_ecef};

pub struct DEMGdal {
    pub data: gdal::Dataset,
}
unsafe impl Send for DEMGdal {}
unsafe impl Sync for DEMGdal {}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopernicusDemType {
    Cop30,
    Cop90,
}

impl std::fmt::Display for CopernicusDemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopernicusDemType::Cop30 => write!(f, "COP30"),
            CopernicusDemType::Cop90 => write!(f, "COP90"),
        }
    }
}

impl DEMGdal {
    /// Download a DEM from the OpenTopography API.
    ///
    /// # Arguments
    ///
    /// - `bounds` [min_lat, max_lat, min_lon, max_lon] - The bounding box of the DEM.
    /// - `dem_type` - The type of DEM to download.
    ///
    /// # Returns
    ///
    /// - A new [`DEM`].
    ///
    pub fn download_dem(bounds: [f64; 4], dem_type: CopernicusDemType) -> Self {
        let [min_lat, max_lat, min_lon, max_lon] = bounds;
        let api_key =
            std::env::var("OPENTOPOGRAPHY_API_KEY").expect("OPENTOPOGRAPHY_API_KEY not set");
        let url = format!(
            "https://portal.opentopography.org/API/globaldem?demtype={}&south={}&north={}&west={}&east={}&outputFormat=GTiff&API_Key={}",
            dem_type, min_lat, max_lat, min_lon, max_lon, api_key
        );
        let response = ureq::get(url).call().expect("Failed to download DEM");
        if response.status() == 200 {
            let body = response.into_body();
            let mut reader = body.into_reader();
            let file_path = std::env::temp_dir().join(format!("dem_{}.tif", dem_type));
            let mut dem_file = std::fs::File::create(file_path.clone()).unwrap();
            std::io::copy(&mut reader, &mut dem_file).unwrap();
            Self::open_file(file_path)
        } else {
            panic!("Failed to download DEM");
        }
    }
    /// Opens the DEM file and returns the DEM struct
    pub fn open_file<P: AsRef<Path>>(dem_file_path: P) -> Self {
        let data = gdal::Dataset::open(dem_file_path).expect("Failed to open DEM GeoTIFF file");
        Self { data }
    }

    /// Maps (latitude, longitude) to raster [row, col].
    pub fn get_index_at_lat_lon(&self, lat: f64, lon: f64) -> [usize; 2] {
        let data = &self.data;
        let gt = data.geo_transform().expect("Could not find a Transform");
        let row = (lat - gt[3]) / gt[5];
        let col = (lon - gt[0]) / gt[1];
        let row = row.round() as usize;
        let col = col.round() as usize;
        [row, col]
    }

    /// Maps raster (row, col) to [latitude, longitude].
    pub fn get_lat_lon_at_index(&self, row: usize, col: usize) -> [f64; 2] {
        let data = &self.data;
        let gt = data.geo_transform().expect("Could not find a Transform");
        let row = row as f64;
        let col = col as f64;
        let lat = gt[3] + col * gt[4] + row * gt[5];
        let lon = gt[0] + col * gt[1] + row * gt[2];
        [lat, lon]
    }

    /// Gets the height value at the index (row, col)
    pub fn get_value_at_index(&self, row: usize, col: usize) -> f64 {
        let data = &self.data;
        let band = data
            .rasterband(1)
            .expect("Could not read first raster band.");
        let value = band.read_as::<f32>((col as isize, row as isize), (1, 1), (1, 1), None);
        match value {
            Ok(value) => value[(0, 0)] as f64,
            Err(_) => f64::NAN,
        }
        // value[(0, 0)] as f64
    }

    /// Gets the height value at the coordinates (lat, lon)
    pub fn get_height_at_lat_lon(&self, lat: f64, lon: f64) -> f64 {
        let [row, col] = self.get_index_at_lat_lon(lat, lon);
        self.get_value_at_index(row, col)
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel at (row, col)
    pub fn get_ecef_at_pixel(&self, row: usize, col: usize) -> [f64; 3] {
        let [lat, lon] = self.get_lat_lon_at_index(row, col);
        let height = self.get_value_at_index(row, col);
        let geoid_height = egm_2008::geoid_height(lat, lon).unwrap();
        geodetic_to_ecef(lat, lon, height + geoid_height)
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel at (row, col)
    pub fn get_ecef_at_lat_lon(&self, lat: f64, lon: f64) -> [f64; 3] {
        let height = self.get_height_at_lat_lon(lat, lon);
        let geoid_height = egm_2008::geoid_height(lat, lon).unwrap();
        geodetic_to_ecef(lat, lon, height + geoid_height)
    }

    /// Gets the normal vector at a specific latitude/longitude position using the DEM data.
    /// The normal is estimated by calculating the cross product of two tangent vectors
    /// determined from neighboring points.
    ///
    /// If normal estimation fails (e.g., point or its neighbors are out of bounds),
    /// falls back to the ellipsoid's local_normal.
    ///
    /// # Returns
    ///
    /// A unit normal vector in ECEF coordinates [nx, ny, nz]
    pub fn get_normal_at_lat_lon(&self, lat: f64, lon: f64) -> [f64; 3] {
        // Get index in the DEM grid
        let [row, col] = self.get_index_at_lat_lon(lat, lon);
        let (rows, cols) = self.array_dim();

        // Check if neighbors would be within bounds
        if row == 0 || col == 0 || row >= rows - 1 || col >= cols - 1 {
            // Fall back to ellipsoid normal if too close to edge
            return crate::geodesy::local_normal(lat, lon);
        }

        // Get ECEF coordinates of the center point and neighbors
        let p0 = self.get_ecef_at_pixel(row, col);

        // Get east and north neighbors
        let p_east = self.get_ecef_at_pixel(row, col + 1);
        let p_north = self.get_ecef_at_pixel(row - 1, col); // Row decreases as latitude increases

        // Calculate tangent vectors
        let tangent_east = [p_east[0] - p0[0], p_east[1] - p0[1], p_east[2] - p0[2]];

        let tangent_north = [p_north[0] - p0[0], p_north[1] - p0[1], p_north[2] - p0[2]];

        // Cross product to get normal vector
        let normal = [
            tangent_east[1] * tangent_north[2] - tangent_east[2] * tangent_north[1],
            tangent_east[2] * tangent_north[0] - tangent_east[0] * tangent_north[2],
            tangent_east[0] * tangent_north[1] - tangent_east[1] * tangent_north[0],
        ];

        // Calculate magnitude
        let magnitude =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();

        if magnitude > 1e-10 {
            // Normalize
            let local_normal = [
                normal[0] / magnitude,
                normal[1] / magnitude,
                normal[2] / magnitude,
            ];
            // The reference frame vector in the ECEF frame
            let global_normal = crate::geodesy::local_normal(lat, lon);
            // Perform basis change
            todo!()
        } else {
            // Fall back to ellipsoid normal if calculation failed
            crate::geodesy::local_normal(lat, lon)
        }
    }

    /// Copies the DEM raster data into a new, owned, 2D ndarray and returns it.
    pub fn read_raster_data(&self) -> Array2<f32> {
        let data = &self.data;
        let band = data
            .rasterband(1)
            .expect("Could not read first raster band.");
        let buf: Buffer<f32> = band
            .read_band_as()
            .expect("Could not read band as Buffer<f32>");
        buf.to_array()
            .expect("Could not convert Buffer<f32> to ndarray")
    }

    pub fn lat_lon_iter(&self) -> LatLonIter {
        let (rows, cols) = self.array_dim();
        LatLonIter {
            dem: self,
            i: 0,
            j: 0,
            rows,
            cols,
        }
    }

    pub fn indexed_lat_lon_height(&self) -> IndexedLatLonHeight {
        let (rows, cols) = self.array_dim();
        IndexedLatLonHeight {
            dem: self,
            i: 0,
            j: 0,
            rows,
            cols,
        }
    }

    /// Gets the 3D vertex positions of the DEM.
    /// Useful to build a 3D Mesh of the DEM.
    pub fn vertex_positions(&self) -> Vec<[f32; 3]> {
        let (rows, cols) = self.array_dim();
        let mut vertex_positions = Vec::<[f32; 3]>::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                vertex_positions.push(self.get_ecef_at_pixel(i, j).map(|val| val as f32));
            }
        }
        vertex_positions
    }

    /// Gets a default grayscale color for each triangle in the 3D Mesh.
    /// Useful to build a 3D Mesh of the DEM.
    pub fn vertex_colors(&self) -> Vec<u32> {
        let array = self.read_raster_data();
        let min_val = *array
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_val = *array
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let range = max_val - min_val;
        let vertex_colors: Vec<u32> = array
            .iter()
            .map(|&val| {
                let norm = if range > 0.0 {
                    (val - min_val) / range
                } else {
                    0.0 // flat surface fallback
                };
                let gray = (norm * 255.0).round() as u32;
                (gray << 24) | (gray << 16) | (gray << 8) | 0xFF // 0xRRGGBBAA
            })
            .collect();
        vertex_colors
    }

    /// Gets the corresponding triangle indices in the 3D Mesh.
    /// Useful to build a 3D Mesh of the DEM.
    ///
    /// See also [`vertex_positions`]
    pub fn triangle_indices(&self) -> Vec<[u32; 3]> {
        let (rows, cols) = self.array_dim();
        let mut triangle_indices = Vec::with_capacity((rows - 1) * (cols - 1) * 2);
        for i in 0..(rows - 1) {
            for j in 0..(cols - 1) {
                let a = (i * cols + j) as u32;
                let b = (i * cols + j + 1) as u32;
                let c = ((i + 1) * cols + j) as u32;
                let d = ((i + 1) * cols + j + 1) as u32;
                triangle_indices.push([a, b, d]);
                triangle_indices.push([a, d, c]);
            }
        }
        triangle_indices
    }

    /// The underlying array dimensions, i.e.
    /// the pair `(rows, columns)`.
    ///
    /// The underlying data array is in Row-major ("C-like") logical order.
    pub fn array_dim(&self) -> (usize, usize) {
        let band = self
            .data
            .rasterband(1)
            .expect("Could not read first raster band.");
        let (columns, rows) = band.size();
        (rows, columns)
    }

    /// Gets the [lat, lon] of each of these 4 corners of the DEM extent:
    ///
    /// Upper left, Lower left, Lower right, Upper right
    pub fn corners_lat_lon(&self) -> [[f64; 2]; 4] {
        let data = &self.data;
        let (width, height) = data.raster_size();
        let gt = data.geo_transform().unwrap(); // [gt0, gt1, gt2, gt3, gt4, gt5]

        let (gt0, gt1, gt2, gt3, gt4, gt5) = (gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]);

        // Pixel-to-geo function
        let to_geo = |col: isize, row: isize| {
            let lon = gt0 + col as f64 * gt1 + row as f64 * gt2;
            let lat = gt3 + col as f64 * gt4 + row as f64 * gt5;
            [lat, lon]
        };

        [
            (0, 0),                            // Upper Left
            (0, height as isize),              // Lower Left
            (width as isize, height as isize), // Lower Right
            (width as isize, 0),               // Upper Right
        ]
        .map(|(col, row)| to_geo(col, row))
    }
}

pub struct IndexedLatLonHeight<'a> {
    dem: &'a DEMGdal,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
}

impl Iterator for IndexedLatLonHeight<'_> {
    type Item = (usize, usize, f64, f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.rows {
            return None;
        }

        let [lat, lon] = self.dem.get_lat_lon_at_index(self.i, self.j);
        let height = self.dem.get_height_at_lat_lon(lat, lon);
        let result = Some((self.i, self.j, lat, lon, height));

        self.j += 1;
        if self.j >= self.cols {
            self.j = 0;
            self.i += 1;
        }

        result
    }
}

pub struct LatLonIter<'a> {
    dem: &'a DEMGdal,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
}

impl Iterator for LatLonIter<'_> {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.rows {
            return None;
        }

        let [lat, lon] = self.dem.get_lat_lon_at_index(self.i, self.j);
        let result = Some((lat, lon));

        self.j += 1;
        if self.j >= self.cols {
            self.j = 0;
            self.i += 1;
        }

        result
    }
}

#[cfg(test)]
mod manual_tests {
    use super::DEMGdal;
    use gdal::{Dataset, Metadata};

    #[test]
    fn dem_gdal() {
        let dataset = Dataset::open("dem.tif").unwrap();
        println!("Dataset description: {}", dataset.description().unwrap());
        let band_count = dataset.raster_count();
        println!("Number of bands: {band_count}");
        // Beware! In GDAL, band indexes are 1-based!
        for i in 1..=band_count {
            println!("  Band {i}");
            let band = dataset.rasterband(i).unwrap();
            // Depending on the file, the description field may be the empty string :(
            println!("    Description: '{}'", band.description().unwrap());
            // In GDAL, all no-data values are coerced to floating point types, regardless of the
            // underlying pixel type.
            println!("    No-data value: {:?}", band.no_data_value());
            println!("    Pixel data type: {}", band.band_type());
            // Scale and offset are often used with integral pixel types to convert between pixel value
            // to some physical unit (e.g. watts per square meter per steradian)
            println!("    Scale: {:?}", band.scale());
            println!("    Offset: {:?}", band.offset());
            println!("    Size: {:?}", band.size());
            let rv = band
                .read_as::<f32>((0, 0), band.size(), band.size(), None)
                .unwrap();
            // `Rasterband::read_as` returns a `Buffer` struct, which contains the shape of the output
            // `(cols, rows)` and a `Vec<_>` containing the pixel values.
            println!("    Data size: {:?}", rv.shape());
            // println!("    Data values: {:?}", rv.data());
        }
    }

    #[test]
    #[ignore]
    fn dem_extent() {
        let dem = DEMGdal::open_file("dem.tif");
        println!("DEM extent = {:?}", dem.corners_lat_lon());
        let [lat, lon] = [19.41882131107135, -99.1144313400927];
        let xyz = dem.get_ecef_at_lat_lon(lat, lon);
        println!("pos({lat}, {lon}) = {:?}", xyz);
    }
    #[test]
    #[ignore]
    fn dem_mesh_test() {
        let dem = DEMGdal::open_file("dem.tif");
        let rec = rerun::RecordingStreamBuilder::new("dem_mesh_test")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let (rows, cols) = dem.array_dim();

        let vertex_positions: Vec<[f32; 3]> = dem.vertex_positions();
        let vertex_normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; rows * cols];
        let vertex_colors: Vec<u32> = dem.vertex_colors();
        let triangle_indices = dem.triangle_indices();
        rec.log(
            "dem_mesh3d",
            &rerun::Mesh3D::new(vertex_positions)
                .with_vertex_normals(vertex_normals)
                .with_vertex_colors(vertex_colors)
                .with_triangle_indices(triangle_indices),
        )
        .unwrap();
    }
}
