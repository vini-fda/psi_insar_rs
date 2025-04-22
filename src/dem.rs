use std::path::Path;

use geotiff::{
    GeoTiff,
    raster_data::{RasterData, RasterValue},
};
use ndarray::Array2;

use crate::{egm_2008, geodesy::geodetic_to_ecef};

pub struct DEM {
    pub data: GeoTiff,
}

impl DEM {
    /// Opens the DEM file and returns the DEM struct
    pub fn open_file<P: AsRef<Path>>(dem_file_path: P) -> Self {
        let geotiff_file =
            std::fs::File::open(dem_file_path).expect("Failed to open DEM TIFF file");
        let data = GeoTiff::read(geotiff_file).expect("Failed to parse TIFF file");
        Self { data }
    }

    /// Maps raster (row, col) to longitude and latitude.
    pub fn get_lon_lat_at_index(&self, row: usize, col: usize) -> (f64, f64) {
        let data = &self.data;
        let transform = data.coordinate_transform.as_ref().unwrap();
        let coord = transform.transform_to_model(&geo_types::Coord {
            x: col as f64,
            y: row as f64,
        });
        coord.x_y()
    }

    /// Gets the height value at the index (row, col)
    pub fn get_value_at_index(&self, row: usize, col: usize) -> f32 {
        match self.data.get_value_at_pixel(col, row, 0) {
            Some(RasterValue::F32(value)) => value,
            _ => panic!(),
        }
    }

    /// Gets the height value at the coordinates (lat, lon)
    pub fn get_height_at_lat_lon(&self, lat: f64, lon: f64) -> f32 {
        let coord = geo_types::Coord { x: lon, y: lat };
        match self.data.get_value_at(&coord, 0) {
            Some(RasterValue::F32(value)) => value,
            _ => panic!(),
        }
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel at (row, col)
    pub fn get_ecef_at_pixel(&self, row: usize, col: usize) -> [f32; 3] {
        let (lon, lat) = self.get_lon_lat_at_index(row, col);
        let height = self.get_value_at_index(row, col);
        let geoid_height = egm_2008::geoid_height(lat, lon).unwrap();
        geodetic_to_ecef(lat as f32, lon as f32, height + geoid_height as f32)
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel at (row, col)
    pub fn get_ecef_at_lat_lon(&self, lat: f64, lon: f64) -> [f32; 3] {
        let height = self.get_height_at_lat_lon(lat, lon);
        let geoid_height = egm_2008::geoid_height(lat, lon).unwrap();
        geodetic_to_ecef(lat as f32, lon as f32, height + geoid_height as f32)
    }

    /// Copies the DEM raster data into a new, owned, 2D ndarray and returns it.
    pub fn read_raster_data(&self) -> Array2<f32> {
        let data = &self.data;

        match data.raster_data {
            RasterData::F32(ref vec) => {
                Array2::from_shape_vec((data.raster_height, data.raster_width), vec.clone())
                    .unwrap()
            }
            _ => panic!(),
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
                vertex_positions.push(self.get_ecef_at_pixel(i, j));
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
        (self.data.raster_height, self.data.raster_width)
    }
}

pub struct IndexedLatLonHeight<'a> {
    dem: &'a DEM,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
}

impl<'a> Iterator for IndexedLatLonHeight<'a> {
    type Item = (usize, usize, f64, f64, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.rows {
            return None;
        }

        let (lon, lat) = self.dem.get_lon_lat_at_index(self.i, self.j);
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

#[cfg(test)]
mod manual_tests {
    use super::DEM;
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
        let dem = DEM::open_file("dem.tif");
        println!("DEM extent = {:?}", dem.data.model_extent());
        let [lat, lon] = [19.41882131107135, -99.1144313400927];
        let xyz = dem.get_ecef_at_lat_lon(lat, lon);
        println!("pos({lat}, {lon}) = {:?}", xyz);
    }
    #[test]
    #[ignore]
    fn dem_mesh_test() {
        let dem = DEM::open_file("dem.tif");
        let rec = rerun::RecordingStreamBuilder::new("dem_mesh_test")
            .connect_tcp()
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
