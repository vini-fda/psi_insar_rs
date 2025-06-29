use geotiff::raster_data::RasterData;
use ndarray::Array2;
use std::path::Path;

use crate::{egm_2008, geodesy::geodetic_to_ecef};

pub struct DEM {
    /// Geographical transform
    geo_transform: [f64; 4],
    /// Inverse geographical transform
    inv_geo_transform: [f64; 4],
    rows: usize,
    cols: usize,
    /// Height data for each pixel in the DEM
    height: Vec<f32>,
}

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

impl DEM {
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
            "https://portal.opentopography.org/API/globaldem?demtype={dem_type}&south={min_lat}&north={max_lat}&west={min_lon}&east={max_lon}&outputFormat=GTiff&API_Key={api_key}"
        );
        let response = ureq::get(url).call().expect("Failed to download DEM");
        if response.status() == 200 {
            let body = response.into_body();
            let mut reader = body.into_reader();
            let file_path = std::env::temp_dir().join(format!("dem_{dem_type}.tif"));
            let mut dem_file = std::fs::File::create(file_path.clone()).unwrap();
            std::io::copy(&mut reader, &mut dem_file).unwrap();
            Self::open_file(file_path)
        } else {
            panic!("Failed to download DEM");
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn open_file<P: AsRef<Path>>(path: P) -> Self {
        let file = std::fs::File::open(path).expect("Could not open GeoTIFF file.");
        let data = geotiff::GeoTiff::read(file).expect("Could not read GeoTIFF.");
        let transform = data.coordinate_transform.unwrap();
        let (geo_transform, inv_geo_transform) = match transform {
            geotiff::coordinate_transform::CoordinateTransform::TiePointAndPixelScale {
                raster_point,
                model_point,
                pixel_scale,
            } => (
                [
                    pixel_scale.x,
                    -raster_point.x * pixel_scale.x + model_point.x,
                    -pixel_scale.y,
                    raster_point.y * pixel_scale.y + model_point.y,
                ],
                [
                    (1.0 / pixel_scale.x),
                    -model_point.x / pixel_scale.x + raster_point.x,
                    (-1.0 / pixel_scale.y),
                    model_point.y / pixel_scale.y + raster_point.y,
                ],
            ),
            _ => panic!("Unsupported transform."),
        };
        let height = match data.raster_data {
            RasterData::F32(data) => data,
            _ => panic!("Could not read raster data."),
        };
        let rows = data.raster_height;
        let cols = data.raster_width;
        Self {
            geo_transform,
            inv_geo_transform,
            rows,
            cols,
            height,
        }
    }

    /// Maps (latitude, longitude) to raster [row, col].
    pub fn get_pixel_at_lat_lon(&self, lat: f64, lon: f64) -> [usize; 2] {
        let igt = self.inv_geo_transform;
        let col = lon * igt[0] + igt[1];
        let row = lat * igt[2] + igt[3];
        let col = col.round() as usize;
        let row = row.round() as usize;
        [row, col]
    }

    /// Maps raster (row, col) to [latitude, longitude].
    pub fn get_lat_lon_at_pixel(&self, row: usize, col: usize) -> [f64; 2] {
        let gt = self.geo_transform;
        let col = col as f64;
        let row = row as f64;
        let lon = col * gt[0] + gt[1];
        let lat = row * gt[2] + gt[3];
        [lat, lon]
    }

    /// Gets the height value at the given index
    pub fn get_value_at_index(&self, index: usize) -> f32 {
        self.height[index]
    }

    /// Gets the height value at the pixel (row, col)
    pub fn get_value_at_pixel(&self, row: usize, col: usize) -> f32 {
        let index = row * self.cols + col;
        self.get_value_at_index(index)
    }

    /// Gets the height value at the coordinates (lat, lon)
    pub fn get_height_at_lat_lon(&self, lat: f64, lon: f64) -> f32 {
        let [row, col] = self.get_pixel_at_lat_lon(lat, lon);
        self.get_value_at_pixel(row, col)
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel (row, col)
    pub fn get_ecef_at_pixel(&self, row: usize, col: usize) -> [f64; 3] {
        let height = self.get_value_at_pixel(row, col);
        let [lat, lon] = self.get_lat_lon_at_pixel(row, col);
        let geoid_height = egm_2008::geoid_height(lat, lon).unwrap();
        geodetic_to_ecef(lat, lon, height as f64 + geoid_height)
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel (row, col)
    pub fn get_ecef_at_lat_lon(&self, lat: f64, lon: f64) -> [f64; 3] {
        let height = self.get_height_at_lat_lon(lat, lon);
        let geoid_height = egm_2008::geoid_height(lat, lon).unwrap();
        geodetic_to_ecef(lat, lon, height as f64 + geoid_height)
    }

    /// Returns an iterator over the DEM's (lat, lon) elements, in memory order (i.e. row-major ascending order).
    pub fn lat_lon_iter(&self) -> LatLonIter {
        LatLonIter {
            dem: self,
            i: 0,
            j: 0,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Returns an iterator over the DEM's (lat, lon, height) elements, in memory order (i.e. row-major ascending order).
    pub fn lat_lon_height_iter(&self) -> LatLonHeightIter {
        LatLonHeightIter {
            dem: self,
            i: 0,
            j: 0,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Returns an iterator over the DEM's (i, j, lat, lon, height) elements, in memory order (i.e. row-major ascending order).
    pub fn indexed_lat_lon_height_iter(&self) -> IndexedLatLonHeightIter {
        IndexedLatLonHeightIter {
            dem: self,
            i: 0,
            j: 0,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Gets the [lat, lon] of each of these 4 corners of the DEM extent:
    ///
    /// Upper left, Lower left, Lower right, Upper right
    pub fn corners_lat_lon(&self) -> [[f64; 2]; 4] {
        let rows = self.rows;
        let cols = self.cols;
        [
            (0, 0),       // Upper Left
            (0, rows),    // Lower Left
            (cols, rows), // Lower Right
            (cols, 0),    // Upper Right
        ]
        .map(|(col, row)| self.get_lat_lon_at_pixel(row, col))
    }

    /// Gets the [lat, lon] of each of these 4 corners of the DEM extent:
    ///
    /// Upper left, Lower left, Lower right, Upper right, Upper left (again)
    pub fn closed_corners_lat_lon(&self) -> [[f64; 2]; 5] {
        let rows = self.rows;
        let cols = self.cols;
        [
            (0, 0),       // Upper Left
            (0, rows),    // Lower Left
            (cols, rows), // Lower Right
            (cols, 0),    // Upper Right
            (0, 0),       // Upper Left
        ]
        .map(|(col, row)| self.get_lat_lon_at_pixel(row, col))
    }

    /// Gets the 3D vertex positions of the DEM.
    /// Useful to build a 3D Mesh of the DEM.
    pub fn vertex_positions(&self) -> Vec<[f32; 3]> {
        let rows = self.rows;
        let cols = self.cols;
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
        let min_val = self
            .height
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_val = self
            .height
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let range = max_val - min_val;

        let vertex_colors: Vec<u32> = if range > 0.0 {
            self.height
                .iter()
                .map(|&val| {
                    let norm = (val - min_val) / range;
                    let gray = (norm * 255.0).round() as u32;
                    (gray << 24) | (gray << 16) | (gray << 8) | 0xFF // 0xRRGGBBAA
                })
                .collect()
        } else {
            vec![0; self.height.len()]
        };
        vertex_colors
    }

    /// Gets the corresponding triangle indices in the 3D Mesh.
    /// Useful to build a 3D Mesh of the DEM.
    ///
    /// See also [`vertex_positions`]
    pub fn triangle_indices(&self) -> Vec<[u32; 3]> {
        let rows = self.rows;
        let cols = self.cols;
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

    pub fn array(&self) -> Array2<f32> {
        let rows = self.rows;
        let cols = self.cols;
        Array2::from_shape_vec((rows, cols), self.height.clone()).unwrap()
    }
}

/// An iterator over the DEM's (lat, lon) elements, in memory order (i.e. row-major ascending order).
pub struct LatLonIter<'a> {
    dem: &'a DEM,
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

        let [lat, lon] = self.dem.get_lat_lon_at_pixel(self.i, self.j);
        let result = Some((lat, lon));

        self.j += 1;
        if self.j >= self.cols {
            self.j = 0;
            self.i += 1;
        }

        result
    }
}

/// An iterator over the DEM's (lat, lon, height) elements, in memory order (i.e. row-major ascending order).
pub struct LatLonHeightIter<'a> {
    dem: &'a DEM,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
}

impl Iterator for LatLonHeightIter<'_> {
    type Item = (f64, f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.rows {
            return None;
        }

        let [lat, lon] = self.dem.get_lat_lon_at_pixel(self.i, self.j);
        let height = self.dem.get_height_at_lat_lon(lat, lon);
        let result = Some((lat, lon, height as f64));

        self.j += 1;
        if self.j >= self.cols {
            self.j = 0;
            self.i += 1;
        }

        result
    }
}

/// An iterator over the DEM's (i, j, lat, lon, height) elements, in memory order (i.e. row-major ascending order).
pub struct IndexedLatLonHeightIter<'a> {
    dem: &'a DEM,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
}

impl Iterator for IndexedLatLonHeightIter<'_> {
    type Item = (usize, usize, f64, f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.rows {
            return None;
        }

        let [lat, lon] = self.dem.get_lat_lon_at_pixel(self.i, self.j);
        let height = self.dem.get_height_at_lat_lon(lat, lon);
        let result = Some((self.i, self.j, lat, lon, height as f64));

        self.j += 1;
        if self.j >= self.cols {
            self.j = 0;
            self.i += 1;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_test() {
        let bounds = [
            19.28241180526043,
            19.67909634636101,
            -99.4141148418833,
            -98.53735764573854,
        ];
        let dem_type = CopernicusDemType::Cop90;
        let dem = DEM::download_dem(bounds, dem_type);
        println!("transform = {:?}", dem.geo_transform);
        println!("inv transform = {:?}", dem.inv_geo_transform);
    }

    #[test]
    #[ignore]
    fn dem_mesh_test() {
        let dem = DEM::open_file("dem.tif");
        let rec = rerun::RecordingStreamBuilder::new("dem_mesh_test")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let rows = dem.rows;
        let cols = dem.cols;

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
