use std::path::Path;

use geotiff::{
    GeoTiff,
    raster_data::{RasterData, RasterValue},
};
use ndarray::Array2;

use crate::geodesy::geodetic_to_ecef;

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
    pub fn get_lon_lat_at_pixel(&self, row: usize, col: usize) -> (f64, f64) {
        let data = &self.data;
        let transform = data.coordinate_transform.as_ref().unwrap();
        let coord = transform.transform_to_model(&geo_types::Coord {
            x: row as f64,
            y: col as f64,
        });
        coord.x_y()
    }

    /// Gets the height value at pixel (row, col)
    pub fn get_value_at_pixel(&self, row: usize, col: usize) -> f32 {
        match self.data.get_value_at_pixel(row, col, 0) {
            Some(RasterValue::F32(value)) => value,
            _ => panic!(),
        }
    }

    /// Gets the Earth-Centered Earth-Fixed (ECEF) cartesian coordinates of the pixel at (row, col)
    pub fn get_ecef_at_pixel(&self, row: usize, col: usize) -> (f32, f32, f32) {
        let (lon, lat) = self.get_lon_lat_at_pixel(row, col);
        let height = self.get_value_at_pixel(row, col);
        geodetic_to_ecef(lat as f32, lon as f32, height)
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
}
