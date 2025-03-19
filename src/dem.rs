use geotiff::GeoTiff;

/// A digital elevation model (DEM).
pub struct ElevationModel {
    data: GeoTiff,
}

impl ElevationModel {
    pub fn get_height(&self, x: f64, y: f64) -> f64 {
        todo!()
    }
}
