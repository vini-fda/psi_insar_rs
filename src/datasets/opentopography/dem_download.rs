//! OpenTopography DEM Download
//!
//! Downloads Copernicus DEM rasters from the OpenTopography global DEM API.

use crate::dem::DEM;

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
pub fn download_dem(bounds: [f64; 4], dem_type: CopernicusDemType) -> DEM {
    let [min_lat, max_lat, min_lon, max_lon] = bounds;
    let api_key = std::env::var("OPENTOPOGRAPHY_API_KEY").expect("OPENTOPOGRAPHY_API_KEY not set");
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
        DEM::open_file(file_path)
    } else {
        panic!("Failed to download DEM");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Needs to download external data"]
    fn simple_test() {
        let bounds = [
            19.28241180526043,
            19.67909634636101,
            -99.4141148418833,
            -98.53735764573854,
        ];
        let dem_type = CopernicusDemType::Cop90;
        let dem = download_dem(bounds, dem_type);
        println!("DEM size: {} x {}", dem.rows(), dem.cols());
    }
}
