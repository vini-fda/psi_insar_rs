use crate::download_orbit::CDSEOrbitDownloader;
use crate::granule_id::{Sentinel1GranuleId, Sentinel1TIFFFileName};
use crate::metadata::annotation_xml::SlcProductAnnotation;
use crate::metadata::calibration_xml::Calibration;
use crate::metadata::noise_xml::Noise;
use crate::satellite_orbit::{ContinuousOrbitalStateHistory, OrbitalStateHistory};
use crate::slc_image::SlcImage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::AsRef;
use std::path::{Path, PathBuf};

/// Represents a geographic point in WGS84 coordinates
///
/// Source: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-geometry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    pub latitude: f64,
    pub longitude: f64,
    pub height: f64,
}

/// Represents a geographic bounding box in WGS84 coordinates
///
/// Source: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-geometry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoBoundingBox {
    pub min_latitude: f64,
    pub min_longitude: f64,
    pub max_latitude: f64,
    pub max_longitude: f64,
}

/// Represents the metadata for a burst in TOPS mode.
///
/// The TOPS (Terrain Observation with Progressive Scans) acquisition mode is used by Sentinel-1
/// to acquire data over a wide swath with enhanced image performance.
///
/// Sources:
/// - https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/tops-processing
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstMetadata {
    pub burst_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub center_time: DateTime<Utc>,
    pub azimuth_time_interval: f64,
    pub range_sampling_rate: f64,
    pub lines: usize,
    pub samples: usize,
    pub first_valid_sample: usize,
    pub last_valid_sample: usize,
    pub first_valid_line: usize,
    pub last_valid_line: usize,
    pub bounding_box: GeoBoundingBox,
    pub doppler_centroid: Vec<f64>,
    pub fm_rate: Vec<f64>,
}

/// Represents the metadata for a subswath in TOPS mode.
///
/// In IW mode, the data is collected in three sub-swaths using the TOPS technique.
/// Each sub-swath contains a series of bursts, where each burst has been processed as a separate SLC image.
///
/// Sources:
/// - https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/interferometric-wide-swath
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubswathMetadata {
    pub subswath_id: String,
    pub polarization: String,
    pub bursts: Vec<BurstMetadata>,
    pub radar_frequency: f64,
    pub azimuth_steering_rate: f64,
    pub lines: usize,
    pub samples: usize,
    pub bounding_box: GeoBoundingBox,
    pub incidence_angle_mid_swath: f64,
}

/// Represents the orbit state vectors.
///
/// Orbit state vectors provide the position and velocity of the satellite at specific times.
/// They are used for accurate geolocation and InSAR processing.
///
/// Sources:
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
/// - https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1/orbit-accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitStateVector {
    pub time: DateTime<Utc>,
    pub position_x: f64,
    pub position_y: f64,
    pub position_z: f64,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub velocity_z: f64,
}

/// Represents the orbit information.
///
/// Sentinel-1 orbit information includes details about the orbit type (e.g., precise, restituted),
/// orbit number, pass direction (ascending or descending), and state vectors.
///
/// Sources:
/// - https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1/orbit-accuracy
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitInformation {
    pub orbit_number: u32,
    pub orbit_type: String,     // POEORB, RESORB, etc.
    pub pass_direction: String, // ASCENDING or DESCENDING
    pub orbit_state_vectors: Vec<OrbitStateVector>,
}

/// Represents the metadata for a Sentinel-1 SLC product.
///
/// Single Look Complex (SLC) products consist of focused SAR data, geo-referenced using orbit
/// and attitude data from the satellite, and provided in slant-range geometry.
///
/// Sources:
/// - https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/single-look-complex
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentinel1SlcMetadata {
    pub granule_id: Sentinel1GranuleId,
    pub acquisition_date: DateTime<Utc>,
    pub orbit_information: OrbitInformation,
    pub subswaths: Vec<SubswathMetadata>,
    pub product_info: HashMap<String, String>,
    pub processing_info: HashMap<String, String>,
    pub instrument_info: HashMap<String, String>,
    pub platform_info: HashMap<String, String>,
    pub bounding_box: GeoBoundingBox,
}

/// Represents a single burst of Sentinel-1 SLC data.
///
/// A burst is the basic acquisition unit in TOPS mode. Each burst contains SAR data acquired
/// during a single sweep of the antenna beam from back to fore.
///
/// Sources:
/// - https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/tops-processing
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
pub struct Sentinel1SlcProduct {
    pub metadata: SlcProductAnnotation,
    pub calibration: Calibration,
    pub noise: Noise,
    pub granule_id: Sentinel1TIFFFileName,
    pub data: SlcImage,
}

impl Sentinel1SlcProduct {
    /// Load the burst from a directory, choosing the first .tiff file in the measurement directory.
    pub fn load_first_from_directory(directory: impl AsRef<Path>) -> Result<Self, String> {
        let directory = directory.as_ref();
        let measurement_dir = directory.join("measurement");

        // Check if directory exists
        if !measurement_dir.exists() {
            return Err(format!(
                "Measurement directory does not exist: {}",
                measurement_dir.display()
            ));
        }

        // Find file matching pattern
        let entries = std::fs::read_dir(&measurement_dir)
            .map_err(|e| format!("Failed to read measurement directory: {e}"))?;

        let mut granule_str = None;
        for entry_result in entries {
            let entry = entry_result.map_err(|e| format!("Failed to read directory entry: {e}"))?;
            let file_name = entry.file_name();
            if let Some(file_name) = file_name.to_str()
                && (file_name.ends_with(".tiff") || file_name.ends_with(".tif"))
            {
                let removed_extension = file_name.split(".").next().unwrap();
                granule_str = Some(removed_extension.to_string());
            }
        }

        if granule_str.is_none() {
            return Err("No TIFF file found in measurement directory".to_string());
        }

        let granule_str = granule_str.unwrap();

        let granule_id =
            Sentinel1TIFFFileName::parse(&granule_str).map_err(|e| format!("ERROR: {e}"))?;

        // Construct paths to necessary files
        let calibration_path = Self::find_calibration_xml(directory, &granule_str)?;
        let noise_path = Self::find_noise_xml(directory, &granule_str)?;
        let annotation_path = Self::find_annotation_xml(directory, &granule_str)?;
        let measurement_path = Self::find_measurement_tiff(directory, &granule_str)?;

        // Parse calibration XML
        let calibration_xml_content = std::fs::read_to_string(&calibration_path)
            .expect("Failed to read calibration XML file");
        let calibration: Calibration = quick_xml::de::from_str(&calibration_xml_content)
            .expect("Failed to parse calibration XML");
        // Parse noise XML
        let noise_xml_content =
            std::fs::read_to_string(&noise_path).expect("Failed to read noise XML file");
        let noise: Noise =
            quick_xml::de::from_str(&noise_xml_content).expect("Failed to parse noise XML");
        // Parse annotation XML to get annotation metadata
        let annotation_xml_content =
            std::fs::read_to_string(&annotation_path).expect("Failed to read annotation XML file");
        let metadata: SlcProductAnnotation = quick_xml::de::from_str(&annotation_xml_content)
            .expect("Failed to parse annotation XML");

        // Load data from GeoTiff
        let data = SlcImage::new(&measurement_path);

        // Create the SlcBurst instance
        Ok(Sentinel1SlcProduct {
            calibration,
            noise,
            metadata,
            granule_id,
            data,
        })
    }

    /// Finds the calibration XML file for a given slug in the calibration directory.
    ///
    /// # Arguments
    ///
    /// * `directory` - Base directory containing the SAFE product structure
    /// * `slug` - Identifier part of the filename (e.g. "s1a-iw3-slc-vv-20151022t122546-20151022t122549-008265-00ba51-001")
    ///
    /// # Returns
    ///
    /// * `Ok(PathBuf)` - Path to the calibration XML file if found
    /// * `Err(String)` - Error message if the file couldn't be found or there was an I/O error
    fn find_calibration_xml(directory: impl AsRef<Path>, slug: &str) -> Result<PathBuf, String> {
        let directory = directory.as_ref();
        // Find the calibration XML file
        let calibration_dir = directory.join("annotation").join("calibration");
        let pattern = format!("calibration-{slug}.xml");

        // Check if directory exists
        if !calibration_dir.exists() {
            return Err(format!(
                "Calibration directory does not exist: {}",
                calibration_dir.display()
            ));
        }

        // Find file matching pattern
        let entries = std::fs::read_dir(&calibration_dir)
            .map_err(|e| format!("Failed to read calibration directory: {e}"))?;

        for entry_result in entries {
            let entry = entry_result.map_err(|e| format!("Failed to read directory entry: {e}"))?;

            if let Some(file_name) = entry.file_name().to_str()
                && file_name == pattern
            {
                return Ok(entry.path());
            }
        }

        // File not found - return a meaningful error
        Err(format!(
            "Calibration file 'calibration-{}.xml' not found in {}",
            slug,
            calibration_dir.display()
        ))
    }

    /// Finds the noise XML file for a given slug in the calibration directory.
    ///
    /// # Arguments
    ///
    /// * `directory` - Base directory containing the SAFE product structure
    /// * `slug` - Identifier part of the filename (e.g. "s1a-iw3-slc-vv-20151022t122546-20151022t122549-008265-00ba51-001")
    ///
    /// # Returns
    ///
    /// * `Ok(PathBuf)` - Path to the noise XML file if found
    /// * `Err(String)` - Error message if the file couldn't be found or there was an I/O error
    fn find_noise_xml(directory: impl AsRef<Path>, slug: &str) -> Result<PathBuf, String> {
        let directory = directory.as_ref();
        // Find the noise XML file
        let noise_dir = directory.join("annotation").join("calibration");
        let pattern = format!("noise-{slug}.xml");

        // Check if directory exists
        if !noise_dir.exists() {
            return Err(format!(
                "Calibration directory does not exist: {}",
                noise_dir.display()
            ));
        }

        // Find file matching pattern
        let entries = std::fs::read_dir(&noise_dir)
            .map_err(|e| format!("Failed to read calibration directory: {e}"))?;

        for entry_result in entries {
            let entry = entry_result.map_err(|e| format!("Failed to read directory entry: {e}"))?;

            if let Some(file_name) = entry.file_name().to_str()
                && file_name == pattern
            {
                return Ok(entry.path());
            }
        }

        // File not found - return a meaningful error
        Err(format!(
            "Noise file 'noise-{}.xml' not found in {}",
            slug,
            noise_dir.display()
        ))
    }

    /// Finds the annotation XML file for a given slug in the annotation directory.
    ///
    /// # Arguments
    ///
    /// * `directory` - Base directory containing the SAFE product structure
    /// * `slug` - Identifier part of the filename (e.g. "s1a-iw3-slc-vv-20151022t122546-20151022t122549-008265-00ba51-001")
    ///
    /// # Returns
    ///
    /// * `Ok(PathBuf)` - Path to the annotation XML file if found
    /// * `Err(String)` - Error message if the file couldn't be found or there was an I/O error
    fn find_annotation_xml(directory: impl AsRef<Path>, slug: &str) -> Result<PathBuf, String> {
        let directory = directory.as_ref();
        // Find the annotation XML file
        let annotation_dir = directory.join("annotation");
        let pattern = format!("{slug}.xml");

        // Check if directory exists
        if !annotation_dir.exists() {
            return Err(format!(
                "Annotation directory does not exist: {}",
                annotation_dir.display()
            ));
        }

        // Find file matching pattern
        let entries = std::fs::read_dir(&annotation_dir)
            .map_err(|e| format!("Failed to read annotation directory: {e}"))?;

        for entry_result in entries {
            let entry = entry_result.map_err(|e| format!("Failed to read directory entry: {e}"))?;

            if let Some(file_name) = entry.file_name().to_str()
                && file_name == pattern
            {
                return Ok(entry.path());
            }
        }

        // File not found - return a meaningful error
        Err(format!(
            "Annotation file '{}.xml' not found in {}",
            slug,
            annotation_dir.display()
        ))
    }

    /// Finds the measurement TIFF file for a given slug in the measurement directory.
    ///
    /// # Arguments
    ///
    /// * `directory` - Base directory containing the SAFE product structure
    /// * `slug` - Identifier part of the filename (e.g. "s1a-iw3-slc-vv-20151022t122546-20151022t122549-008265-00ba51-001")
    ///
    /// # Returns
    ///
    /// * `Ok(PathBuf)` - Path to the measurement TIFF file if found
    /// * `Err(String)` - Error message if the file couldn't be found or there was an I/O error
    fn find_measurement_tiff(directory: impl AsRef<Path>, slug: &str) -> Result<PathBuf, String> {
        // Find the measurement TIFF file
        let directory = directory.as_ref();
        let measurement_dir = directory.join("measurement");
        let pattern = format!("{slug}.tiff"); // Note: Using .tiff extension
        let pattern_alt = format!("{slug}.tif"); // Alternative .tif extension

        // Check if directory exists
        if !measurement_dir.exists() {
            return Err(format!(
                "Measurement directory does not exist: {}",
                measurement_dir.display()
            ));
        }

        // Find file matching pattern
        let entries = std::fs::read_dir(&measurement_dir)
            .map_err(|e| format!("Failed to read measurement directory: {e}"))?;

        for entry_result in entries {
            let entry = entry_result.map_err(|e| format!("Failed to read directory entry: {e}"))?;

            if let Some(file_name) = entry.file_name().to_str() {
                // Check for both possible TIFF extensions
                if file_name == pattern || file_name == pattern_alt {
                    return Ok(entry.path());
                }
            }
        }

        // File not found - return a meaningful error
        Err(format!(
            "Measurement file '{}.tif(f)' not found in {}",
            slug,
            measurement_dir.display()
        ))
    }

    pub fn orbital_state_history(&self) -> OrbitalStateHistory {
        let orbit_list = &self.metadata.general_annotation.orbit_list;
        OrbitalStateHistory::from(orbit_list)
    }

    pub fn continuous_orbital_state_history(&self) -> ContinuousOrbitalStateHistory {
        let orbit_list = &self.metadata.general_annotation.orbit_list;
        let osh = OrbitalStateHistory::from(orbit_list);
        let t_start = self
            .metadata
            .image_annotation
            .image_information
            .product_first_line_utc_time;
        ContinuousOrbitalStateHistory::from_osh(&osh, t_start, &self.metadata)
    }

    pub fn precise_orbital_state_history(&self) -> ContinuousOrbitalStateHistory {
        let mission = self.metadata.ads_header.mission_id;
        let start = self.metadata.ads_header.start_time;
        let end = self.metadata.ads_header.stop_time;
        let poe_orbit = CDSEOrbitDownloader::new().search_and_download(mission, start, end);

        ContinuousOrbitalStateHistory::from_poe_timeframe(poe_orbit, start, end, &self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Needs to open external files"]
    fn test_load_slc_burst() {
        // root directory
        let root = PathBuf::from(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        );
        let _ =
            Sentinel1SlcProduct::load_first_from_directory(&root).expect("Failed to load burst");
    }
}
