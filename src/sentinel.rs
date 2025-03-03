use crate::granule_id::Sentinel1GranuleId;
use chrono::{DateTime, Utc};
use ndarray::{Array2, Array3};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Represents a geographic point in WGS84 coordinates
///
/// Source: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-geometry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    pub latitude: f64,
    pub longitude: f64,
    pub height: Option<f64>,
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

/// Represents a single burst of SLC data.
///
/// A burst is the basic acquisition unit in TOPS mode. Each burst contains SAR data acquired
/// during a single sweep of the antenna beam from back to fore.
///
/// Sources:
/// - https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/tops-processing
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone)]
pub struct SlcBurst {
    pub metadata: BurstMetadata,
    pub data: Array2<Complex32>,
}

/// Represents a subswath containing multiple bursts.
///
/// A subswath is a portion of the total swath width. Sentinel-1 IW mode consists of
/// three subswaths (IW1, IW2, IW3), each containing multiple bursts.
///
/// Sources:
/// - https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/interferometric-wide-swath
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone)]
pub struct SlcSubswath {
    pub metadata: SubswathMetadata,
    pub bursts: Vec<SlcBurst>,
}

/// Represents a complete Sentinel-1 SLC product.
///
/// A Sentinel-1 SLC product contains complex-valued SAR imagery preserving both amplitude and phase information.
/// It includes subswaths, each containing multiple bursts, along with metadata and orbit information.
///
/// Sources:
/// - https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/single-look-complex
/// - https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Product-Specification
#[derive(Debug, Clone)]
pub struct Sentinel1SlcProduct {
    pub metadata: Sentinel1SlcMetadata,
    pub subswaths: Vec<SlcSubswath>,
    pub directory_path: PathBuf,
}

impl Sentinel1SlcProduct {
    /// Loads a Sentinel-1 SLC product from the given directory path
    pub fn load(directory_path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        // Implementation would parse XML metadata and load binary data
        // This is a complex operation that would require access to the actual Sentinel-1 data format
        todo!("Implement Sentinel-1 SLC product loading")
    }

    /// Returns the interferometric burst pairs between this SLC product and another
    pub fn get_burst_pairs(&self, other: &Sentinel1SlcProduct) -> Vec<(SlcBurst, SlcBurst)> {
        // Implementation would identify matching bursts between two acquisitions
        // using metadata like burst_id, subswath_id, etc.
        todo!("Implement burst pair identification")
    }

    /// Returns a list of all bursts in the product
    pub fn get_all_bursts(&self) -> Vec<&SlcBurst> {
        let mut bursts = Vec::new();
        for subswath in &self.subswaths {
            for burst in &subswath.bursts {
                bursts.push(burst);
            }
        }
        bursts
    }

    /// Returns bursts that cover the given geographic point
    pub fn get_bursts_covering_point(&self, point: &GeoPoint) -> Vec<&SlcBurst> {
        let mut covering_bursts = Vec::new();

        for subswath in &self.subswaths {
            for burst in &subswath.bursts {
                if point.latitude >= burst.metadata.bounding_box.min_latitude
                    && point.latitude <= burst.metadata.bounding_box.max_latitude
                    && point.longitude >= burst.metadata.bounding_box.min_longitude
                    && point.longitude <= burst.metadata.bounding_box.max_longitude
                {
                    covering_bursts.push(burst);
                }
            }
        }

        covering_bursts
    }

    /// Returns bursts that cover the given geographic area
    pub fn get_bursts_covering_area(&self, area: &GeoBoundingBox) -> Vec<&SlcBurst> {
        let mut covering_bursts = Vec::new();

        for subswath in &self.subswaths {
            for burst in &subswath.bursts {
                // Check if there's any overlap between the burst bounding box and the area
                if area.min_latitude <= burst.metadata.bounding_box.max_latitude
                    && area.max_latitude >= burst.metadata.bounding_box.min_latitude
                    && area.min_longitude <= burst.metadata.bounding_box.max_longitude
                    && area.max_longitude >= burst.metadata.bounding_box.min_longitude
                {
                    covering_bursts.push(burst);
                }
            }
        }

        covering_bursts
    }
}

/// Represents a stack of coregistered SLC data for PSI processing.
///
/// A coregistered SLC stack consists of multiple SLC images aligned to a common reference geometry,
/// which is essential for PSI (Persistent Scatterer Interferometry) processing.
///
/// Sources:
/// - https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/interferometric-applications
/// - https://earth.esa.int/eogateway/documents/20142/37627/TM-19_pt1.pdf
#[derive(Debug, Clone)]
pub struct SlcStack {
    pub reference_product: Sentinel1SlcProduct,
    pub slave_products: Vec<Sentinel1SlcProduct>,
    pub coregistered_data: HashMap<String, Array3<Complex32>>, // Burst ID -> Stack of data
    pub timestamps: Vec<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
}

impl SlcStack {
    /// Creates a new SLC stack with the given reference product
    pub fn new(reference_product: Sentinel1SlcProduct) -> Self {
        let timestamps = vec![reference_product.metadata.acquisition_date];

        SlcStack {
            reference_product,
            slave_products: Vec::new(),
            coregistered_data: HashMap::new(),
            timestamps,
            metadata: HashMap::new(),
        }
    }

    /// Adds a slave product to the stack
    pub fn add_slave_product(&mut self, slave_product: Sentinel1SlcProduct) {
        self.timestamps
            .push(slave_product.metadata.acquisition_date);
        self.slave_products.push(slave_product);
    }

    /// Coregisters all slave products to the reference product
    pub fn coregister(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation would coregister each slave product to the reference
        // This is a complex operation involving orbit information, DEM, etc.
        todo!("Implement SLC stack coregistration")
    }
}

/// Represents different polynomial types for modeling phases.
///
/// These polynomial types are used for modeling phase components in PSI processing,
/// such as deformation, atmospheric effects, and topographic errors.
///
/// - https://earth.esa.int/eogateway/documents/20142/37627/TM-19_pt1.pdf
/// - https://www.mdpi.com/2072-4292/15/4/1165
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolynomialType {
    Linear,
    Quadratic,
    Cubic,
}

/// Represents a persistent scatterer candidate.
///
/// Persistent Scatterer candidates are pixels that potentially maintain coherence over time
/// and are selected based on various criteria such as amplitude stability or phase stability.
///
/// - https://www.sciencedirect.com/science/article/pii/S0924271615002415
/// - https://earth.esa.int/eogateway/documents/20142/37627/TM-19_pt1.pdf
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsCandidate {
    pub id: usize,
    pub row: usize,
    pub col: usize,
    pub latitude: f64,
    pub longitude: f64,
    pub height: f64,
    pub amplitude_mean: f32,
    pub amplitude_dispersion: f32,
    pub coherence: f32,
    pub phase_stability: f32,
    pub selected: bool,
}

/// Represents a persistent scatterer with estimated parameters
///
/// A Persistent Scatterer is a point target that maintains stable scattering characteristics
/// over long time periods. The associated parameters include deformation rate, height correction,
/// and various quality metrics.
///
/// - https://www.sciencedirect.com/science/article/pii/S0924271615002415
/// - https://earth.esa.int/eogateway/documents/20142/37627/TM-19_pt1.pdf
/// - https://www.mdpi.com/2072-4292/15/4/1165
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentScatterer {
    pub id: usize,
    pub row: usize,
    pub col: usize,
    pub latitude: f64,
    pub longitude: f64,
    pub height: f64,
    pub los_velocity: f64, // Line of sight velocity in mm/year
    pub coherence: f32,
    pub los_acceleration: Option<f64>,
    pub height_error: f64,
    pub velocity_error: f64,
    pub phase_residuals: Vec<f32>,
    pub model_parameters: HashMap<String, f64>,
}

/// Represents the results of PSI processing
///
/// PSI (Persistent Scatterer Interferometry) results include a set of identified persistent scatterers,
/// their locations, displacement rates, quality metrics, and associated processing parameters.
///
/// Sources:
/// - https://www.sciencedirect.com/science/article/pii/S0924271615002415
/// - https://earth.esa.int/eogateway/documents/20142/37627/TM-19_pt1.pdf
/// - https://www.mdpi.com/2072-4292/15/4/1165
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsiResults {
    pub reference_point: GeoPoint,
    pub persistent_scatterers: Vec<PersistentScatterer>,
    pub temporal_baselines: Vec<f64>, // In days
    pub processing_parameters: HashMap<String, String>,
    pub processing_date: DateTime<Utc>,
    pub area_of_interest: GeoBoundingBox,
}

impl PsiResults {
    /// Exports PSI results to a GeoJSON file
    pub fn export_to_geojson(&self, file_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation would convert PS results to GeoJSON features
        todo!("Implement GeoJSON export")
    }

    /// Exports PSI results to a GeoTIFF file
    pub fn export_to_geotiff(&self, file_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation would convert PS results to a raster GeoTIFF
        todo!("Implement GeoTIFF export")
    }

    /// Exports PSI results to a CSV file
    pub fn export_to_csv(&self, file_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation would write PS results to a CSV file
        todo!("Implement CSV export")
    }
}
