use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Represents a Sentinel-1 mission identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mission {
    S1A,
    S1B,
}

impl FromStr for Mission {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "S1A" => Ok(Mission::S1A),
            "S1B" => Ok(Mission::S1B),
            _ => Err(GranuleIdError::InvalidMission(s.to_string())),
        }
    }
}

impl fmt::Display for Mission {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Mission::S1A => write!(f, "S1A"),
            Mission::S1B => write!(f, "S1B"),
        }
    }
}

/// Represents a Sentinel-1 mode identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    /// Strip Map
    SM,
    /// Interferometric Wide Swath
    IW,
    /// Extra Wide Swath
    EW,
    /// Wave Mode
    WV,
}

impl FromStr for Mode {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SM" => Ok(Mode::SM),
            "IW" => Ok(Mode::IW),
            "EW" => Ok(Mode::EW),
            "WV" => Ok(Mode::WV),
            _ => Err(GranuleIdError::InvalidMode(s.to_string())),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Mode::SM => write!(f, "SM"),
            Mode::IW => write!(f, "IW"),
            Mode::EW => write!(f, "EW"),
            Mode::WV => write!(f, "WV"),
        }
    }
}

/// Represents the Interferometric Wide (IW) Sub-Swath of a Sentinel-1 acquisition.
///
/// Sentinel-1 IW mode divides the imaged area into three sub-swaths (IW1, IW2, IW3)
/// that are acquired using the Terrain Observation with Progressive Scans (TOPS) technique.
/// Each sub-swath has different frequency ranges and viewing geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IWSwath {
    IW1,
    IW2,
    IW3,
}

impl IWSwath {
    /// Converts a string to a IWSwath enum value
    ///
    /// # Arguments
    ///
    /// * `s` - A string slice that should contain "IW1", "IW2", "IW3" (case sensitive)
    ///
    /// # Returns
    ///
    /// * `Option<IWSwath>` - The corresponding IWSwath variant or None if the string is invalid
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "IW1" => Some(IWSwath::IW1),
            "IW2" => Some(IWSwath::IW2),
            "IW3" => Some(IWSwath::IW3),
            _ => None,
        }
    }

    /// Converts the IWSwath enum to a lowercase string
    ///
    /// # Returns
    ///
    /// * `String` - The string representation ("IW1", "IW2", or "IW3")
    pub fn to_string(&self) -> String {
        match self {
            IWSwath::IW1 => "IW1".to_string(),
            IWSwath::IW2 => "IW2".to_string(),
            IWSwath::IW3 => "IW3".to_string(),
        }
    }
}

impl std::fmt::Display for IWSwath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl std::str::FromStr for IWSwath {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s).ok_or_else(|| format!("Invalid sub-swath: {}", s))
    }
}

/// Represents a Sentinel-1 product type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductType {
    /// Raw Level-0
    RAW,
    /// Single Look Complex
    SLC,
    /// Ground Range Detected
    GRD,
    /// Ocean
    OCN,
}

impl FromStr for ProductType {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "RAW" => Ok(ProductType::RAW),
            "SLC" => Ok(ProductType::SLC),
            "GRD" => Ok(ProductType::GRD),
            "OCN" => Ok(ProductType::OCN),
            _ => Err(GranuleIdError::InvalidProductType(s.to_string())),
        }
    }
}

impl fmt::Display for ProductType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProductType::RAW => write!(f, "RAW"),
            ProductType::SLC => write!(f, "SLC"),
            ProductType::GRD => write!(f, "GRD"),
            ProductType::OCN => write!(f, "OCN"),
        }
    }
}

/// Represents a Sentinel-1 resolution class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resolution {
    /// Full Resolution
    F,
    /// High Resolution
    H,
    /// Medium Resolution
    M,
    /// Not applicable
    NA,
}

impl FromStr for Resolution {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "F" => Ok(Resolution::F),
            "H" => Ok(Resolution::H),
            "M" => Ok(Resolution::M),
            "_" => Ok(Resolution::NA),
            _ => Err(GranuleIdError::InvalidResolution(s.to_string())),
        }
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Resolution::F => write!(f, "F"),
            Resolution::H => write!(f, "H"),
            Resolution::M => write!(f, "M"),
            Resolution::NA => write!(f, "_"),
        }
    }
}

/// Represents a Sentinel-1 processing level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingLevel {
    /// Level-0
    L0,
    /// Level-1
    L1,
    /// Level-2
    L2,
    /// A
    A,
}

impl FromStr for ProcessingLevel {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(ProcessingLevel::L0),
            "1" => Ok(ProcessingLevel::L1),
            "2" => Ok(ProcessingLevel::L2),
            "A" => Ok(ProcessingLevel::A),
            _ => Err(GranuleIdError::InvalidProcessingLevel(s.to_string())),
        }
    }
}

impl fmt::Display for ProcessingLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProcessingLevel::L0 => write!(f, "0"),
            ProcessingLevel::L1 => write!(f, "1"),
            ProcessingLevel::L2 => write!(f, "2"),
            ProcessingLevel::A => write!(f, "A"),
        }
    }
}

/// Represents a Sentinel-1 polarization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum PolarizationMode {
    /// Single HH
    SH,
    /// Single VV
    SV,
    /// Dual HH/HV
    DH,
    /// Dual VV/VH
    DV,
    /// Partial Dual, HH Only**
    HH,
    /// Partial Dual, HV Only**
    HV,
    /// Partial Dual, VV Only**
    VV,
    /// Partial Dual, VH Only**
    VH,
}

impl FromStr for PolarizationMode {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SH" => Ok(PolarizationMode::SH),
            "SV" => Ok(PolarizationMode::SV),
            "DH" => Ok(PolarizationMode::DH),
            "DV" => Ok(PolarizationMode::DV),
            "HH" => Ok(PolarizationMode::HH),
            "HV" => Ok(PolarizationMode::HV),
            "VV" => Ok(PolarizationMode::VV),
            "VH" => Ok(PolarizationMode::VH),
            _ => Err(GranuleIdError::InvalidPolarizationMode(s.to_string())),
        }
    }
}

impl fmt::Display for PolarizationMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PolarizationMode::SH => write!(f, "SH"),
            PolarizationMode::SV => write!(f, "SV"),
            PolarizationMode::DH => write!(f, "DH"),
            PolarizationMode::DV => write!(f, "DV"),
            PolarizationMode::HH => write!(f, "HH"),
            PolarizationMode::HV => write!(f, "HV"),
            PolarizationMode::VV => write!(f, "VV"),
            PolarizationMode::VH => write!(f, "VH"),
        }
    }
}

/// Represents a Sentinel-1 product class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductClass {
    /// SAR Standard
    SARStandard,
    /// Annotation product
    Annotation,
    /// Noise
    Noise,
    /// Calibration
    Calibration,
    /// ETAD
    ETAD,
}

impl FromStr for ProductClass {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "S" => Ok(ProductClass::SARStandard),
            "A" => Ok(ProductClass::Annotation),
            "N" => Ok(ProductClass::Noise),
            "C" => Ok(ProductClass::Calibration),
            "X" => Ok(ProductClass::ETAD),
            _ => Err(GranuleIdError::InvalidProductType(s.to_string())),
        }
    }
}

impl fmt::Display for ProductClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProductClass::SARStandard => write!(f, "S"),
            ProductClass::Annotation => write!(f, "A"),
            ProductClass::Noise => write!(f, "N"),
            ProductClass::Calibration => write!(f, "C"),
            ProductClass::ETAD => write!(f, "X"),
        }
    }
}

/// Absolute Orbit Number at product start time.
///
/// Range: 000001-999999
///
/// Always represented as a 6-digit number, zero-padded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrbitNumber {
    number: u32,
}

impl OrbitNumber {
    pub fn new(number: u32) -> Result<Self, GranuleIdError> {
        if !(1..=999999).contains(&number) {
            return Err(GranuleIdError::InvalidOrbitNumber);
        }

        Ok(OrbitNumber { number })
    }
}

impl FromStr for OrbitNumber {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let number = u32::from_str_radix(s, 10).map_err(|_| GranuleIdError::InvalidOrbitNumber)?;
        OrbitNumber::new(number)
    }
}

impl fmt::Display for OrbitNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:06}", self.number)
    }
}

/// Custom serialization: store OrbitNumber as a plain integer.
impl Serialize for OrbitNumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.number)
    }
}

/// Custom deserialization: parse an integer into OrbitNumber.
impl<'de> Deserialize<'de> for OrbitNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let number = u32::deserialize(deserializer)?;
        OrbitNumber::new(number).map_err(serde::de::Error::custom)
    }
}

/// Mission Data Take Id (Hexadecimal)
///
/// Range: 000001-FFFFFF
///
/// Always represented as a 6-digit hexadecimal number, zero-padded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataTakeId {
    id: u32,
}

impl DataTakeId {
    pub fn new(id: u32) -> Result<Self, GranuleIdError> {
        if !(1..=0xFFFFFF).contains(&id) {
            return Err(GranuleIdError::InvalidDataTakeId(String::from(
                "Out of range! The Data Take Id must always be in the range 000001-FFFFFF.",
            )));
        }

        Ok(DataTakeId { id })
    }
}

impl FromStr for DataTakeId {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let id = u32::from_str_radix(s, 16)
            .map_err(|_| GranuleIdError::InvalidDataTakeId(s.to_string()))?;
        DataTakeId::new(id)
    }
}

impl fmt::Display for DataTakeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:06X}", self.id)
    }
}

impl Serialize for DataTakeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.id)
    }
}

impl<'de> Deserialize<'de> for DataTakeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let number = u32::deserialize(deserializer)?;
        DataTakeId::new(number).map_err(serde::de::Error::custom)
    }
}

/// Error type for Sentinel-1 Granule ID parsing
#[derive(Error, Debug)]
pub enum GranuleIdError {
    #[error("invalid format for Sentinel-1 Granule ID")]
    InvalidFormat,
    #[error("invalid mission: {0}")]
    InvalidMission(String),
    #[error("invalid mode: {0}")]
    InvalidMode(String),
    #[error("invalid product type: {0}")]
    InvalidProductType(String),
    #[error("invalid resolution: {0}")]
    InvalidResolution(String),
    #[error("invalid processing level: {0}")]
    InvalidProcessingLevel(String),
    #[error("invalid polarization mode: {0}")]
    InvalidPolarizationMode(String),
    #[error("invalid date time format: {0}")]
    InvalidDateTime(#[from] chrono::ParseError),
    #[error("invalid orbit number")]
    InvalidOrbitNumber,
    #[error("invalid data take identifier")]
    InvalidDataTakeId(String),
}

/// Represents a Sentinel-1 Granule ID
///
/// Format: MMM_BB_TTTR_LFPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_OOOOOO_DDDDDD_CCCC
///
/// Where:
/// - MMM: Mission identifier (S1A or S1B)
/// - BB: Mode/beam identifier (SM, IW, EW, WV)
/// - TTT: Product type (RAW, SLC, GRD, OCN)
/// - R: Resolution class (F, H, M or _)
/// - L: Processing level (0, 1, 2)
/// - F: Product class (S for Standard, A for Annotation)
/// - PP: Polarization (SH, SV, DH, DV)
/// - YYYYMMDDTHHMMSS: Start date and time
/// - YYYYMMDDTHHMMSS: End date and time
/// - OOOOOO: Absolute orbit number
/// - DDDDDD: Mission data-take identifier
/// - CCCC: Product unique identifier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentinel1GranuleId {
    pub mission: Mission,
    pub mode: Mode,
    pub product_type: ProductType,
    pub resolution: Resolution,
    pub processing_level: ProcessingLevel,
    pub product_class: ProductClass,
    pub polarization: PolarizationMode,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub orbit_number: OrbitNumber,
    pub data_take_id: DataTakeId,
    pub product_id: String,
    /// Original granule ID string
    pub raw_id: String,
}

impl Sentinel1GranuleId {
    /// Parse a Sentinel-1 Granule ID from its string representation
    pub fn parse(granule_id: &str) -> Result<Self, GranuleIdError> {
        //MMM_BB_TTTR_LFPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_OOOOOO_DDDDDD_CCCC.EEEE
        // Parse mission identifier
        let mission = Mission::from_str(&granule_id[0..3])?;
        // Parse mode beam identifier
        let mode = Mode::from_str(&granule_id[4..6])?;
        // Parse product type
        let product_type = ProductType::from_str(&granule_id[7..10])?;
        // Parse resolution class
        let resolution = Resolution::from_str(&granule_id[10..11])?;
        // Parse processing level
        let processing_level = ProcessingLevel::from_str(&granule_id[12..13])?;
        // Parse product class
        let product_class = ProductClass::from_str(&granule_id[13..14])?;
        // Parse polarization mode
        let polarization = PolarizationMode::from_str(&granule_id[14..16])?;
        // Parse start time
        let start_time = NaiveDateTime::parse_from_str(&granule_id[17..32], "%Y%m%dT%H%M%S")?;
        let start_time = DateTime::from_naive_utc_and_offset(start_time, Utc);
        // Parse end time
        let end_time = NaiveDateTime::parse_from_str(&granule_id[33..48], "%Y%m%dT%H%M%S")?;
        let end_time = DateTime::from_naive_utc_and_offset(end_time, Utc);
        // Parse orbit number
        let orbit_number = OrbitNumber::from_str(&granule_id[49..55])?;
        // Parse data take identifier
        let data_take_id = DataTakeId::from_str(&granule_id[56..62])?;
        // Parse product unique identifier
        let product_id = granule_id[63..67].to_string();

        Ok(Sentinel1GranuleId {
            mission,
            mode,
            product_type,
            resolution,
            processing_level,
            product_class,
            polarization,
            start_time,
            end_time,
            orbit_number,
            data_take_id,
            product_id,
            raw_id: granule_id.to_string(),
        })
    }
}

impl FromStr for Sentinel1GranuleId {
    type Err = GranuleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Sentinel1GranuleId::parse(s)
    }
}

impl fmt::Display for Sentinel1GranuleId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}{}_{}{}{}_{}_{}_{}_{}_{}",
            self.mission,
            self.mode,
            self.product_type,
            self.resolution,
            self.processing_level,
            self.product_class,
            self.polarization,
            self.start_time.format("%Y%m%dT%H%M%S"),
            self.end_time.format("%Y%m%dT%H%M%S"),
            self.orbit_number,
            self.data_take_id,
            self.product_id,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_granule_id() {
        let granule_id = "S1A_IW_SLC__1SDV_20180101T103955_20180101T104022_019964_021FFD_0A9F";
        let parsed = Sentinel1GranuleId::parse(granule_id).unwrap();

        assert_eq!(parsed.mission, Mission::S1A);
        assert_eq!(parsed.mode, Mode::IW);
        assert_eq!(parsed.product_type, ProductType::SLC);
        assert_eq!(parsed.resolution, Resolution::NA);
        assert_eq!(parsed.processing_level, ProcessingLevel::L1);
        assert_eq!(parsed.product_class, ProductClass::SARStandard);
        assert_eq!(parsed.polarization, PolarizationMode::DV);
        assert_eq!(parsed.orbit_number, OrbitNumber::new(19964).unwrap());
        assert_eq!(parsed.data_take_id, DataTakeId::new(0x021FFD).unwrap());
        assert_eq!(
            parsed.start_time.format("%Y%m%dT%H%M%S").to_string(),
            "20180101T103955"
        );
        assert_eq!(
            parsed.end_time.format("%Y%m%dT%H%M%S").to_string(),
            "20180101T104022"
        );
        assert_eq!(parsed.product_id, "0A9F");
    }

    #[test]
    fn test_parse_invalid_granule_id() {
        let granule_id = "INVALID_GRANULE_ID";
        assert!(Sentinel1GranuleId::parse(granule_id).is_err());
    }

    #[test]
    fn test_display() {
        let granule_id = "S1A_IW_SLC__1SDV_20180101T103955_20180101T104022_019964_021FFD_0A9F";
        let parsed = Sentinel1GranuleId::parse(granule_id).unwrap();
        let display_str = format!("{}", parsed);
        println!("{}", display_str);

        assert!(display_str.starts_with("S1A_IW_SLC_"));
        assert!(display_str.contains("1SDV"));
        assert!(display_str.contains("20180101T103955"));
        assert!(display_str.contains("019964"));
    }

    #[test]
    fn test_serde() {
        let granule_id = "S1A_IW_SLC__1SDV_20180101T103955_20180101T104022_019964_021FFD_0A9F";
        let parsed = Sentinel1GranuleId::parse(granule_id).unwrap();

        // Test serialization
        let serialized = serde_json::to_string(&parsed).unwrap();

        // Test deserialization
        let deserialized: Sentinel1GranuleId = serde_json::from_str(&serialized).unwrap();

        assert_eq!(parsed, deserialized);
    }
}
