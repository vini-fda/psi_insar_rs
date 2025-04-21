//! Module to help parsing and manipulation of the metadata included in Sentinel-1 SLC Products.
use crate::granule_id::{
    DataTakeId, IWSwath, Mission, Mode, OrbitNumber, PolarizationMode, ProductType,
};
use annotation_xml::datetime_format;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod annotation_xml;
pub mod calibration_xml;
pub mod noise_xml;

#[derive(Serialize, Deserialize)]
pub struct AdsHeader {
    #[serde(rename = "missionId")]
    pub mission_id: Mission,
    #[serde(rename = "productType")]
    pub product_type: ProductType,
    pub polarisation: PolarizationMode,
    pub mode: Mode,
    pub swath: IWSwath,
    #[serde(rename = "startTime", with = "datetime_format")]
    pub start_time: DateTime<Utc>,
    #[serde(rename = "stopTime", with = "datetime_format")]
    pub stop_time: DateTime<Utc>,
    #[serde(rename = "absoluteOrbitNumber")]
    pub absolute_orbit_number: OrbitNumber,
    #[serde(rename = "missionDataTakeId")]
    pub mission_data_take_id: DataTakeId,
    #[serde(rename = "imageNumber")]
    pub image_number: String,
}
