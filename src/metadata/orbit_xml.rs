use std::path::Path;

use chrono::{DateTime, Utc};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

/// The EarthExplorerFile (.EOF) format describes both Precise Orbit Ephemerides `AUX_POEORB`
/// and Restituted Orbit files `AUX_RESORB`.
#[derive(Serialize, Deserialize)]
pub struct EarthExplorerFile {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "Earth_Explorer_Header")]
    pub earth_explorer_header: EarthExplorerHeader,
    #[serde(rename = "Data_Block")]
    pub data_block: DataBlock,
}

#[derive(Serialize, Deserialize)]
pub struct EarthExplorerHeader {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "Fixed_Header")]
    pub fixed_header: FixedHeader,
    #[serde(rename = "Variable_Header")]
    pub variable_header: VariableHeader,
}

#[derive(Serialize, Deserialize)]
pub struct FixedHeader {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "File_Name")]
    pub file_name: String,
    #[serde(rename = "File_Description")]
    pub file_description: String,
    #[serde(rename = "Notes")]
    pub notes: Notes,
    #[serde(rename = "Mission")]
    pub mission: String,
    #[serde(rename = "File_Class")]
    pub file_class: String,
    #[serde(rename = "File_Type")]
    pub file_type: String,
    #[serde(rename = "Validity_Period")]
    pub validity_period: ValidityPeriod,
    #[serde(rename = "File_Version")]
    pub file_version: String,
    #[serde(rename = "Source")]
    pub source: Source,
}

#[derive(Serialize, Deserialize)]
pub struct Notes {}

#[derive(Serialize, Deserialize)]
pub struct ValidityPeriod {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "Validity_Start")]
    pub validity_start: String,
    #[serde(rename = "Validity_Stop")]
    pub validity_stop: String,
}

#[derive(Serialize, Deserialize)]
pub struct Source {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "System")]
    pub system: String,
    #[serde(rename = "Creator")]
    pub creator: String,
    #[serde(rename = "Creator_Version")]
    pub creator_version: String,
    #[serde(rename = "Creation_Date")]
    pub creation_date: String,
}

#[derive(Serialize, Deserialize)]
pub struct VariableHeader {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "Ref_Frame")]
    pub ref_frame: String,
    #[serde(rename = "Time_Reference")]
    pub time_reference: String,
}

#[derive(Serialize, Deserialize)]
pub struct DataBlock {
    #[serde(rename = "@type")]
    pub data_block_type: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "List_of_OSVs")]
    pub list_of_osvs: ListOfOsvs,
}

#[derive(Serialize, Deserialize)]
pub struct ListOfOsvs {
    #[serde(rename = "@count")]
    pub count: usize,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "OSV")]
    pub osv: Vec<Osv>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Osv {
    #[serde(rename = "$text")]
    pub text: Option<String>,

    #[serde(rename = "TAI", deserialize_with = "parse_datetime_tag")]
    pub tai: DateTime<Utc>,

    #[serde(rename = "UTC", deserialize_with = "parse_datetime_tag")]
    pub utc: DateTime<Utc>,

    #[serde(rename = "UT1", deserialize_with = "parse_datetime_tag")]
    pub ut1: DateTime<Utc>,

    #[serde(rename = "Absolute_Orbit")]
    pub absolute_orbit: String, // keep as-is if used as ID, otherwise convert to u32

    #[serde(rename = "X", deserialize_with = "parse_unit_tag")]
    pub x: f64,

    #[serde(rename = "Y", deserialize_with = "parse_unit_tag")]
    pub y: f64,

    #[serde(rename = "Z", deserialize_with = "parse_unit_tag")]
    pub z: f64,

    #[serde(rename = "VX", deserialize_with = "parse_unit_tag")]
    pub vx: f64,

    #[serde(rename = "VY", deserialize_with = "parse_unit_tag")]
    pub vy: f64,

    #[serde(rename = "VZ", deserialize_with = "parse_unit_tag")]
    pub vz: f64,

    #[serde(rename = "Quality")]
    pub quality: String,
}

#[derive(Serialize, Deserialize)]
pub struct X {
    #[serde(rename = "@unit")]
    pub unit: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Y {
    #[serde(rename = "@unit")]
    pub unit: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Z {
    #[serde(rename = "@unit")]
    pub unit: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Vx {
    #[serde(rename = "@unit")]
    pub unit: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Vy {
    #[serde(rename = "@unit")]
    pub unit: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Vz {
    #[serde(rename = "@unit")]
    pub unit: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

fn parse_unit_tag<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(rename = "$text")]
        text: Option<String>,
    }

    let wrapper = Wrapper::deserialize(deserializer)?;
    wrapper
        .text
        .ok_or_else(|| de::Error::missing_field("$text"))?
        .parse::<f64>()
        .map_err(de::Error::custom)
}

fn parse_datetime_tag<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    let s = s
        .split('=')
        .nth(1)
        .ok_or_else(|| de::Error::custom("Expected format like 'UTC=...'"))?;

    // Remove trailing zeros if present for nanoseconds compatibility
    let s = s.trim_end_matches('0').trim_end_matches('.');

    // Parse with chrono (assumes full seconds or microsecond precision)
    Ok(DateTime::parse_from_rfc3339(&format!("{s}Z"))
        .map_err(de::Error::custom)?
        .with_timezone(&Utc))
}

impl EarthExplorerFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Self {
        let reader = std::fs::File::open(path).unwrap();
        let buf_reader = std::io::BufReader::new(reader);
        quick_xml::de::from_reader(buf_reader).unwrap()
    }

    pub fn parse(s: &str) -> Self {
        quick_xml::de::from_str(s).expect("Unable to parse Earth Explorer File")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::de::from_str;

    #[test]
    fn test_orbit_parsing() {
        let xml_content = include_str!("test_data/orbit_example.xml");
        let orbit: EarthExplorerFile = from_str(xml_content).expect("Failed to parse orbit XML");
        let osv = &orbit.data_block.list_of_osvs.osv[0];
        println!("First OSV = {osv:?}");
    }
}
