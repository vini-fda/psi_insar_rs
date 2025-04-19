use serde::{Deserialize, Deserializer, Serialize};

use super::AdsHeader;

#[derive(Serialize, Deserialize)]
pub struct Calibration {
    #[serde(rename = "adsHeader")]
    pub ads_header: AdsHeader,
    #[serde(rename = "calibrationInformation")]
    pub calibration_information: CalibrationInformation,
    #[serde(rename = "calibrationVectorList")]
    pub calibration_vector_list: CalibrationVectorList,
}

#[derive(Serialize, Deserialize)]
pub struct CalibrationInformation {
    #[serde(rename = "absoluteCalibrationConstant")]
    pub absolute_calibration_constant: f64,
}

#[derive(Serialize, Deserialize)]
pub struct CalibrationVectorList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "calibrationVector")]
    pub calibration_vectors: Vec<CalibrationVector>,
}

#[derive(Serialize, Deserialize)]
pub struct CalibrationVector {
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub line: String,
    pub pixel: Pixel,
    #[serde(rename = "sigmaNought")]
    pub sigma_nought: SigmaNought,
    #[serde(rename = "betaNought")]
    pub beta_nought: BetaNought,
    pub gamma: Gamma,
    pub dn: Dn,
}

/// Represents a list of pixel indices along a range line.
#[derive(Debug, Serialize, Deserialize)]
pub struct Pixel {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(
        rename = "$text",
        deserialize_with = "deserialize_space_separated_u32",
        serialize_with = "serialize_space_separated_u32"
    )]
    pub values: Vec<u32>,
}

/// Represents the sigma nought (σ⁰) calibration values.
/// These values are space-separated floats.
#[derive(Serialize, Deserialize)]
pub struct SigmaNought {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(
        rename = "$text",
        deserialize_with = "deserialize_space_separated_floats",
        serialize_with = "serialize_space_separated_floats"
    )]
    pub values: Vec<f64>,
}

/// Represents the beta nought (β⁰) calibration values.
/// These values are space-separated floats.
#[derive(Serialize, Deserialize)]
pub struct BetaNought {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(
        rename = "$text",
        deserialize_with = "deserialize_space_separated_floats",
        serialize_with = "serialize_space_separated_floats"
    )]
    pub values: Vec<f64>,
}

/// Represents the gamma (γ) calibration values.
/// These values are space-separated floats.
#[derive(Serialize, Deserialize)]
pub struct Gamma {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(
        rename = "$text",
        deserialize_with = "deserialize_space_separated_floats",
        serialize_with = "serialize_space_separated_floats"
    )]
    pub values: Vec<f64>,
}

/// Represents the Digital Number (DN) calibration values.
/// The values are space-separated integers.
#[derive(Serialize, Deserialize)]
pub struct Dn {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(
        rename = "$text",
        deserialize_with = "deserialize_space_separated_u32",
        serialize_with = "serialize_space_separated_u32"
    )]
    pub values: Vec<u32>,
}

/// Custom deserializer function for space-separated floats
fn deserialize_space_separated_floats<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    // First deserialize as an Option<String>
    let text = Option::<String>::deserialize(deserializer)?;
    Ok(text
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect())
}

fn serialize_space_separated_floats<S>(values: &Vec<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let space_separated_string = values
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    serializer.serialize_str(&space_separated_string)
}

/// Custom deserializer function for space-separated unsigned 32-bit integers.
fn deserialize_space_separated_u32<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    let text: Option<String> = Option::deserialize(deserializer)?;
    Ok(text
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect())
}

pub fn serialize_space_separated_u32<S>(values: &Vec<u32>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let space_separated_string = values
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    serializer.serialize_str(&space_separated_string)
}

#[cfg(test)]
mod tests {
    use crate::{
        granule_id::{
            DataTakeId, IWSwath, Mission, Mode, OrbitNumber, PolarizationMode, ProductType,
        },
        metadata::calibration_xml::Calibration,
    };
    use quick_xml::de::from_str;

    #[test]
    fn test_calibration_parsing() {
        // Read the file content
        let xml_content = include_str!("test_data/calibration_example.xml");

        // Parse the XML into our Calibration struct
        let calibration: Calibration =
            from_str(xml_content).expect("Failed to parse calibration XML");

        // Test AdsHeader fields
        assert_eq!(calibration.ads_header.mission_id, Mission::S1A);
        assert_eq!(calibration.ads_header.product_type, ProductType::SLC);
        assert_eq!(calibration.ads_header.polarisation, PolarizationMode::VV);
        assert_eq!(calibration.ads_header.mode, Mode::IW);
        assert_eq!(calibration.ads_header.swath, IWSwath::IW3);
        assert_eq!(
            calibration.ads_header.start_time,
            "2015-10-22T12:25:46.721151"
        );
        assert_eq!(
            calibration.ads_header.stop_time,
            "2015-10-22T12:25:49.816819"
        );
        assert_eq!(
            calibration.ads_header.absolute_orbit_number,
            OrbitNumber::new(8265).unwrap()
        );
        assert_eq!(
            calibration.ads_header.mission_data_take_id,
            DataTakeId::new(47697).unwrap()
        );
        assert_eq!(calibration.ads_header.image_number, "001");

        // Test calibration information
        assert_eq!(
            calibration
                .calibration_information
                .absolute_calibration_constant,
            1.000000e+00
        );

        // Test calibration vector list
        assert_eq!(calibration.calibration_vector_list.count, 2);
        assert_eq!(
            calibration
                .calibration_vector_list
                .calibration_vectors
                .len(),
            2
        );

        // Test first calibration vector
        let first_vector = &calibration.calibration_vector_list.calibration_vectors[0];
        assert_eq!(first_vector.azimuth_time, "2015-10-22T12:25:44.298538");
        assert_eq!(first_vector.line, "-1343");
        assert_eq!(first_vector.pixel.count, 5);
        assert_eq!(first_vector.pixel.values, [0, 40, 80, 120, 160]);
        assert_eq!(first_vector.sigma_nought.count, 5);
        assert!(first_vector.sigma_nought.values.contains(&2.89e+02));
        assert_eq!(first_vector.beta_nought.count, 5);
        assert!(first_vector.beta_nought.values.contains(&2.37e+02));
        assert_eq!(first_vector.gamma.count, 5);
        assert!(first_vector.gamma.values.contains(&2.49e+02));
        assert_eq!(first_vector.dn.count, 5);
        assert_eq!(first_vector.dn.values, [500, 505, 510, 515, 520]);

        // Test second calibration vector
        let second_vector = &calibration.calibration_vector_list.calibration_vectors[1];
        assert_eq!(second_vector.azimuth_time, "2015-10-22T12:25:45.123456");
        assert_eq!(second_vector.line, "-1200");
        assert_eq!(second_vector.pixel.count, 5);
        assert_eq!(second_vector.pixel.values, [0, 50, 100, 150, 200]);
        assert_eq!(second_vector.sigma_nought.count, 5);
        assert!(second_vector.sigma_nought.values.contains(&2.85e+02));
        assert_eq!(second_vector.beta_nought.count, 5);
        assert!(second_vector.beta_nought.values.contains(&2.36e+02));
        assert_eq!(second_vector.gamma.count, 5);
        assert!(second_vector.gamma.values.contains(&2.44e+02));
        assert_eq!(second_vector.dn.count, 5);
        assert_eq!(second_vector.dn.values, [520, 525, 530, 535, 540]);

        // Test that we can convert and manipulate the data
        let sigma_vec = &second_vector.sigma_nought.values;

        // Simple data manipulation test (e.g., calculating average sigma value)
        let avg_sigma = sigma_vec.iter().sum::<f64>() / sigma_vec.len() as f64;
        assert!(avg_sigma > 280.0 && avg_sigma < 285.0);
    }
}
