use super::AdsHeader;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Noise {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "adsHeader")]
    pub ads_header: AdsHeader,
    #[serde(rename = "noiseVectorList")]
    pub noise_vector_list: NoiseVectorList,
}

#[derive(Serialize, Deserialize)]
pub struct NoiseVectorList {
    #[serde(rename = "@count")]
    pub count: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "noiseVector")]
    pub noise_vector: Vec<NoiseVector>,
}

#[derive(Serialize, Deserialize)]
pub struct NoiseVector {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub line: String,
    pub pixel: Pixel,
    #[serde(rename = "noiseLut")]
    pub noise_lut: NoiseLut,
}

#[derive(Serialize, Deserialize)]
pub struct Pixel {
    #[serde(rename = "@count")]
    pub count: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct NoiseLut {
    #[serde(rename = "@count")]
    pub count: String,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::de::from_str;

    #[test]
    fn test_noise_parsing() {
        // Read the file content
        let xml_content = include_str!("test_data/noise_example.xml");

        // Parse the XML into our Noise struct
        let noise: Noise = from_str(&xml_content).expect("Failed to parse noise XML");
    }
}
