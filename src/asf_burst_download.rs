//! ASF Burst Download
//!
//! Downloads Sentinel-1 bursts from ASF, as specified in a CSV file.
//! It uses credentials from a .env file for authentication.
//! https://sentinel1-burst-documentation.asf.alaska.edu/#api-specification

use base64::prelude::*;
use std::env;
use std::fs::File;
use std::io::{self, copy};
use std::path::Path;

/// List of Sentinel-1 bursts to download.
/// The first element is the granule name, the second is the URL.
const VALUES: [(&str, &str); 7] = [
    (
        "S1_305967_IW3_20150916T122546_VV_8302-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20150916T122538_20150916T122603_007740_00AC19_8302/IW3/VV/2.zip",
    ),
    (
        "S1_305967_IW3_20150928T122546_VV_5407-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20150928T122539_20150928T122606_007915_00B0D8_5407/IW3/VV/2.zip",
    ),
    (
        "S1_305967_IW3_20151010T122546_VV_7501-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151010T122539_20151010T122603_008090_00B578_7501/IW3/VV/2.zip",
    ),
    (
        "S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151022T122539_20151022T122606_008265_00BA51_5A48/IW3/VV/2.zip",
    ),
    (
        "S1_305967_IW3_20151103T122546_VV_AE93-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151103T122539_20151103T122603_008440_00BEE0_AE93/IW3/VV/2.zip",
    ),
    (
        "S1_305967_IW3_20151115T122546_VV_8956-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151115T122533_20151115T122600_008615_00C3B4_8956/IW3/VV/4.zip",
    ),
    (
        "S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151127T122533_20151127T122557_008790_00C894_14CF/IW3/VV/4.zip",
    ),
];

/// Error type for ASF download operations
#[derive(Debug)]
pub enum AsfDownloadError {
    /// Error from HTTP request
    RequestError(ureq::Error),
    /// I/O error
    IoError(io::Error),
    /// Missing environment variable
    EnvError(env::VarError),
}

impl From<ureq::Error> for AsfDownloadError {
    fn from(err: ureq::Error) -> Self {
        AsfDownloadError::RequestError(err)
    }
}

impl From<io::Error> for AsfDownloadError {
    fn from(err: io::Error) -> Self {
        AsfDownloadError::IoError(err)
    }
}

impl From<env::VarError> for AsfDownloadError {
    fn from(err: env::VarError) -> Self {
        AsfDownloadError::EnvError(err)
    }
}

/// Download a single zip file from ASF.
///
/// # Arguments
///
/// * `url` - The URL of the file to download
/// * `output_path` - The path where the downloaded file will be saved
/// * `username` - ASF username
/// * `password` - ASF password
///
/// # Returns
///
/// Result indicating success or the reason for failure
pub fn download_file(
    url: &str,
    output_path: &Path,
    username: &str,
    password: &str,
) -> Result<(), AsfDownloadError> {
    // Create the agent for connection pooling
    let agent = ureq::agent();

    // Create the authorization header with basic auth
    let auth = format!(
        "Basic {}",
        BASE64_STANDARD.encode(format!("{}:{}", username, password))
    );

    // Make the request with basic authentication
    let response = agent.get(url).header("Authorization", &auth).call()?;

    // Create the output file
    let mut file = File::create(output_path)?;

    // Copy the response body to the file
    let body = response.into_body();
    let mut reader = body.into_reader();
    copy(&mut reader, &mut file)?;

    Ok(())
}

/// Download a file from ASF using environment variables for authentication.
///
/// # Arguments
///
/// * `url` - The URL of the file to download
/// * `output_path` - The path where the downloaded file will be saved
///
/// # Returns
///
/// Result indicating success or the reason for failure
pub fn download_file_with_env_auth(url: &str, output_path: &Path) -> Result<(), AsfDownloadError> {
    let username = env::var("ASF_USERNAME")?;
    let password = env::var("ASF_PASSWORD")?;

    download_file(url, output_path, &username, &password)
}

/// Example usage function
pub fn download_example() -> Result<(), AsfDownloadError> {
    // Get the first burst from the VALUES constant
    let (name, url) = VALUES[0];

    // Create output path
    let output_path = Path::new(name).with_extension("zip");

    println!("Downloading {name} from {url}");
    download_file_with_env_auth(url, &output_path)?;
    println!("Successfully downloaded to {}", output_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_file() {
        let url = "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151022T122539_20151022T122606_008265_00BA51_5A48/IW3/VV/2.zip";
        let output_path = Path::new("test_output.zip");
        let result = download_file(url, output_path, "test_user", "test_password");
        assert!(result.is_ok());
    }
}
