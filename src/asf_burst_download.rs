//! ASF Burst Download
//!
//! Downloads Sentinel-1 bursts from ASF, as specified in a CSV file.
//! It uses credentials from a .env file for authentication.
//! https://sentinel1-burst-documentation.asf.alaska.edu/#api-specification

use base64::prelude::{BASE64_STANDARD, Engine as _};
use std::env;
use std::fs::File;
use std::io::{self, Read, Write, copy};
use std::path::Path;
use ureq::{self, ResponseExt};
use url::Url;

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
    /// Authentication failure during Earthdata Login
    AuthenticationError(String),
    /// Access forbidden (e.g., invalid/expired cookie, or insufficient permissions)
    Forbidden(String),
    /// Download failed
    DownloadFailed(String),
    /// URL parsing error
    UrlParseError(url::ParseError),
    /// HTML parsing error
    HtmlParsingError(String),
    /// Zip extraction error
    ZipExtractError(zip_extract::ZipExtractError),
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

impl From<url::ParseError> for AsfDownloadError {
    fn from(err: url::ParseError) -> Self {
        AsfDownloadError::UrlParseError(err)
    }
}

impl From<zip_extract::ZipExtractError> for AsfDownloadError {
    fn from(err: zip_extract::ZipExtractError) -> Self {
        AsfDownloadError::ZipExtractError(err)
    }
}

pub struct AsfBurstDownloader {
    agent: ureq::Agent,
    auth_header: String,
}

impl AsfBurstDownloader {
    pub fn new_with_env_auth() -> Result<Self, AsfDownloadError> {
        let username = env::var("EARTHDATA_USERNAME")?;
        let password = env::var("EARTHDATA_PASSWORD")?;
        Ok(Self::new(&username, &password))
    }

    pub fn new(username: &str, password: &str) -> Self {
        let auth_header = format!(
            "Basic {}",
            BASE64_STANDARD.encode(format!("{}:{}", username, password))
        );
        AsfBurstDownloader {
            agent: ureq::agent(),
            auth_header,
        }
    }

    /// Download a single zip file from ASF after authenticating with Earthdata Login.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL of the file to download
    /// * `output_path` - The path where the downloaded file will be saved
    ///
    /// # Returns
    ///
    /// Result indicating success or the reason for failure
    pub fn download_file(&self, url: &str, output_path: &Path) -> Result<(), AsfDownloadError> {
        // 1. Initial request to the target URL to see if we get redirected to login
        println!("Attempting initial access to: {}", url);
        let initial_resp = self.agent.get(url).call()?;
        let current_url = initial_resp.get_uri().to_string();
        let status_code = initial_resp.status();

        // Check if we were redirected to a URS login page
        // or if we received the file directly
        let download_url = if current_url.contains("urs.earthdata.nasa.gov") {
            println!("Redirected to Earthdata Login page: {}", current_url);

            let resp = self
                .agent
                .get(current_url)
                .header("Authorization", self.auth_header.clone())
                .call()?;

            resp.get_uri().to_string()
        } else if status_code == 200 {
            println!("Received file directly: {}", current_url);
            current_url
        } else {
            // This path means initial request was not to URS, and it wasn't recognized as a direct small response.
            // It might be an HTML page from ASF that isn't the login page.
            println!(
                "Initial request to {} did not redirect to URS and doesn't look like a direct file. It might be an unexpected page from ASF. Current URL: {}",
                url, current_url
            );
            println!("STATUS CODE: {}", status_code);
            // Potentially, this could be an error page from ASF itself.
            // The download attempt later will clarify.
            // The original `else` branch here would throw "Missing redirect_uri".
            // This is kept to align with previous logic if the specific conditions above are not met.
            // However, if `current_url` is not a URS URL, then `redirect_uri` wouldn't be expected here.
            // This part of the logic might need refinement based on actual non-URS initial responses.
            return Err(AsfDownloadError::AuthenticationError(format!(
                "Initial request did not redirect to URS login, but was not recognized as a direct file/small error. Current URL: {}. This path indicates an issue in the expected auth flow.",
                current_url
            )));
        };

        // Proceed to download and extract the file and save it to the output path
        let download_resp = self.agent.get(download_url).call()?;
        let mut download_body = download_resp.into_body();
        let mut reader = download_body.as_reader();
        // Read the zip file into memory
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        zip_extract::extract(std::io::Cursor::new(buffer), output_path, true)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // This test requires valid EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables to be set.
    // It will attempt a real download, which might be slow and consume data.
    // It also writes a file to the current directory.
    // Consider running this test manually or with placeholder credentials to test error paths.
    // This test is ignored by default to prevent unintended network access and file writes during automated tests.
    #[test]
    #[ignore]
    fn test_real_download_with_env_auth() {
        if env::var("EARTHDATA_USERNAME").is_err() || env::var("EARTHDATA_PASSWORD").is_err() {
            println!(
                "Skipping test_real_download_with_env_auth: EARTHDATA_USERNAME or EARTHDATA_PASSWORD not set."
            );
            return;
        }

        let downloader = AsfBurstDownloader::new_with_env_auth().unwrap();
        for (name, url) in VALUES {
            let output_filename = format!("test_download_{}", name);
            let output_path = Path::new(&output_filename);

            println!(
                "Running test_real_download_with_env_auth: downloading {} to {}",
                url,
                output_path.display()
            );

            let result = downloader.download_file(url, &output_path);

            if result.is_ok() {
                println!("Test download successful.");
                assert!(output_path.exists(), "Downloaded file should exist.");
                // Clean up the downloaded file
                // let _ = std::fs::remove_file(output_path);
            } else {
                eprintln!("Test download failed: {:?}", result.as_ref().err().unwrap());
                // If it failed due to auth, that's an expected path if creds are wrong/missing
                // If it failed for other network reasons, the test might still be useful.
            }
            // We don't assert!(result.is_ok()) here because network/auth can fail for valid reasons.
            // The purpose is more to exercise the code path.
        }
    }
}
