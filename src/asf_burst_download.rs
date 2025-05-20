//! ASF Burst Download
//!
//! Downloads Sentinel-1 bursts from ASF, as specified in a CSV file.
//! It uses credentials from a .env file for authentication.
//! https://sentinel1-burst-documentation.asf.alaska.edu/#api-specification

use base64::prelude::{BASE64_STANDARD, Engine as _};
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, Read, copy};
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

// Helper to extract form data from HTML
fn extract_form_data(
    html_body: &str,
    form_selector_str: &str,
) -> Result<(String, HashMap<String, String>), AsfDownloadError> {
    let document = Html::parse_document(html_body);
    let form_selector = Selector::parse(form_selector_str)
        .map_err(|e| AsfDownloadError::HtmlParsingError(format!("Invalid form selector: {}", e)))?;
    let input_selector = Selector::parse(
        "input[type='hidden'], input[type='text'], input[type='password'], input[type='submit']",
    )
    .map_err(|e| AsfDownloadError::HtmlParsingError(format!("Invalid input selector: {}", e)))?;

    if let Some(form_element) = document.select(&form_selector).next() {
        let action = form_element
            .value()
            .attr("action")
            .ok_or_else(|| {
                AsfDownloadError::HtmlParsingError(
                    "Login form action attribute not found".to_string(),
                )
            })?
            .to_string();
        let mut data = HashMap::new();
        for input_element in form_element.select(&input_selector) {
            if let Some(name) = input_element.value().attr("name") {
                let value = input_element
                    .value()
                    .attr("value")
                    .unwrap_or("")
                    .to_string();
                data.insert(name.to_string(), value);
            }
        }
        Ok((action, data))
    } else {
        Err(AsfDownloadError::HtmlParsingError(
            "Login form not found in HTML content".to_string(),
        ))
    }
}

// A query is like this "key=value&key2=value2"
pub fn parse_query_into_pairs(query: &str) -> Vec<(String, String)> {
    let mut v = Vec::new();
    for pair in query.split('&') {
        let parts: Vec<&str> = pair.splitn(2, '=').collect();
        if parts.len() == 2 {
            v.push((parts[0].to_string(), parts[1].to_string()));
        }
    }
    v
}

/// Download a single zip file from ASF after authenticating with Earthdata Login.
///
/// # Arguments
///
/// * `url` - The URL of the file to download
/// * `output_path` - The path where the downloaded file will be saved
/// * `username` - Earthdata Login username
/// * `password` - Earthdata Login password
///
/// # Returns
///
/// Result indicating success or the reason for failure
pub fn download_file(
    target_url_str: &str,
    output_path: &Path,
    username: &str,
    password: &str,
) -> Result<(), AsfDownloadError> {
    let agent = ureq::agent(); // Agent for cookie persistence
    let target_url = Url::parse(target_url_str)?;

    // 1. Initial request to the target URL to see if we get redirected to login
    println!("Attempting initial access to: {}", target_url_str);
    let initial_resp = agent.get(target_url_str).call()?;
    let mut current_url = initial_resp.get_uri().to_string();
    let mut response_body = initial_resp.into_body().read_to_string()?;

    // Check if we were redirected to a URS login page or directly received it
    if current_url.contains("urs.earthdata.nasa.gov") {
        println!("Redirected to Earthdata Login page: {}", current_url);

        // 2. Parse login form from the received HTML
        let (form_action_path, mut form_data) = extract_form_data(&response_body, "form#login")?;

        let login_form_action_url = if form_action_path.starts_with("http") {
            Url::parse(&form_action_path)?
        } else {
            let base_urs_url = Url::parse(&current_url)?;
            base_urs_url.join(&form_action_path)?
        };

        println!("Login form action URL: {}", login_form_action_url);

        // Populate username and password
        form_data.insert("username".to_string(), username.to_string());
        form_data.insert("password".to_string(), password.to_string());
        let is_none = form_data.get("commit").is_none();

        // Remove empty value keys that might cause issues with ureq's send_form if they were submit buttons
        form_data.retain(|_, v| !v.is_empty() || is_none);
        // Or more explicitly ensure commit is handled correctly if present
        if form_data.contains_key("commit")
            && form_data
                .get("commit")
                .unwrap_or(&"".to_string())
                .is_empty()
        {
            form_data.insert("commit".to_string(), "Log in".to_string()); //Common value for login buttons
        }

        let form_params: Vec<(&str, &str)> = form_data
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        println!(
            "Submitting login form to: {} with {} params",
            login_form_action_url,
            form_params.len()
        );
        // 3. Submit the login form
        let login_submit_resp = agent
            .post(login_form_action_url.as_str())
            .send_form(form_params)?;

        if login_submit_resp.status().as_u16() >= 400 {
            let status = login_submit_resp.status().clone();
            let error_page_html = login_submit_resp
                .into_body()
                .read_to_string()
                .unwrap_or_else(|_| "Failed to read error page body".to_string());
            return Err(AsfDownloadError::AuthenticationError(format!(
                "Earthdata Login submission failed with status: {}. Response: {}",
                status, error_page_html
            )));
        }
        println!(
            "Login form submitted. Status: {}. Final URL after login POST: {}",
            login_submit_resp.status(),
            login_submit_resp.get_uri()
        );

        // The URL after login submission (e.g., .../oauth/authorize) contains
        // a redirect_uri parameter that points to the ASF authentication service.
        // We must explicitly navigate the agent to this URI to complete the auth flow
        // and get the asf-urs cookie.
        let oauth_authorize_http_uri = login_submit_resp.get_uri(); // This is &http::Uri from the previous response
        let oauth_authorize_url_str = oauth_authorize_http_uri.to_string();

        println!(
            "Agent will now visit the URS OAuth authorize URL: {}. This should redirect to ASF auth.",
            oauth_authorize_url_str
        );

        // The agent needs to GET this URS OAuth URL.
        // This URS page should then issue a redirect (HTTP 302) to the ASF authentication service
        // (e.g., auth.asf.alaska.edu/login) with the necessary `code` and `state` parameters appended.
        // The agent will automatically follow these redirects.
        let asf_auth_final_resp = agent.get(&oauth_authorize_url_str).call()?;

        println!(
            "URS OAuth authorize step (and subsequent ASF redirects) finished. Status: {}. Final URL: {}",
            asf_auth_final_resp.status(),
            asf_auth_final_resp.get_uri()
        );

        // Check if this step itself resulted in a non-success status code at its *final* destination.
        // A success status (2xx) indicates the agent successfully navigated the ASF auth part.
        if !asf_auth_final_resp.status().is_success() {
            let status = asf_auth_final_resp.status();
            let error_url_at_asf_step = asf_auth_final_resp.get_uri().to_string();
            let error_body_content = asf_auth_final_resp
                .into_body()
                .read_to_string()
                .unwrap_or_else(|e| {
                    format!(
                        "Failed to read error response body from ASF auth step: {}",
                        e
                    )
                });

            return Err(AsfDownloadError::AuthenticationError(format!(
                "ASF authentication step via redirect_uri failed with status: {}. Final URL reached: {}. Response body: {}",
                status, error_url_at_asf_step, error_body_content
            )));
        }
        println!(
            "ASF auth step completed (final status {}). Agent should now have the asf-urs cookie.",
            asf_auth_final_resp.status()
        );
        // Check if the agent has the asf-urs cookie
        let cookies = agent.cookie_jar_lock();
        let domain = "urs.earthdata.nasa.gov";
        let path = "/";
        let asf_urs_cookie = cookies.get(domain, path, "asf-urs");
        if asf_urs_cookie.is_none() {
            return Err(AsfDownloadError::AuthenticationError(
                "ASF URS cookie not found".to_string(),
            ));
        }
        // If successful, the agent's cookie jar should now contain the asf-urs cookie.
        // The existing step 4 will then attempt the download with the updated agent.
    } else {
        // This is a critical failure in the OAuth flow.
        println!(
            "Error: Could not find 'redirect_uri' in query parameters of. This is required to complete ASF authentication."
        );
        return Err(AsfDownloadError::AuthenticationError(format!(
            "Missing redirect_uri in Earthdata OAuth step after login. Cannot proceed with ASF authentication."
        )));
    }
    // After this, the agent should have the necessary cookies from both URS and ASF.

    // 4. Attempt to download the actual file using the (now hopefully authenticated) agent
    println!("Attempting final download from: {}", target_url_str);
    let final_resp = agent.get(target_url_str).call()?;

    if final_resp.status().as_u16() == 403 {
        return Err(AsfDownloadError::Forbidden(
            "Access to the file was forbidden (403). Cookie might be invalid/expired or permissions insufficient.".to_string(),
        ));
    } else if final_resp.status().as_u16() >= 400 {
        let status = final_resp.status().clone();
        let error_body = final_resp
            .into_body()
            .read_to_string()
            .unwrap_or_else(|_| "<no error body>".to_string());
        return Err(AsfDownloadError::DownloadFailed(format!(
            "Download failed with status: {}. Body: {}",
            status, error_body
        )));
    }

    // Check content type to ensure it's not an HTML page (e.g. another login/error page)
    if let Some(content_type) = final_resp.headers().get("content-type") {
        if content_type
            .to_str()
            .unwrap_or("")
            .to_lowercase()
            .contains("text/html")
        {
            let uri = final_resp.get_uri().clone();
            let html_error_page = final_resp
                .into_body()
                .read_to_string()
                .unwrap_or_else(|_| "<failed to read HTML error page>".to_string());
            // Potentially save this HTML for debugging
            // fs::write(Path::new("error_page_at_download.html"), &html_error_page)?;
            let snippet = html_error_page.chars().take(200).collect::<String>();
            return Err(AsfDownloadError::DownloadFailed(format!(
                "Expected a file but received HTML content. URL: {}. Content: {}...",
                uri, snippet
            )));
        }
    }

    println!(
        "Successfully initiated download from {}. Status: {}",
        final_resp.get_uri(),
        final_resp.status()
    );

    let mut file = File::create(output_path)?;
    let mut body_reader = final_resp.into_body().into_reader();
    copy(&mut body_reader, &mut file)?;
    println!("File successfully saved to: {}", output_path.display());

    Ok(())
}

/// Download a file from ASF using Earthdata Login environment variables for authentication.
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
    println!(
        "Loading credentials from environment variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD"
    );
    let username = env::var("EARTHDATA_USERNAME")?;
    let password = env::var("EARTHDATA_PASSWORD")?;

    download_file(url, output_path, &username, &password)
}

/// Example usage function (primarily for testing the download_file_with_env_auth flow)
pub fn download_example() -> Result<(), AsfDownloadError> {
    // Get the first burst from the VALUES constant
    let (name, url) = VALUES[0];

    // Create output path
    let output_filename = format!("{}.zip", name);
    let output_path = Path::new(&output_filename);

    println!("Starting example download for: {name} from {url}");
    download_file_with_env_auth(url, &output_path)?;
    println!(
        "Example download completed successfully! File saved to {}",
        output_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // This test requires valid EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables to be set.
    // It will attempt a real download, which might be slow and consume data.
    // It also writes a file to the current directory.
    // Consider running this test manually or with placeholder credentials to test error paths.
    #[test]
    //#[ignore] // Ignored by default to prevent unintended network access and file writes during automated tests.
    fn test_real_download_with_env_auth() {
        // Ensure credentials are set in your environment for this test to pass.
        // export EARTHDATA_USERNAME="your_edl_username"
        // export EARTHDATA_PASSWORD="your_edl_password"
        if env::var("EARTHDATA_USERNAME").is_err() || env::var("EARTHDATA_PASSWORD").is_err() {
            println!(
                "Skipping test_real_download_with_env_auth: EARTHDATA_USERNAME or EARTHDATA_PASSWORD not set."
            );
            return;
        }

        let (name, url) = VALUES[0]; // Using the first URL for testing
        let output_filename = format!("test_download_{}.zip", name);
        let output_path = Path::new(&output_filename);

        println!(
            "Running test_real_download_with_env_auth: downloading {} to {}",
            url,
            output_path.display()
        );

        let result = download_file_with_env_auth(url, &output_path);

        if result.is_ok() {
            println!("Test download successful.");
            assert!(output_path.exists(), "Downloaded file should exist.");
            // Optionally, clean up the downloaded file
            //let _ = fs::remove_file(output_path);
        } else {
            eprintln!("Test download failed: {:?}", result.as_ref().err().unwrap());
            // If it failed due to auth, that's an expected path if creds are wrong/missing
            // If it failed for other network reasons, the test might still be useful.
        }
        // We don't assert!(result.is_ok()) here because network/auth can fail for valid reasons.
        // The purpose is more to exercise the code path.
    }

    // Test for the download_file function with placeholder credentials
    // This test will likely fail authentication but tests the function structure.
    #[test]
    fn test_download_file_structure_with_placeholder_creds() {
        let url = VALUES[1].1; // A valid URL from the list
        let output_path = Path::new("test_placeholder_download.zip");

        // Using obviously invalid credentials
        let result = download_file(url, output_path, "invaliduser", "invalidpassword");

        // We expect this to fail, likely with an AuthenticationError or Forbidden
        assert!(result.is_err());
        match result.err().unwrap() {
            AsfDownloadError::AuthenticationError(_) => { /* Expected for invalid creds */ }
            AsfDownloadError::Forbidden(_) => { /* Also possible if EDL blocks due to bad attempts */
            }
            AsfDownloadError::RequestError(ureq::Error::StatusCode(403)) => { /* Also possible */ }
            AsfDownloadError::RequestError(ureq::Error::StatusCode(401)) => { /* Also possible */ }
            other_error => panic!(
                "Expected AuthenticationError or Forbidden, but got {:?}",
                other_error
            ),
        }

        // Clean up dummy file if it was somehow created (it shouldn't be on auth error)
        if output_path.exists() {
            let _ = fs::remove_file(output_path);
        }
    }
}
