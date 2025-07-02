use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use urlencoding::encode;

use crate::{granule_id::Mission, metadata::orbit_xml::EarthExplorerFile};

/// Base URL for Identity Access Management on CDSE
const CDSE_IAM_BASE_URL: &str =
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token";
/// Base URL for searching CDSE Products
const CDSE_PRODUCTS_SEARCH_BASE_URL: &str =
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products";
/// Base URL for downloading CDSE Products
const CDSE_PRODUCTS_DOWNLOAD_BASE_URL: &str =
    "https://download.dataspace.copernicus.eu/odata/v1/Products";

#[derive(Deserialize)]
struct CDSETokenResponse {
    access_token: String,
}

/// Obtains precise orbit files from the Copernicus Data Space Ecosystem (CDSE)
pub struct CDSEOrbitDownloader {
    agent: ureq::Agent,
    auth_token: String,
}

impl CDSEOrbitDownloader {
    pub fn new(username: &str, password: &str) -> Self {
        let auth_token: String = Self::obtain_auth_token(username, password);
        // let agent = ureq::agent();
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .https_only(true)
            .http_status_as_error(false)
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
            .build();

        let agent = ureq::Agent::new_with_config(config);
        Self { agent, auth_token }
    }

    fn obtain_auth_token(username: &str, password: &str) -> String {
        let response = ureq::post(CDSE_IAM_BASE_URL)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send_form([
                ("client_id", "cdse-public"),
                ("username", username),
                ("password", password),
                ("grant_type", "password"),
            ]);

        match response {
            Ok(mut response) => {
                let token: CDSETokenResponse = response
                    .body_mut()
                    .read_json()
                    .expect("Failed to parse JSON response");
                token.access_token
            }
            Err(e) => panic!("Failed to obtain token: {e}"),
        }
    }

    /// Searches for the precise orbital ephemerides (`AUX_POEORB`) related to the given Sentinel-1 mission
    /// which includes the [start, end] time range.
    pub fn search(&self, mission: Mission, start: DateTime<Utc>, end: DateTime<Utc>) -> String {
        assert!(
            start < end,
            "The `start` datetime must refer to a moment in time before the `end` datetime parameter."
        );
        let start_rfc3339 = start.to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
        let end_rfc3339 = end.to_rfc3339_opts(chrono::SecondsFormat::Micros, true);

        let mission = mission.to_string();
        let filter = format!(
            "Online eq true and \
            startswith(Name,'{mission}_') and \
            Attributes/OData.CSC.StringAttribute/any(i0:i0/Name eq 'productType' and i0/Value eq 'AUX_POEORB') and \
            Collection/Name eq 'SENTINEL-1' and \
            ContentDate/Start le {end_rfc3339} and ContentDate/End ge {start_rfc3339}"
        );
        let filter = encode(&filter);

        let orderby = encode("ContentDate/Start desc");

        let url = format!(
            "{CDSE_PRODUCTS_SEARCH_BASE_URL}?$filter={filter}&$orderby={orderby}&$top=1&$select=Id"
        );

        let auth_header = format!("Bearer {}", self.auth_token);
        let mut response = self
            .agent
            .get(url)
            .header("Authorization", auth_header)
            .call()
            .unwrap();
        let status_code = response.status();
        if status_code == 200 {
            let json_str = response
                .body_mut()
                .read_to_string()
                .expect("Could not read response as String");
            let parsed: Value = serde_json::from_str(&json_str).unwrap();

            extract_id(&parsed).unwrap()
        } else {
            panic!("Error");
        }
    }

    /// Downloads the product with the given `uuid`, and returns the result as a String.
    fn download(&self, uuid: &str) -> String {
        let url = format!("{CDSE_PRODUCTS_DOWNLOAD_BASE_URL}({uuid})/$value");
        let auth_header = format!("Bearer {}", self.auth_token);

        let response = self
            .agent
            .get(url)
            .header("Authorization", auth_header)
            .call();

        match response {
            Ok(mut response) => {
                let status_code = response.status();
                if status_code == 200 {
                    response
                        .body_mut()
                        .read_to_string()
                        .expect("Could not read response as String")
                } else {
                    println!("status_code = {status_code}");
                    panic!("Error");
                }
            }
            Err(e) => panic!("Error {e}"),
        }
    }

    /// Searches for the precise orbital ephemerides (`AUX_POEORB`) related to the given Sentinel-1 mission
    /// which includes the [start, end] time range, and downloads them. The end result is the parsed [`EarthExplorerFile`]
    pub fn search_and_download(
        &self,
        mission: Mission,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> EarthExplorerFile {
        let uuid = self.search(mission, start, end);

        let text = self.download(&uuid);
        EarthExplorerFile::parse(&text)
    }
}

/// Extracts the product UUID from the JSON response
fn extract_id(response: &Value) -> Option<String> {
    response["value"]
        .as_array()?
        .first()?
        .get("Id")?
        .as_str()
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;

    use crate::{download_orbit::CDSEOrbitDownloader, granule_id::Mission};

    #[test]
    fn simple_orbit_info() {
        let username = std::env::var("CDSE_USERNAME").expect("expected CDSE_USERNAME");
        let password = std::env::var("CDSE_PAWSSWORD").expect("expected CDSE_PAWSSWORD");
        let client = CDSEOrbitDownloader::new(&username, &password);
        let start = DateTime::parse_from_rfc3339("2025-06-09T13:59:42Z")
            .unwrap()
            .to_utc();
        let end = DateTime::parse_from_rfc3339("2025-06-09T14:59:42Z")
            .unwrap()
            .to_utc();
        let eef = client.search_and_download(Mission::S1A, start, end);
        println!("{}", eef.earth_explorer_header.fixed_header.file_name);
    }
}
