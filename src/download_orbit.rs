use chrono::{DateTime, Utc};
use serde_json::Value;
use urlencoding::encode;

use crate::{granule_id::Mission, metadata::orbit_xml::EarthExplorerFile};

/// Base URL for searching CDSE Products
const CDSE_PRODUCTS_SEARCH_BASE_URL: &str =
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products";
/// Base URL for downloading CDSE Products
const CDSE_PRODUCTS_DOWNLOAD_BASE_URL: &str =
    "https://download.dataspace.copernicus.eu/odata/v1/Products";

/// Obtains precise orbit files from the Copernicus Data Space Ecosystem (CDSE)
pub struct CDSEOrbitDownloader {
    agent: ureq::Agent,
    access_token: String,
}

impl CDSEOrbitDownloader {
    /// Creates a new ``CDSEOrbitDownloader`` instance.
    ///
    /// # Panics
    ///
    /// Panics if the `CDSE_ACCESS_TOKEN` environment variable is not set.
    #[must_use]
    pub fn new() -> Self {
        let access_token = std::env::var("CDSE_ACCESS_TOKEN").expect(
            "The CDSEOrbitDownloader needs the CDSE_ACCESS_TOKEN environment variable set up.",
        );
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .https_only(true)
            .http_status_as_error(false)
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
            .build();

        let agent = ureq::Agent::new_with_config(config);
        Self {
            agent,
            access_token,
        }
    }

    /// Searches for the precise orbital ephemerides (`AUX_POEORB`) related to the given Sentinel-1 mission
    /// which includes the [start, end] time range.
    ///
    /// # Panics
    ///
    /// Panics if the `start` datetime is after the `end` datetime.
    #[must_use]
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

        let auth_header = format!("Bearer {}", self.access_token);
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
        let auth_header = format!("Bearer {}", self.access_token);

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
    ///
    /// # Panics
    ///
    /// Panics if the search or download fails.
    #[must_use]
    pub fn search_and_download(
        &self,
        mission: Mission,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> EarthExplorerFile {
        let uuid = self.search(mission, start, end);

        let text = self.download(&uuid);
        text.parse::<EarthExplorerFile>()
            .expect("Failed to parse Earth Explorer File")
    }
}

impl Default for CDSEOrbitDownloader {
    fn default() -> Self {
        Self::new()
    }
}

/// Extracts the product UUID from the JSON response
fn extract_id(response: &Value) -> Option<String> {
    response["value"]
        .as_array()?
        .first()?
        .get("Id")?
        .as_str()
        .map(std::string::ToString::to_string)
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;

    use crate::{download_orbit::CDSEOrbitDownloader, granule_id::Mission};

    #[test]
    #[ignore = "Needs to download external data"]
    fn simple_orbit_info() {
        let client = CDSEOrbitDownloader::new();
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
