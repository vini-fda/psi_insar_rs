//! CDSE Orbit Download
//!
//! Downloads Sentinel-1 precise orbit ephemerides (`AUX_POEORB`) from the Copernicus Data Space
//! Ecosystem (CDSE) and caches them on disk.
//!
//! ```no_run
//! use chrono::{TimeZone, Utc};
//! use psi_insar_rs::datasets::cdse::orbit_download::CdseOrbitDownloader;
//! use psi_insar_rs::granule_id::Mission;
//!
//! // Credentials default to CDSE_USERNAME/CDSE_PASSWORD (or CDSE_ACCESS_TOKEN), and orbit
//! // files are cached in `default_cache_dir()` unless `.cache_dir(...)` is given.
//! let downloader = CdseOrbitDownloader::builder().build();
//! let start = Utc.with_ymd_and_hms(2015, 10, 22, 12, 25, 46).unwrap();
//! let end = Utc.with_ymd_and_hms(2015, 10, 22, 12, 25, 49).unwrap();
//! let orbit = downloader.fetch_poe_orbit(Mission::S1A, start, end).unwrap();
//! ```
//!
//! # Caching
//!
//! Orbit files are stored as `{cache_dir}/{mission}/{product name}`, e.g.
//! `S1A/S1A_OPER_AUX_POEORB_OPOD_20210309T053936_V20151021T225943_20151023T005943.EOF`. The
//! product name encodes the validity window (`V{start}_{end}`), so with caching enabled (the
//! default) a cached file covering the requested interval is used without network access or
//! credentials. If several cached files cover it, the one produced last is used. With
//! [`CdseOrbitDownloaderBuilder::cache`]`(false)`, every request searches and downloads again
//! and replaces the cached file.
//!
//! # Requests
//!
//! 1. **Search** (catalogue, OData [\[2\]]): the newest `AUX_POEORB` product of the mission
//!    whose `ContentDate` (the validity window) covers `[start, end]`. The catalogue is public,
//!    and no token is sent: the catalogue answers `403` to an expired token.
//! 2. **Download** (`GET {download}/Products({id})/$value` with `Authorization: Bearer
//!    {token}`). This returns the EOF XML file itself; without a valid token it returns `401`.
//! 3. **Token** [\[1\]]: with a username and password, an access token is requested from the
//!    CDSE identity service (`client_id=cdse-public`, `grant_type=password`). It is reused
//!    until shortly before it expires (`expires_in`, 30 minutes when observed), and requested
//!    again once if a download answers `401`/`403`. A fixed access token
//!    (`CDSE_ACCESS_TOKEN`) cannot be renewed, so an expired one is an authentication error.
//!
//! Transport errors, `429` and `5xx` responses are retried with exponential backoff. The
//! search filter, the token request and the endpoints come from [\[1\]] and [\[2\]]; the
//! status codes, the plain EOF response and the token lifetime were observed with curl in
//! October 2026.
//!
//! # Sources
//!
//! 1. CDSE documentation, access token generation:
//!    <https://documentation.dataspace.copernicus.eu/APIs/Token.html>
//! 2. CDSE documentation, OData API:
//!    <https://documentation.dataspace.copernicus.eu/APIs/OData.html>
//!
//! [\[1\]]: https://documentation.dataspace.copernicus.eu/APIs/Token.html
//! [\[2\]]: https://documentation.dataspace.copernicus.eu/APIs/OData.html

use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDateTime, Utc};
use log::{info, warn};
use serde_json::Value;
use ureq::Body;
use ureq::http::{Response, header};

use crate::datasets::http::{
    RetryError, TempPath, USER_AGENT, body_snippet, call_with_retries, path_with_suffix,
};
use crate::{granule_id::Mission, metadata::orbit_xml::EarthExplorerFile};

/// Base URL of the CDSE OData catalogue, used for searching products.
pub const CDSE_CATALOGUE_URL: &str = "https://catalogue.dataspace.copernicus.eu/odata/v1";
/// Base URL of the CDSE OData download service.
pub const CDSE_DOWNLOAD_URL: &str = "https://download.dataspace.copernicus.eu/odata/v1";
/// CDSE identity service token endpoint.
pub const CDSE_TOKEN_URL: &str =
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token";
/// Environment variables holding the CDSE credentials.
pub const CDSE_USERNAME_VAR: &str = "CDSE_USERNAME";
pub const CDSE_PASSWORD_VAR: &str = "CDSE_PASSWORD";
pub const CDSE_ACCESS_TOKEN_VAR: &str = "CDSE_ACCESS_TOKEN";

/// Tokens are renewed this long before they expire.
const TOKEN_EXPIRY_MARGIN: Duration = Duration::from_secs(60);

/// The default cache directory: `cdse_orbits` under [`cache_root`](crate::datasets::cache_root),
/// i.e. `{user cache dir}/psi_insar_rs/cdse_orbits`.
pub fn default_cache_dir() -> PathBuf {
    crate::datasets::cache_root().join("cdse_orbits")
}

/// Error type for CDSE orbit download operations
#[derive(Debug, thiserror::Error)]
pub enum CdseOrbitError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] ureq::Error),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// A download is needed but no CDSE credentials were configured
    #[error("{0}")]
    MissingCredentials(String),
    #[error("CDSE authentication failed: {0}")]
    Authentication(String),
    /// The requested time range is invalid
    #[error("Invalid time range: {0}")]
    InvalidTimeRange(String),
    /// No precise orbit file covers the requested time range
    #[error("No AUX_POEORB orbit file covers {0}")]
    NotFound(String),
    #[error("HTTP {status} from {url}: {body}")]
    Http {
        status: u16,
        url: String,
        body: String,
    },
    /// The server answered with something unexpected
    #[error("Unexpected response: {0}")]
    InvalidResponse(String),
    #[error("Could not parse orbit file {path}: {message}")]
    Parse { path: PathBuf, message: String },
}

/// The metadata encoded in a precise orbit file name:
/// `{mission}_OPER_AUX_POEORB_OPOD_{production time}_V{validity start}_{validity end}.EOF`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoeOrbitFileName {
    pub mission: Mission,
    pub production_time: DateTime<Utc>,
    pub validity_start: DateTime<Utc>,
    pub validity_end: DateTime<Utc>,
}

impl PoeOrbitFileName {
    /// Parses a file name with the `.EOF` extension. Returns `None` for any other name.
    pub fn parse(name: &str) -> Option<Self> {
        let parts: Vec<&str> = name.strip_suffix(".EOF")?.split('_').collect();
        let [
            mission,
            "OPER",
            "AUX",
            "POEORB",
            "OPOD",
            production,
            start,
            end,
        ] = parts[..]
        else {
            return None;
        };
        Some(PoeOrbitFileName {
            mission: mission.parse().ok()?,
            production_time: parse_compact_time(production)?,
            validity_start: parse_compact_time(start.strip_prefix('V')?)?,
            validity_end: parse_compact_time(end)?,
        })
    }

    /// Whether the validity window includes `[start, end]`.
    pub fn covers(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> bool {
        self.validity_start <= start && end <= self.validity_end
    }
}

fn parse_compact_time(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y%m%dT%H%M%S")
        .ok()
        .map(|t| t.and_utc())
}

/// A product found in the CDSE catalogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbitProduct {
    /// Product UUID, used for downloading.
    pub id: String,
    /// Product (file) name.
    pub name: String,
}

enum CdseAuth {
    Password { username: String, password: String },
    AccessToken(String),
}

struct CachedToken {
    value: String,
    renew_at: Instant,
}

/// Builder for [`CdseOrbitDownloader`].
pub struct CdseOrbitDownloaderBuilder {
    auth: Option<CdseAuth>,
    cache_dir: PathBuf,
    cache_enabled: bool,
    max_retries: u32,
    initial_backoff: Duration,
    catalogue_url: String,
    download_url: String,
    token_url: String,
}

impl Default for CdseOrbitDownloaderBuilder {
    fn default() -> Self {
        let auth = match (env::var(CDSE_USERNAME_VAR), env::var(CDSE_PASSWORD_VAR)) {
            (Ok(username), Ok(password)) => Some(CdseAuth::Password { username, password }),
            _ => env::var(CDSE_ACCESS_TOKEN_VAR)
                .ok()
                .map(CdseAuth::AccessToken),
        };
        CdseOrbitDownloaderBuilder {
            auth,
            cache_dir: default_cache_dir(),
            cache_enabled: true,
            max_retries: 3,
            initial_backoff: Duration::from_secs(2),
            catalogue_url: CDSE_CATALOGUE_URL.to_string(),
            download_url: CDSE_DOWNLOAD_URL.to_string(),
            token_url: CDSE_TOKEN_URL.to_string(),
        }
    }
}

impl CdseOrbitDownloaderBuilder {
    /// CDSE username and password, used to request (and renew) access tokens. Defaults to the
    /// `CDSE_USERNAME` and `CDSE_PASSWORD` environment variables, if both are set; otherwise
    /// to a fixed `CDSE_ACCESS_TOKEN`.
    ///
    /// Credentials are only required when an orbit file has to be downloaded, so cached orbit
    /// files can be used without them.
    pub fn credentials(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.auth = Some(CdseAuth::Password {
            username: username.into(),
            password: password.into(),
        });
        self
    }

    /// A fixed access token, used instead of a username and password. It cannot be renewed,
    /// so downloads fail once it expires.
    pub fn access_token(mut self, access_token: impl Into<String>) -> Self {
        self.auth = Some(CdseAuth::AccessToken(access_token.into()));
        self
    }

    /// Directory where orbit files are stored, as `{cache_dir}/{mission}/{product name}`.
    /// Defaults to [`default_cache_dir`]. A relative path is resolved against the working
    /// directory at the time of each download.
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Whether orbit files already in the cache directory are reused (the default). When
    /// disabled, every request searches and downloads again and replaces the cached file.
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Number of retries after transport errors, `429` and `5xx` responses. Defaults to 3.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Point all requests at another server, e.g. a mock server in tests.
    #[cfg(test)]
    fn base_url(mut self, base_url: &str) -> Self {
        self.catalogue_url = format!("{base_url}/catalogue");
        self.download_url = format!("{base_url}/download");
        self.token_url = format!("{base_url}/token");
        self.initial_backoff = Duration::from_millis(10);
        self
    }

    pub fn build(self) -> CdseOrbitDownloader {
        let config = ureq::Agent::config_builder()
            .https_only(cfg!(not(test)))
            .http_status_as_error(false)
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
            .user_agent(USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .timeout_recv_body(Some(Duration::from_secs(10 * 60)))
            .build();
        CdseOrbitDownloader {
            agent: ureq::Agent::new_with_config(config),
            auth: self.auth,
            token: Mutex::new(None),
            cache_dir: self.cache_dir,
            cache_enabled: self.cache_enabled,
            max_retries: self.max_retries,
            initial_backoff: self.initial_backoff,
            catalogue_url: self.catalogue_url,
            download_url: self.download_url,
            token_url: self.token_url,
        }
    }
}

/// Obtains precise orbit files from the Copernicus Data Space Ecosystem (CDSE) and caches them
/// on disk. Create it with [`CdseOrbitDownloader::builder`].
pub struct CdseOrbitDownloader {
    agent: ureq::Agent,
    /// `None` when no credentials were configured; only an error once a download is needed.
    auth: Option<CdseAuth>,
    token: Mutex<Option<CachedToken>>,
    cache_dir: PathBuf,
    cache_enabled: bool,
    max_retries: u32,
    initial_backoff: Duration,
    catalogue_url: String,
    download_url: String,
    token_url: String,
}

impl CdseOrbitDownloader {
    pub fn builder() -> CdseOrbitDownloaderBuilder {
        CdseOrbitDownloaderBuilder::default()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Searches, downloads (unless cached) and parses the precise orbit file (`AUX_POEORB`) of
    /// `mission` whose validity window includes `[start, end]`.
    pub fn fetch_poe_orbit(
        &self,
        mission: Mission,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<EarthExplorerFile, CdseOrbitError> {
        let path = self.fetch_poe_orbit_file(mission, start, end)?;
        let text = fs::read_to_string(&path)?;
        text.parse::<EarthExplorerFile>()
            .map_err(|err| CdseOrbitError::Parse {
                path,
                message: format!("{err:?}"),
            })
    }

    /// Like [`fetch_poe_orbit`](Self::fetch_poe_orbit), but returns the path of the orbit file
    /// in the cache directory instead of parsing it.
    ///
    /// With caching enabled, a cached file covering `[start, end]` is returned without network
    /// access or credentials.
    pub fn fetch_poe_orbit_file(
        &self,
        mission: Mission,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<PathBuf, CdseOrbitError> {
        if start > end {
            return Err(CdseOrbitError::InvalidTimeRange(format!(
                "start {start} is after end {end}"
            )));
        }
        if self.cache_enabled
            && let Some(path) = self.find_cached(mission, start, end)
        {
            info!("Using cached orbit file {}", path.display());
            return Ok(path);
        }

        let product = self.search(mission, start, end)?;
        // The name is used as a file name, so only accept well-formed orbit file names.
        match PoeOrbitFileName::parse(&product.name) {
            Some(name) if name.mission == mission && name.covers(start, end) => {}
            _ => {
                return Err(CdseOrbitError::InvalidResponse(format!(
                    "The catalogue returned {:?}, which is not a {mission} POEORB file covering \
                     {start} to {end}",
                    product.name
                )));
            }
        }
        let path = self.cache_dir.join(mission.to_string()).join(&product.name);
        self.download(&product, &path)?;
        info!("Downloaded orbit file {}", path.display());
        Ok(path)
    }

    /// The cached orbit file of `mission` covering `[start, end]`, if any. If several cover
    /// it, the one produced last is returned.
    pub fn find_cached(
        &self,
        mission: Mission,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Option<PathBuf> {
        let entries = fs::read_dir(self.cache_dir.join(mission.to_string())).ok()?;
        entries
            .flatten()
            .filter_map(|entry| {
                let file_name = entry.file_name().into_string().ok()?;
                let name = PoeOrbitFileName::parse(&file_name)?;
                let is_file = entry.file_type().is_ok_and(|t| t.is_file());
                (is_file && name.mission == mission && name.covers(start, end))
                    .then(|| (name.production_time, entry.path()))
            })
            .max()
            .map(|(_, path)| path)
    }

    /// Searches the CDSE catalogue for the newest precise orbit file (`AUX_POEORB`) of
    /// `mission` whose validity window includes `[start, end]`.
    pub fn search(
        &self,
        mission: Mission,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<OrbitProduct, CdseOrbitError> {
        let start_rfc3339 = start.to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
        let end_rfc3339 = end.to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
        let filter = format!(
            "Online eq true and \
            startswith(Name,'{mission}_') and \
            Attributes/OData.CSC.StringAttribute/any(i0:i0/Name eq 'productType' and i0/Value eq 'AUX_POEORB') and \
            Collection/Name eq 'SENTINEL-1' and \
            ContentDate/Start le {start_rfc3339} and ContentDate/End ge {end_rfc3339}"
        );
        let url = format!("{}/Products", self.catalogue_url);
        // No token: the catalogue is public, and answers 403 to an expired token.
        let mut resp = self.call_with_retries("orbit search", || {
            self.agent
                .get(&url)
                .query("$filter", &filter)
                .query("$orderby", "ContentDate/Start desc")
                .query("$top", "1")
                .query("$select", "Id,Name")
                .call()
        })?;
        if resp.status() != 200 {
            return Err(http_error(&url, &mut resp));
        }
        let json: Value = resp.body_mut().read_json()?;
        let products = json["value"].as_array().ok_or_else(|| {
            CdseOrbitError::InvalidResponse(format!("Search response without `value`: {json}"))
        })?;
        let Some(product) = products.first() else {
            return Err(CdseOrbitError::NotFound(format!(
                "{mission} from {start} to {end}"
            )));
        };
        match (product["Id"].as_str(), product["Name"].as_str()) {
            (Some(id), Some(name)) => Ok(OrbitProduct {
                id: id.to_string(),
                name: name.to_string(),
            }),
            _ => Err(CdseOrbitError::InvalidResponse(format!(
                "Search result without Id or Name: {product}"
            ))),
        }
    }

    /// Downloads `product` to `path`, through a temporary file that is only renamed to `path`
    /// once complete.
    fn download(&self, product: &OrbitProduct, path: &Path) -> Result<(), CdseOrbitError> {
        let url = format!("{}/Products({})/$value", self.download_url, product.id);
        let mut renewed_token = false;
        let mut resp = loop {
            let token = self.access_token(&url)?;
            let mut resp = self.call_with_retries("orbit download", || {
                self.agent
                    .get(&url)
                    .header(header::AUTHORIZATION, format!("Bearer {token}"))
                    .call()
            })?;
            match resp.status().as_u16() {
                200 => break resp,
                401 | 403 if !renewed_token && self.can_renew_token() => {
                    warn!("CDSE rejected the access token, requesting a new one");
                    *self.token.lock().unwrap() = None;
                    renewed_token = true;
                }
                401 | 403 => {
                    let hint = if self.can_renew_token() {
                        ""
                    } else {
                        " (a fixed access token cannot be renewed; set CDSE_USERNAME and \
                         CDSE_PASSWORD instead)"
                    };
                    return Err(CdseOrbitError::Authentication(format!(
                        "{}{hint}",
                        http_error(&url, &mut resp)
                    )));
                }
                _ => return Err(http_error(&url, &mut resp)),
            }
        };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let part_path = TempPath(path_with_suffix(path, ".part"));
        let expected_len = resp.body().content_length();
        // Read/write: the file is checked after writing it.
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&part_path.0)?;
        let written = io::copy(&mut resp.body_mut().as_reader(), &mut file)?;
        if let Some(expected_len) = expected_len
            && written != expected_len
        {
            return Err(CdseOrbitError::InvalidResponse(format!(
                "Truncated download of {}: got {written} of {expected_len} bytes",
                product.name
            )));
        }
        check_earth_explorer_file(&mut file, &product.name)?;
        drop(file);
        if path.exists() {
            // Only reached when caching is disabled, or for a file that does not cover the
            // requested interval according to its name.
            fs::remove_file(path)?;
        }
        fs::rename(&part_path.0, path)?;
        Ok(())
    }

    fn can_renew_token(&self) -> bool {
        matches!(self.auth, Some(CdseAuth::Password { .. }))
    }

    /// An access token for downloading `url`, requesting a new one when needed.
    fn access_token(&self, url: &str) -> Result<String, CdseOrbitError> {
        let (username, password) = match &self.auth {
            None => {
                return Err(CdseOrbitError::MissingCredentials(format!(
                    "Downloading {url} requires CDSE credentials: set {CDSE_USERNAME_VAR} and \
                     {CDSE_PASSWORD_VAR} (or {CDSE_ACCESS_TOKEN_VAR}), or call \
                     CdseOrbitDownloaderBuilder::credentials"
                )));
            }
            Some(CdseAuth::AccessToken(token)) => return Ok(token.clone()),
            Some(CdseAuth::Password { username, password }) => (username, password),
        };
        let mut cached = self.token.lock().unwrap();
        if let Some(token) = cached.as_ref()
            && Instant::now() < token.renew_at
        {
            return Ok(token.value.clone());
        }

        let mut resp = self.call_with_retries("token request", || {
            self.agent.post(&self.token_url).send_form([
                ("client_id", "cdse-public"),
                ("grant_type", "password"),
                ("username", username.as_str()),
                ("password", password.as_str()),
            ])
        })?;
        match resp.status().as_u16() {
            200 => {}
            400 | 401 => {
                return Err(CdseOrbitError::Authentication(format!(
                    "CDSE rejected the username/password: {}",
                    body_snippet(&mut resp)
                )));
            }
            _ => return Err(http_error(&self.token_url, &mut resp)),
        }
        let json: Value = resp.body_mut().read_json()?;
        let value = json["access_token"].as_str().ok_or_else(|| {
            CdseOrbitError::InvalidResponse("Token response without access_token".to_string())
        })?;
        let lifetime = Duration::from_secs(json["expires_in"].as_u64().unwrap_or(0));
        *cached = Some(CachedToken {
            value: value.to_string(),
            renew_at: Instant::now() + lifetime.saturating_sub(TOKEN_EXPIRY_MARGIN),
        });
        Ok(value.to_string())
    }

    /// [`call_with_retries`] with this downloader's retry settings.
    fn call_with_retries(
        &self,
        what: &str,
        request: impl FnMut() -> Result<Response<Body>, ureq::Error>,
    ) -> Result<Response<Body>, CdseOrbitError> {
        call_with_retries(what, self.max_retries, self.initial_backoff, request).map_err(|err| {
            match err {
                RetryError::Request(err) => err.into(),
                RetryError::Exhausted {
                    retries,
                    last_error,
                } => CdseOrbitError::InvalidResponse(format!(
                    "{what} failed after {retries} retries: {last_error}"
                )),
            }
        })
    }
}

fn http_error(url: &str, resp: &mut Response<Body>) -> CdseOrbitError {
    CdseOrbitError::Http {
        status: resp.status().as_u16(),
        url: url.to_string(),
        body: body_snippet(resp),
    }
}

/// Cheap check that `file` holds a complete Earth Explorer XML file, without parsing it.
fn check_earth_explorer_file(file: &mut File, name: &str) -> Result<(), CdseOrbitError> {
    let mut head = [0; 256];
    file.seek(SeekFrom::Start(0))?;
    let head_len = file.read(&mut head)?;
    let head = String::from_utf8_lossy(&head[..head_len]);

    let len = file.seek(SeekFrom::End(0))?;
    let mut tail = Vec::new();
    file.seek(SeekFrom::Start(len.saturating_sub(256)))?;
    file.read_to_end(&mut tail)?;
    let tail = String::from_utf8_lossy(&tail);

    if head.contains("<Earth_Explorer_File") && tail.trim_end().ends_with("</Earth_Explorer_File>")
    {
        Ok(())
    } else {
        Err(CdseOrbitError::InvalidResponse(format!(
            "{name} is not a complete Earth Explorer file, it starts with: {head:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::mock_server::{MockResponse, mock_server};

    const ORBIT_NAME: &str =
        "S1A_OPER_AUX_POEORB_OPOD_20210309T053936_V20151021T225943_20151023T005943.EOF";

    fn utc(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().to_utc()
    }

    /// The acquisition window of the 2015-10-22 reference burst used in the rerun tests.
    fn burst_window() -> (DateTime<Utc>, DateTime<Utc>) {
        (utc("2015-10-22T12:25:46Z"), utc("2015-10-22T12:25:49Z"))
    }

    fn orbit_example() -> Vec<u8> {
        fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/metadata/test_data/orbit_example.xml"
        ))
        .unwrap()
    }

    fn search_result(name: &str) -> MockResponse {
        MockResponse::json_body(
            200,
            format!(r#"{{"value":[{{"Id":"uuid-1","Name":"{name}"}}]}}"#),
        )
    }

    fn token_response(token: &str) -> MockResponse {
        MockResponse::json_body(
            200,
            format!(r#"{{"access_token":"{token}","expires_in":1800,"token_type":"Bearer"}}"#),
        )
    }

    fn eof_response() -> MockResponse {
        MockResponse::bytes("application/octet-stream", orbit_example())
    }

    /// A downloader with fake credentials, caching into a fresh directory named `name`.
    fn test_builder(name: &str, base: &str) -> CdseOrbitDownloaderBuilder {
        let cache_dir = env::temp_dir()
            .join("psi_insar_rs_cdse_orbit_download")
            .join(name);
        let _ = fs::remove_dir_all(&cache_dir);
        CdseOrbitDownloader::builder()
            .credentials("user", "pass")
            .cache_dir(cache_dir)
            .max_retries(2)
            .base_url(base)
    }

    #[test]
    fn parses_orbit_file_names() {
        let name = PoeOrbitFileName::parse(ORBIT_NAME).unwrap();
        assert_eq!(name.mission, Mission::S1A);
        assert_eq!(name.production_time, utc("2021-03-09T05:39:36Z"));
        assert_eq!(name.validity_start, utc("2015-10-21T22:59:43Z"));
        assert_eq!(name.validity_end, utc("2015-10-23T00:59:43Z"));
        let (start, end) = burst_window();
        assert!(name.covers(start, end));
        assert!(!name.covers(utc("2015-10-21T22:00:00Z"), end));
        assert!(!name.covers(start, utc("2015-10-23T01:00:00Z")));

        for invalid in [
            "S1A_OPER_AUX_POEORB_OPOD_20210309T053936_V20151021T225943_20151023T005943.xml",
            "S1A_OPER_AUX_RESORB_OPOD_20210309T053936_V20151021T225943_20151023T005943.EOF",
            "S1X_OPER_AUX_POEORB_OPOD_20210309T053936_V20151021T225943_20151023T005943.EOF",
            "../S1A_OPER_AUX_POEORB_OPOD_20210309T053936_V20151021T225943_20151023T005943.EOF",
            "S1A_OPER_AUX_POEORB_OPOD_20210309T053936_20151021T225943_20151023T005943.EOF",
        ] {
            assert_eq!(PoeOrbitFileName::parse(invalid), None, "{invalid}");
        }
    }

    #[test]
    fn downloads_parses_and_caches_orbit_file() {
        let (base, requests) = mock_server(vec![
            search_result(ORBIT_NAME),
            token_response("tok-1"),
            eof_response(),
        ]);
        let downloader = test_builder("download", &base).build();
        let (start, end) = burst_window();
        let orbit = downloader
            .fetch_poe_orbit(Mission::S1A, start, end)
            .unwrap();
        assert_eq!(
            orbit.earth_explorer_header.fixed_header.file_name,
            ORBIT_NAME.trim_end_matches(".EOF")
        );
        let path = downloader.cache_dir().join("S1A").join(ORBIT_NAME);
        assert!(path.is_file());

        let requests = requests.all();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].path(), "/catalogue/Products");
        let query = requests[0].target.split_once('?').unwrap().1;
        let filter = url::form_urlencoded::parse(query.as_bytes())
            .find(|(key, _)| key == "$filter")
            .unwrap()
            .1;
        assert!(filter.contains(
            "ContentDate/Start le 2015-10-22T12:25:46.000000Z and \
             ContentDate/End ge 2015-10-22T12:25:49.000000Z"
        ));
        assert_eq!(
            requests[0].authorization, None,
            "search must not send a token"
        );
        assert_eq!(requests[1].method, "POST");
        assert_eq!(requests[1].path(), "/token");
        assert!(requests[1].body.contains("grant_type=password"));
        assert_eq!(requests[2].path(), "/download/Products(uuid-1)/$value");
        assert_eq!(requests[2].authorization.as_deref(), Some("Bearer tok-1"));

        // Cached: the server has no responses left.
        assert_eq!(
            downloader
                .fetch_poe_orbit_file(Mission::S1A, start, end)
                .unwrap(),
            path
        );
    }

    #[test]
    fn cached_orbit_files_need_no_credentials() {
        let (base, requests) = mock_server(vec![]);
        let mut builder = test_builder("no_credentials", &base);
        builder.auth = None;
        let downloader = builder.build();
        let older = ORBIT_NAME.replace("20210309T053936", "20151111T121612");
        let mission_dir = downloader.cache_dir().join("S1A");
        fs::create_dir_all(&mission_dir).unwrap();
        fs::write(mission_dir.join(&older), "").unwrap();
        fs::write(mission_dir.join(ORBIT_NAME), "").unwrap();

        let (start, end) = burst_window();
        // The file produced last wins.
        assert_eq!(
            downloader
                .fetch_poe_orbit_file(Mission::S1A, start, end)
                .unwrap(),
            mission_dir.join(ORBIT_NAME)
        );
        assert_eq!(requests.len(), 0);
    }

    #[test]
    fn missing_credentials_fail_before_download() {
        let (base, requests) = mock_server(vec![search_result(ORBIT_NAME)]);
        let mut builder = test_builder("missing_credentials", &base);
        builder.auth = None;
        let (start, end) = burst_window();
        let result = builder
            .build()
            .fetch_poe_orbit_file(Mission::S1A, start, end);
        assert!(matches!(result, Err(CdseOrbitError::MissingCredentials(_))));
        assert_eq!(requests.paths(), ["/catalogue/Products"]);
    }

    #[test]
    fn disabled_cache_downloads_again_and_replaces_file() {
        let (base, requests) = mock_server(vec![
            search_result(ORBIT_NAME),
            token_response("tok-1"),
            eof_response(),
            search_result(ORBIT_NAME),
            eof_response(),
        ]);
        let downloader = test_builder("uncached", &base).cache(false).build();
        let (start, end) = burst_window();
        let path = downloader
            .fetch_poe_orbit_file(Mission::S1A, start, end)
            .unwrap();
        fs::write(&path, "stale").unwrap();
        assert_eq!(
            downloader
                .fetch_poe_orbit_file(Mission::S1A, start, end)
                .unwrap(),
            path
        );
        assert_eq!(fs::read(&path).unwrap(), orbit_example());
        // The token is reused for the second download.
        assert_eq!(requests.len(), 5);
    }

    #[test]
    fn renews_rejected_token_once() {
        let (base, requests) = mock_server(vec![
            search_result(ORBIT_NAME),
            token_response("tok-1"),
            MockResponse::json(401, "Expired"),
            token_response("tok-2"),
            eof_response(),
        ]);
        let downloader = test_builder("renew", &base).build();
        let (start, end) = burst_window();
        downloader
            .fetch_poe_orbit_file(Mission::S1A, start, end)
            .unwrap();
        let requests = requests.all();
        assert_eq!(requests[4].authorization.as_deref(), Some("Bearer tok-2"));
    }

    #[test]
    fn fixed_token_is_not_renewed() {
        let (base, requests) = mock_server(vec![
            search_result(ORBIT_NAME),
            MockResponse::json(401, "Expired"),
        ]);
        let downloader = test_builder("fixed_token", &base)
            .access_token("expired")
            .build();
        let (start, end) = burst_window();
        let result = downloader.fetch_poe_orbit_file(Mission::S1A, start, end);
        assert!(matches!(result, Err(CdseOrbitError::Authentication(_))));
        assert_eq!(requests.len(), 2);
        assert!(!downloader.cache_dir().join("S1A").join(ORBIT_NAME).exists());
    }

    #[test]
    fn rejected_credentials_are_an_authentication_error() {
        let (base, _) = mock_server(vec![
            search_result(ORBIT_NAME),
            MockResponse::json_body(401, r#"{"error":"invalid_grant"}"#),
        ]);
        let downloader = test_builder("bad_credentials", &base).build();
        let (start, end) = burst_window();
        let result = downloader.fetch_poe_orbit_file(Mission::S1A, start, end);
        assert!(matches!(result, Err(CdseOrbitError::Authentication(_))));
    }

    #[test]
    fn retries_transient_errors() {
        let (base, requests) = mock_server(vec![
            MockResponse::json(503, "Service Unavailable"),
            search_result(ORBIT_NAME),
            token_response("tok-1"),
            MockResponse::json(500, "Internal Server Error"),
            eof_response(),
        ]);
        let downloader = test_builder("transient", &base).build();
        let (start, end) = burst_window();
        downloader
            .fetch_poe_orbit_file(Mission::S1A, start, end)
            .unwrap();
        assert_eq!(requests.len(), 5);
    }

    #[test]
    fn empty_search_is_not_found() {
        let (base, _) = mock_server(vec![MockResponse::json_body(200, r#"{"value":[]}"#)]);
        let downloader = test_builder("not_found", &base).build();
        let (start, end) = burst_window();
        let result = downloader.fetch_poe_orbit_file(Mission::S1A, start, end);
        assert!(matches!(result, Err(CdseOrbitError::NotFound(_))));
    }

    #[test]
    fn incomplete_file_is_rejected_without_output() {
        let mut truncated = orbit_example();
        truncated.truncate(10_000);
        let (base, _) = mock_server(vec![
            search_result(ORBIT_NAME),
            token_response("tok-1"),
            MockResponse::bytes("application/octet-stream", truncated),
        ]);
        let downloader = test_builder("incomplete", &base).build();
        let (start, end) = burst_window();
        let result = downloader.fetch_poe_orbit_file(Mission::S1A, start, end);
        assert!(matches!(result, Err(CdseOrbitError::InvalidResponse(_))));
        let mission_dir = downloader.cache_dir().join("S1A");
        assert_eq!(fs::read_dir(mission_dir).unwrap().count(), 0);
    }

    #[test]
    fn invalid_time_range_is_rejected() {
        let downloader = CdseOrbitDownloader::builder().build();
        let (start, end) = burst_window();
        let result = downloader.fetch_poe_orbit_file(Mission::S1A, end, start);
        assert!(matches!(result, Err(CdseOrbitError::InvalidTimeRange(_))));
    }

    // Needs CDSE_USERNAME and CDSE_PASSWORD (or a valid CDSE_ACCESS_TOKEN). Downloads the orbit
    // file of the 2015-10-22 reference burst with caching disabled, replacing the copy in
    // default_cache_dir().
    #[test]
    #[ignore = "Needs to download external data"]
    fn simple_orbit_info() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Info)
            .try_init();
        let downloader = CdseOrbitDownloader::builder().cache(false).build();
        let (start, end) = burst_window();
        let eef = downloader
            .fetch_poe_orbit(Mission::S1A, start, end)
            .unwrap();
        println!("{}", eef.earth_explorer_header.fixed_header.file_name);
        assert!(downloader.find_cached(Mission::S1A, start, end).is_some());
    }
}
