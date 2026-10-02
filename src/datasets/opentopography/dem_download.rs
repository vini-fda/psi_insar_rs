//! OpenTopography DEM Download
//!
//! Downloads Copernicus DEM rasters from the OpenTopography global DEM API and caches them on
//! disk.
//!
//! ```no_run
//! use psi_insar_rs::datasets::opentopography::dem_download::{
//!     CopernicusDemType, OpenTopographyDemDownloader,
//! };
//!
//! // The API key defaults to OPENTOPOGRAPHY_API_KEY, and rasters are cached in
//! // `default_cache_dir()` unless `.cache_dir(...)` is given.
//! let downloader = OpenTopographyDemDownloader::builder().build();
//! // [min_lat, max_lat, min_lon, max_lon]
//! let bounds = [19.28, 19.68, -99.41, -98.54];
//! let dem = downloader.fetch_dem(bounds, CopernicusDemType::Cop90).unwrap();
//! ```
//!
//! # Caching
//!
//! Rasters are stored as `{cache_dir}/{dem type}/S{south}_N{north}_W{west}_E{east}.tif`, with
//! the bounds written exactly as they are sent to the API. Only a raster with exactly the
//! requested bounds is reused, not one covering a larger area, because a larger DEM changes
//! the results of code that iterates over all DEM pixels. With caching enabled (the default),
//! a cached raster is used without network access or an API key. With
//! [`OpenTopographyDemDownloaderBuilder::cache`]`(false)`, every request downloads the raster
//! again and replaces the cached file.
//!
//! # Requests
//!
//! `GET {api}/globaldem?demtype=..&south=..&north=..&west=..&east=..&outputFormat=GTiff&API_Key=..`
//! returns, according to the API specification [\[1\]]:
//!
//! | Response | Meaning     | Action                                    |
//! |----------|-------------|-------------------------------------------|
//! | `200`    | OK          | Stream to disk, check it is a GeoTIFF     |
//! | `204`    | No Data     | Fail: [`DemDownloadError::NoData`]        |
//! | `400`    | Bad request | Fail (e.g. `north` <= `south`, area too large) |
//! | `401`    | Unauthorized| Fail: missing or invalid API key          |
//! | `500`    | Internal error | Retry with exponential backoff         |
//!
//! Transport errors and `429` are retried too. [\[1\]] limits requests to 450,000 km² for
//! COP30 and 4,050,000 km² for COP90. Observed with curl in October 2026: error bodies are
//! XML (`<error>Error: ...</error>`), the `401` body repeats the API key that was sent, and a
//! small request took about 10 s.
//!
//! The API key is part of the request URL, so URLs are never logged, and the key is removed
//! from error messages.
//!
//! # Sources
//!
//! 1. OpenTopography API specification, `globaldem` endpoint:
//!    <https://portal.opentopography.org/apidocs/#/Public/getGlobalDem> (OpenAPI document:
//!    <https://portal.opentopography.org/apidocs/openapi.json>)
//!
//! [\[1\]]: https://portal.opentopography.org/apidocs/#/Public/getGlobalDem

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use log::info;
use ureq::Body;
use ureq::http::Response;

use crate::datasets::http::{
    RetryError, TempPath, USER_AGENT, body_snippet, call_with_retries, path_with_suffix,
};
use crate::dem::DEM;

/// Base URL of the OpenTopography API.
pub const OPENTOPOGRAPHY_API_URL: &str = "https://portal.opentopography.org/API";
/// Environment variable holding the OpenTopography API key.
pub const OPENTOPOGRAPHY_API_KEY_VAR: &str = "OPENTOPOGRAPHY_API_KEY";

const TIFF_MAGIC: [&[u8; 4]; 2] = [b"II*\0", b"MM\0*"];

/// The default cache directory: `opentopography_dem` under
/// [`cache_root`](crate::datasets::cache_root), i.e.
/// `{user cache dir}/psi_insar_rs/opentopography_dem`.
pub fn default_cache_dir() -> PathBuf {
    crate::datasets::cache_root().join("opentopography_dem")
}

/// Copernicus DEM products available from the OpenTopography global DEM API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopernicusDemType {
    /// Copernicus GLO-30 (30 m)
    Cop30,
    /// Copernicus GLO-90 (90 m)
    Cop90,
}

impl std::fmt::Display for CopernicusDemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopernicusDemType::Cop30 => write!(f, "COP30"),
            CopernicusDemType::Cop90 => write!(f, "COP90"),
        }
    }
}

/// Error type for OpenTopography DEM download operations. Messages never contain the API key.
#[derive(Debug, thiserror::Error)]
pub enum DemDownloadError {
    #[error("HTTP request failed: {0}")]
    Request(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// A download is needed but no API key was configured
    #[error("{0}")]
    MissingApiKey(String),
    /// The bounding box is invalid
    #[error("Invalid bounds: {0}")]
    InvalidBounds(String),
    /// OpenTopography has no data for the bounding box (HTTP 204)
    #[error("No DEM data for {0}")]
    NoData(String),
    /// The API key is missing or invalid (HTTP 401)
    #[error("OpenTopography rejected the API key: {0}")]
    Authentication(String),
    #[error("HTTP {status} from OpenTopography: {body}")]
    Http { status: u16, body: String },
    /// The server answered with something unexpected
    #[error("Unexpected response: {0}")]
    InvalidResponse(String),
}

/// Builder for [`OpenTopographyDemDownloader`].
pub struct OpenTopographyDemDownloaderBuilder {
    api_key: Option<String>,
    cache_dir: PathBuf,
    cache_enabled: bool,
    max_retries: u32,
    initial_backoff: Duration,
    api_url: String,
}

impl Default for OpenTopographyDemDownloaderBuilder {
    fn default() -> Self {
        OpenTopographyDemDownloaderBuilder {
            api_key: env::var(OPENTOPOGRAPHY_API_KEY_VAR).ok(),
            cache_dir: default_cache_dir(),
            cache_enabled: true,
            max_retries: 3,
            initial_backoff: Duration::from_secs(2),
            api_url: OPENTOPOGRAPHY_API_URL.to_string(),
        }
    }
}

impl OpenTopographyDemDownloaderBuilder {
    /// OpenTopography API key. Defaults to the `OPENTOPOGRAPHY_API_KEY` environment variable.
    ///
    /// The key is only required when a raster has to be downloaded, so cached rasters can be
    /// used without it.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Directory where rasters are stored, as
    /// `{cache_dir}/{dem type}/S{south}_N{north}_W{west}_E{east}.tif`. Defaults to
    /// [`default_cache_dir`]. A relative path is resolved against the working directory at the
    /// time of each download.
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Whether rasters already in the cache directory are reused (the default). When disabled,
    /// every request downloads the raster again and replaces the cached file.
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Number of retries after transport errors, `429` and `5xx` responses. Defaults to 3.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Point requests at another server, e.g. a mock server in tests.
    #[cfg(test)]
    fn api_url(mut self, api_url: &str) -> Self {
        self.api_url = api_url.to_string();
        self.initial_backoff = Duration::from_millis(10);
        self
    }

    pub fn build(self) -> OpenTopographyDemDownloader {
        let config = ureq::Agent::config_builder()
            .https_only(cfg!(not(test)))
            .http_status_as_error(false)
            .user_agent(USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(5 * 60)))
            .timeout_recv_body(Some(Duration::from_secs(30 * 60)))
            .build();
        OpenTopographyDemDownloader {
            agent: ureq::Agent::new_with_config(config),
            api_key: self.api_key,
            cache_dir: self.cache_dir,
            cache_enabled: self.cache_enabled,
            max_retries: self.max_retries,
            initial_backoff: self.initial_backoff,
            api_url: self.api_url,
        }
    }
}

/// Downloads DEM rasters from OpenTopography and caches them on disk. Create it with
/// [`OpenTopographyDemDownloader::builder`].
pub struct OpenTopographyDemDownloader {
    agent: ureq::Agent,
    /// `None` when no API key was configured; only an error once a download is needed.
    api_key: Option<String>,
    cache_dir: PathBuf,
    cache_enabled: bool,
    max_retries: u32,
    initial_backoff: Duration,
    api_url: String,
}

impl OpenTopographyDemDownloader {
    pub fn builder() -> OpenTopographyDemDownloaderBuilder {
        OpenTopographyDemDownloaderBuilder::default()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Where the raster for `bounds` (`[min_lat, max_lat, min_lon, max_lon]`) is stored.
    pub fn cached_path(&self, bounds: [f64; 4], dem_type: CopernicusDemType) -> PathBuf {
        let [south, north, west, east] = bounds;
        self.cache_dir
            .join(dem_type.to_string())
            .join(format!("S{south}_N{north}_W{west}_E{east}.tif"))
    }

    /// Downloads (unless cached) and opens the DEM covering `bounds`
    /// (`[min_lat, max_lat, min_lon, max_lon]`, WGS 84 degrees).
    pub fn fetch_dem(
        &self,
        bounds: [f64; 4],
        dem_type: CopernicusDemType,
    ) -> Result<DEM, DemDownloadError> {
        let path = self.fetch_dem_file(bounds, dem_type)?;
        Ok(DEM::open_file(path))
    }

    /// Like [`fetch_dem`](Self::fetch_dem), but returns the path of the GeoTIFF in the cache
    /// directory instead of opening it.
    ///
    /// With caching enabled, a cached raster is returned without network access or API key.
    pub fn fetch_dem_file(
        &self,
        bounds: [f64; 4],
        dem_type: CopernicusDemType,
    ) -> Result<PathBuf, DemDownloadError> {
        validate_bounds(bounds)?;
        let path = self.cached_path(bounds, dem_type);
        if self.cache_enabled && path.is_file() {
            info!("Using cached DEM {}", path.display());
            return Ok(path);
        }
        let Some(api_key) = &self.api_key else {
            return Err(DemDownloadError::MissingApiKey(format!(
                "Downloading the {dem_type} DEM for {bounds:?} requires an OpenTopography API \
                 key: set {OPENTOPOGRAPHY_API_KEY_VAR}, or call \
                 OpenTopographyDemDownloaderBuilder::api_key"
            )));
        };
        let redact = |message: String| message.replace(api_key.as_str(), "<API key>");

        let [south, north, west, east] = bounds;
        let url = format!("{}/globaldem", self.api_url);
        let mut resp = call_with_retries(
            "DEM download",
            self.max_retries,
            self.initial_backoff,
            || {
                self.agent
                    .get(&url)
                    .query("demtype", dem_type.to_string())
                    .query("south", south.to_string())
                    .query("north", north.to_string())
                    .query("west", west.to_string())
                    .query("east", east.to_string())
                    .query("outputFormat", "GTiff")
                    .query("API_Key", api_key)
                    .call()
            },
        )
        .map_err(|err| match err {
            RetryError::Request(err) => DemDownloadError::Request(redact(err.to_string())),
            RetryError::Exhausted {
                retries,
                last_error,
            } => DemDownloadError::Request(redact(format!(
                "failed after {retries} retries: {last_error}"
            ))),
        })?;
        match resp.status().as_u16() {
            200 => {}
            204 => {
                return Err(DemDownloadError::NoData(format!("{dem_type} {bounds:?}")));
            }
            401 => {
                return Err(DemDownloadError::Authentication(redact(body_snippet(
                    &mut resp,
                ))));
            }
            status => {
                return Err(DemDownloadError::Http {
                    status,
                    body: redact(body_snippet(&mut resp)),
                });
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        self.save_geotiff(&mut resp, &path)?;
        info!("Downloaded DEM {}", path.display());
        Ok(path)
    }

    /// Streams the response to a temporary file, checks that it is a readable GeoTIFF, and
    /// only then moves it to `path`.
    fn save_geotiff(&self, resp: &mut Response<Body>, path: &Path) -> Result<(), DemDownloadError> {
        let part_path = TempPath(path_with_suffix(path, ".part"));
        let expected_len = resp.body().content_length();
        let written = io::copy(
            &mut resp.body_mut().as_reader(),
            &mut File::create(&part_path.0)?,
        )?;
        if let Some(expected_len) = expected_len
            && written != expected_len
        {
            return Err(DemDownloadError::InvalidResponse(format!(
                "Truncated download: got {written} of {expected_len} bytes"
            )));
        }

        let mut head = Vec::with_capacity(256);
        File::open(&part_path.0)?.take(256).read_to_end(&mut head)?;
        if !TIFF_MAGIC.iter().any(|magic| head.starts_with(*magic)) {
            let start = String::from_utf8_lossy(&head);
            return Err(DemDownloadError::InvalidResponse(format!(
                "Expected a GeoTIFF, got: {start:?}"
            )));
        }
        geotiff::GeoTiff::read(File::open(&part_path.0)?).map_err(|err| {
            DemDownloadError::InvalidResponse(format!("Unreadable GeoTIFF: {err:?}"))
        })?;

        if path.exists() {
            // Only reached when caching is disabled.
            fs::remove_file(path)?;
        }
        fs::rename(&part_path.0, path)?;
        Ok(())
    }
}

fn validate_bounds(bounds: [f64; 4]) -> Result<(), DemDownloadError> {
    let [south, north, west, east] = bounds;
    let error = if bounds.iter().any(|v| !v.is_finite()) {
        Some("all bounds must be finite")
    } else if !(-90.0..=90.0).contains(&south) || !(-90.0..=90.0).contains(&north) {
        Some("latitudes must be in [-90, 90]")
    } else if !(-180.0..=180.0).contains(&west) || !(-180.0..=180.0).contains(&east) {
        Some("longitudes must be in [-180, 180]")
    } else if south >= north {
        Some("min_lat must be less than max_lat")
    } else if west >= east {
        Some("min_lon must be less than max_lon")
    } else {
        None
    };
    match error {
        Some(error) => Err(DemDownloadError::InvalidBounds(format!(
            "{error}, got [min_lat, max_lat, min_lon, max_lon] = {bounds:?}"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::mock_server::{MockResponse, mock_server};

    const BOUNDS: [f64; 4] = [
        19.28241180526043,
        19.67909634636101,
        -99.4141148418833,
        -98.53735764573854,
    ];

    /// A small, valid GeoTIFF: one 2x2 float raster with a tie point and pixel scale.
    fn geotiff_bytes() -> Vec<u8> {
        // Little-endian TIFF; the values that do not fit in an IFD entry follow the IFD.
        let mut tiff = Vec::new();
        let entries: u16 = 10;
        let ifd_offset: u32 = 8;
        let data_offset = ifd_offset + 2 + entries as u32 * 12 + 4;
        let scale_offset = data_offset + 16;
        let tiepoint_offset = scale_offset + 24;
        tiff.extend_from_slice(b"II*\0");
        tiff.extend_from_slice(&ifd_offset.to_le_bytes());
        tiff.extend_from_slice(&entries.to_le_bytes());
        let mut entry = |tag: u16, typ: u16, count: u32, value: u32| {
            tiff.extend_from_slice(&tag.to_le_bytes());
            tiff.extend_from_slice(&typ.to_le_bytes());
            tiff.extend_from_slice(&count.to_le_bytes());
            tiff.extend_from_slice(&value.to_le_bytes());
        };
        entry(256, 3, 1, 2); // ImageWidth
        entry(257, 3, 1, 2); // ImageLength
        entry(258, 3, 1, 32); // BitsPerSample
        entry(262, 3, 1, 1); // PhotometricInterpretation: BlackIsZero
        entry(273, 4, 1, data_offset); // StripOffsets
        entry(278, 3, 1, 2); // RowsPerStrip
        entry(279, 4, 1, 16); // StripByteCounts
        entry(339, 3, 1, 3); // SampleFormat: IEEE float
        entry(33550, 12, 3, scale_offset); // ModelPixelScale
        entry(33922, 12, 6, tiepoint_offset); // ModelTiepoint
        tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        for height in [1.0f32, 2.0, 3.0, 4.0] {
            tiff.extend_from_slice(&height.to_le_bytes());
        }
        for value in [0.5f64, 0.5, 0.0] {
            tiff.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0f64, 0.0, 0.0, -99.0, 19.5, 0.0] {
            tiff.extend_from_slice(&value.to_le_bytes());
        }
        tiff
    }

    /// A downloader with a fake API key, caching into a fresh directory named `name`.
    fn test_builder(name: &str, base: &str) -> OpenTopographyDemDownloaderBuilder {
        let cache_dir = env::temp_dir()
            .join("psi_insar_rs_opentopography_dem_download")
            .join(name);
        let _ = fs::remove_dir_all(&cache_dir);
        OpenTopographyDemDownloader::builder()
            .api_key("secret-key")
            .cache_dir(cache_dir)
            .max_retries(2)
            .api_url(base)
    }

    fn tiff_response() -> MockResponse {
        MockResponse::bytes("application/octet-stream", geotiff_bytes())
    }

    fn xml_error(status: u16, message: &str) -> MockResponse {
        let mut resp = MockResponse::json_body(
            status,
            format!(r#"<?xml version="1.0"?><error>Error: {message}</error>"#),
        );
        resp.headers = vec![("Content-Type", "application/xml".into())];
        resp
    }

    #[test]
    fn downloads_opens_and_caches_dem() {
        let (base, requests) = mock_server(vec![tiff_response()]);
        let downloader = test_builder("download", &base).build();
        let dem = downloader
            .fetch_dem(BOUNDS, CopernicusDemType::Cop30)
            .unwrap();
        assert_eq!((dem.rows(), dem.cols()), (2, 2));
        let path = downloader.cached_path(BOUNDS, CopernicusDemType::Cop30);
        assert_eq!(
            path,
            downloader.cache_dir().join(
                "COP30/S19.28241180526043_N19.67909634636101_W-99.4141148418833_E-98.53735764573854.tif"
            )
        );
        assert!(path.is_file());

        let request = &requests.all()[0];
        assert_eq!(request.path(), "/globaldem");
        let query: Vec<(String, String)> =
            url::form_urlencoded::parse(request.target.split_once('?').unwrap().1.as_bytes())
                .into_owned()
                .collect();
        let get = |key: &str| query.iter().find(|(k, _)| k == key).unwrap().1.clone();
        assert_eq!(get("demtype"), "COP30");
        assert_eq!(get("south"), "19.28241180526043");
        assert_eq!(get("east"), "-98.53735764573854");
        assert_eq!(get("outputFormat"), "GTiff");
        assert_eq!(get("API_Key"), "secret-key");

        // Cached: the server has no responses left.
        assert_eq!(
            downloader
                .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
                .unwrap(),
            path
        );
        // Other bounds or DEM types are not served from the cache.
        assert_ne!(
            downloader.cached_path(BOUNDS, CopernicusDemType::Cop90),
            path
        );
    }

    #[test]
    fn cached_dem_needs_no_api_key() {
        let (base, requests) = mock_server(vec![tiff_response()]);
        let builder = test_builder("no_api_key", &base);
        let cache_dir = builder.cache_dir.clone();
        builder
            .build()
            .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
            .unwrap();

        let mut builder = OpenTopographyDemDownloader::builder()
            .cache_dir(cache_dir)
            .api_url("http://127.0.0.1:1");
        builder.api_key = None;
        let downloader = builder.build();
        downloader
            .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
            .unwrap();
        let result = downloader.fetch_dem_file(BOUNDS, CopernicusDemType::Cop90);
        assert!(matches!(result, Err(DemDownloadError::MissingApiKey(_))));
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn disabled_cache_downloads_again_and_replaces_file() {
        let (base, requests) = mock_server(vec![tiff_response(), tiff_response()]);
        let downloader = test_builder("uncached", &base).cache(false).build();
        let path = downloader
            .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
            .unwrap();
        fs::write(&path, "stale").unwrap();
        downloader
            .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), geotiff_bytes());
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn api_key_is_redacted_from_errors() {
        let (base, _) = mock_server(vec![xml_error(
            401,
            "Not a valid format API Key: secret-key. Please register for an API key",
        )]);
        let downloader = test_builder("bad_key", &base).build();
        let err = downloader
            .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
            .unwrap_err();
        assert!(matches!(err, DemDownloadError::Authentication(_)));
        let message = err.to_string();
        assert!(!message.contains("secret-key"), "{message}");
        assert!(message.contains("<API key>"), "{message}");
    }

    #[test]
    fn no_data_and_bad_requests_are_not_retried() {
        let (base, requests) = mock_server(vec![
            MockResponse {
                status: 204,
                headers: Vec::new(),
                body: Vec::new(),
            },
            xml_error(400, "The requested area is too large"),
        ]);
        let downloader = test_builder("no_data", &base).build();
        let result = downloader.fetch_dem_file(BOUNDS, CopernicusDemType::Cop30);
        assert!(matches!(result, Err(DemDownloadError::NoData(_))));
        let result = downloader.fetch_dem_file(BOUNDS, CopernicusDemType::Cop30);
        assert!(matches!(
            result,
            Err(DemDownloadError::Http { status: 400, .. })
        ));
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn retries_server_errors() {
        let (base, requests) = mock_server(vec![
            xml_error(500, "Internal error"),
            xml_error(503, "Unavailable"),
            tiff_response(),
        ]);
        let downloader = test_builder("transient", &base).build();
        downloader
            .fetch_dem_file(BOUNDS, CopernicusDemType::Cop30)
            .unwrap();
        assert_eq!(requests.len(), 3);
    }

    #[test]
    fn non_tiff_body_is_rejected_without_output() {
        let (base, _) = mock_server(vec![xml_error(200, "Unexpected")]);
        let downloader = test_builder("non_tiff", &base).build();
        let result = downloader.fetch_dem_file(BOUNDS, CopernicusDemType::Cop30);
        assert!(matches!(result, Err(DemDownloadError::InvalidResponse(_))));
        let dir = downloader.cache_dir().join("COP30");
        assert_eq!(fs::read_dir(dir).unwrap().count(), 0);
    }

    #[test]
    fn invalid_bounds_are_rejected() {
        let downloader = OpenTopographyDemDownloader::builder().build();
        for bounds in [
            [19.7, 19.3, -99.4, -98.5],
            [19.3, 19.7, -98.5, -99.4],
            [19.3, 91.0, -99.4, -98.5],
            [19.3, 19.7, -181.0, -98.5],
            [f64::NAN, 19.7, -99.4, -98.5],
        ] {
            let result = downloader.fetch_dem_file(bounds, CopernicusDemType::Cop30);
            assert!(
                matches!(result, Err(DemDownloadError::InvalidBounds(_))),
                "{bounds:?}"
            );
        }
    }

    // Needs OPENTOPOGRAPHY_API_KEY. Downloads a COP90 raster with caching disabled, replacing
    // the copy in default_cache_dir().
    #[test]
    #[ignore = "Needs to download external data"]
    fn simple_test() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Info)
            .try_init();
        let downloader = OpenTopographyDemDownloader::builder().cache(false).build();
        let dem = downloader
            .fetch_dem(BOUNDS, CopernicusDemType::Cop90)
            .unwrap();
        println!("DEM size: {} x {}", dem.rows(), dem.cols());
        assert!(!dem.is_empty());
    }
}
