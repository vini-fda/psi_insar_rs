//! ASF Burst Download
//!
//! Downloads Sentinel-1 bursts from the ASF burst extractor service.
//! It uses NASA Earthdata Login (EDL) credentials for authentication.
//!
//! According to the ASF API specification [\[1\]], the extractor responds to
//! `GET /{granule}/{subswath}/{pol}/{burst}.zip` with:
//!
//! * `307` and a `Location` to the extracted product when it is ready,
//! * `202` when an extraction job was just triggered (the client must retry later),
//! * `403` when the `asf-urs` cookie is invalid or expired,
//! * `404` when the path parameters are invalid,
//! * `500` on unhandled server errors.
//!
//! [\[1\]] also says that requests without credentials are redirected to Earthdata Login, and
//! that after a `202` the client should "wait briefly" before requesting the burst again.
//!
//! # Caching
//!
//! [`AsfBurstDownloader::fetch_burst`] stores each burst as a SAFE directory at
//! `{cache_dir}/{granule}/{subswath}/{pol}/{burst_index}.SAFE`, mirroring the API path. The
//! cache directory defaults to the per-user cache directory (see [`default_cache_dir`]), so it
//! does not depend on the working directory, and is set with
//! [`AsfBurstDownloaderBuilder::cache_dir`]. With caching enabled (the default), a burst
//! already in the cache is returned without network access, so Earthdata credentials are
//! only needed for bursts that are not cached yet. With
//! [`AsfBurstDownloaderBuilder::cache`]`(false)`, every request downloads the burst again
//! and replaces the cached copy.
//!
//! # Redirects and authentication
//!
//! Redirects are followed manually (the agent has `max_redirects(0)`), up to
//! `MAX_REDIRECTS` hops. Each `Location` is resolved against the current URL, and
//! session cookies are kept in the agent's cookie jar. Following them by hand lets every
//! intermediate status be inspected and controls which host receives the credentials.
//! NASA's guide for scripted EDL access [\[2\]] uses `curl -L -n -b cookies -c cookies`:
//! it follows redirects, keeps a cookie session, and takes the credentials from a `.netrc`
//! entry for `urs.earthdata.nasa.gov`. That way only the EDL host gets the credentials.
//! This module does the same: the `Authorization: Basic` header is only sent over https to
//! `urs.earthdata.nasa.gov`, never to ASF or S3.
//!
//! Without a session cookie, a burst request goes through the EDL OAuth flow. Neither [\[1\]]
//! nor [\[2\]] documents the individual hops; this chain was observed with curl in October 2026
//! and may change:
//!
//! ```text
//! sentinel1-burst.asf.alaska.edu/{burst}.zip            307 -> auth.asf.alaska.edu/loginservice/in/{burst url}
//! auth.asf.alaska.edu/loginservice/in/...               302 -> urs.earthdata.nasa.gov/oauth/authorize?...
//! urs.earthdata.nasa.gov/oauth/authorize (+ Basic auth) 302 -> auth.asf.alaska.edu/login?code=...
//! auth.asf.alaska.edu/login                             301 -> auth.asf.alaska.edu/loginservice/out
//! auth.asf.alaska.edu/loginservice/out (sets asf-urs)   302 -> sentinel1-burst.asf.alaska.edu/{burst}.zip
//! sentinel1-burst.asf.alaska.edu/{burst}.zip            202 (extracting) or 307 -> presigned S3 URL
//! S3 bucket                                             200 (the zip)
//! ```
//!
//! With a valid `asf-urs` cookie, only the last two steps happen.
//!
//! # Response handling
//!
//! Every download attempt ends at the first non-redirect response, which is handled as follows:
//!
//! | Final response                 | Meaning                                    | Action                                    |
//! |--------------------------------|--------------------------------------------|-------------------------------------------|
//! | `200` from ASF/S3              | The burst zip                              | Stream to disk, validate, extract         |
//! | `200` from EDL                 | EDL showed a page instead of redirecting   | Fail: authentication error                |
//! | `202`                          | Extraction job triggered or still running  | Poll every `poll_interval`, up to `extraction_timeout` |
//! | `401` from EDL                 | Wrong username/password                    | Fail: authentication error                |
//! | `403`                          | Invalid/expired cookie or presigned URL    | Clear cookies and log in again, once      |
//! | `404`                          | Invalid granule/subswath/pol/burst         | Fail immediately                          |
//! | `408`, `429`, `5xx`            | Transient server error                     | Retry with exponential backoff            |
//! | Network/timeout errors         | Transient transport error                  | Retry with exponential backoff            |
//! | Anything else                  | Unexpected                                 | Fail                                      |
//!
//! Retries are bounded by [`RetryPolicy`]. A `200` body is also retried if it is truncated
//! (fewer bytes than `Content-Length`) or cannot be extracted. It fails without a retry if it
//! is not a zip, or if the extracted product lacks `manifest.safe` or a measurement TIFF.
//!
//! The zip is written to `<output>.zip.part` and extracted to `<output>.part`. That directory
//! is only renamed to the output path once validated, so failures never leave an empty or
//! partial SAFE directory behind.
//!
//! The `202`, `403`, `404` and `5xx` meanings come from [\[1\]]. The rest of the table is not
//! from either source:
//!
//! * Transient `500`s from `auth.asf.alaska.edu` were observed with curl (October 2026).
//! * The EDL rows (`200` page, `401`) are assumptions about EDL's behavior and were not
//!   tested with bad credentials or unauthorized applications.
//! * `408`, `429` and expired presigned S3 URLs answering `403` follow standard HTTP and S3
//!   semantics.
//! * Retry counts, backoff and the polling interval are choices made here.
//!
//! # Sources
//!
//! 1. ASF Sentinel-1 burst documentation, API specification section:
//!    <https://sentinel1-burst-documentation.asf.alaska.edu/#api-specification>
//! 2. NASA Earthdata Login documentation, data access with curl and wget:
//!    <https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget>
//!
//! [\[1\]]: https://sentinel1-burst-documentation.asf.alaska.edu/#api-specification
//! [\[2\]]: https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget

use base64::prelude::{BASE64_STANDARD, Engine as _};
use log::{info, warn};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use ureq::Body;
use ureq::http::{Response, header};
use url::Url;

use crate::datasets::http::{TempPath, USER_AGENT, body_snippet, is_transient, path_with_suffix};
use crate::granule_id::IWSwath;

/// Base URL of the ASF burst extractor API.
pub const ASF_BURST_API_URL: &str = "https://sentinel1-burst.asf.alaska.edu";
/// Environment variables holding the Earthdata Login credentials.
pub const EARTHDATA_USERNAME_VAR: &str = "EARTHDATA_USERNAME";
pub const EARTHDATA_PASSWORD_VAR: &str = "EARTHDATA_PASSWORD";

/// Earthdata Login host, the only host that receives the credentials.
const URS_HOST: &str = "urs.earthdata.nasa.gov";
/// The EDL login flow takes ~6 hops, plus one to the extracted product.
const MAX_REDIRECTS: usize = 20;
const ZIP_MAGIC: &[u8; 4] = b"PK\x03\x04";

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
    /// The burst does not exist, or the URL parameters are invalid (HTTP 404)
    NotFound(String),
    /// The server kept answering `202 Accepted` for longer than the configured timeout
    ExtractionTimeout(String),
    /// The downloaded file is not a valid burst SAFE archive
    InvalidArchive(String),
    /// The output path exists, is not empty and does not hold a complete SAFE product
    OutputExists(PathBuf),
    /// A download is needed but no Earthdata Login credentials were configured
    MissingCredentials(String),
    /// The burst request has invalid parameters (e.g. an empty or malformed granule name)
    InvalidBurst(String),
    /// Download failed
    DownloadFailed(String),
    /// URL parsing error
    UrlParseError(url::ParseError),
    /// HTML parsing error
    HtmlParsingError(String),
    /// Zip extraction error
    ZipError(zip::result::ZipError),
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

impl From<zip::result::ZipError> for AsfDownloadError {
    fn from(err: zip::result::ZipError) -> Self {
        AsfDownloadError::ZipError(err)
    }
}

/// The default cache directory: `asf_bursts` under [`cache_root`](crate::datasets::cache_root),
/// i.e. `{user cache dir}/psi_insar_rs/asf_bursts`.
pub fn default_cache_dir() -> PathBuf {
    crate::datasets::cache_root().join("asf_bursts")
}

/// Polarization of a single burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarization {
    HH,
    HV,
    VH,
    VV,
}

impl std::fmt::Display for Polarization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Polarization::HH => "HH",
            Polarization::HV => "HV",
            Polarization::VH => "VH",
            Polarization::VV => "VV",
        };
        f.write_str(name)
    }
}

/// A single burst of a Sentinel-1 IW SLC product, as addressed by the ASF burst extractor:
/// `GET /{granule}/{subswath}/{polarization}/{burst_index}.zip`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurstRequest {
    /// SLC product name, without the `.SAFE` extension.
    pub granule: String,
    pub subswath: IWSwath,
    pub polarization: Polarization,
    /// Index of the burst within the subswath (0-based).
    pub burst_index: u32,
}

impl BurstRequest {
    /// A `.SAFE` suffix on `granule` is ignored.
    pub fn new(
        granule: impl Into<String>,
        subswath: IWSwath,
        polarization: Polarization,
        burst_index: u32,
    ) -> Self {
        let mut granule = granule.into();
        if let Some(stripped) = granule.strip_suffix(".SAFE") {
            granule.truncate(stripped.len());
        }
        BurstRequest {
            granule,
            subswath,
            polarization,
            burst_index,
        }
    }

    /// The granule name is used both in the URL and as a cache directory name, so it must be
    /// a plain product name (ASCII letters, digits and underscores).
    fn validate(&self) -> Result<(), AsfDownloadError> {
        let valid = !self.granule.is_empty()
            && self
                .granule
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_');
        if valid {
            Ok(())
        } else {
            Err(AsfDownloadError::InvalidBurst(format!(
                "Invalid granule name {:?}",
                self.granule
            )))
        }
    }

    /// Path of the burst relative to the API root and to the cache directory, without extension.
    fn relative_path(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.granule, self.subswath, self.polarization, self.burst_index
        )
    }
}

/// Builder for [`AsfBurstDownloader`].
///
/// ```no_run
/// use psi_insar_rs::datasets::asf::burst_download::{AsfBurstDownloader, BurstRequest, Polarization};
/// use psi_insar_rs::granule_id::IWSwath;
///
/// // Caches in `default_cache_dir()` unless `.cache_dir(...)` is given.
/// let downloader = AsfBurstDownloader::builder().build();
/// let burst = BurstRequest::new(
///     "S1A_IW_SLC__1SSV_20151022T122539_20151022T122606_008265_00BA51_5A48",
///     IWSwath::IW3,
///     Polarization::VV,
///     2,
/// );
/// let safe_dir = downloader.fetch_burst(&burst).unwrap();
/// ```
pub struct AsfBurstDownloaderBuilder {
    credentials: Option<(String, String)>,
    cache_dir: PathBuf,
    cache_enabled: bool,
    retry_policy: RetryPolicy,
    base_url: String,
}

impl Default for AsfBurstDownloaderBuilder {
    fn default() -> Self {
        let credentials = match (
            env::var(EARTHDATA_USERNAME_VAR),
            env::var(EARTHDATA_PASSWORD_VAR),
        ) {
            (Ok(username), Ok(password)) => Some((username, password)),
            _ => None,
        };
        AsfBurstDownloaderBuilder {
            credentials,
            cache_dir: default_cache_dir(),
            cache_enabled: true,
            retry_policy: RetryPolicy::default(),
            base_url: ASF_BURST_API_URL.to_string(),
        }
    }
}

impl AsfBurstDownloaderBuilder {
    /// Earthdata Login credentials. Defaults to the `EARTHDATA_USERNAME` and
    /// `EARTHDATA_PASSWORD` environment variables, if both are set.
    ///
    /// Credentials are only required when a burst has to be downloaded, so cached bursts can
    /// be used without them.
    pub fn credentials(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.credentials = Some((username.into(), password.into()));
        self
    }

    /// Directory where bursts are stored, as `{cache_dir}/{granule}/{subswath}/{pol}/{burst_index}.SAFE`.
    /// Defaults to [`default_cache_dir`]. A relative path is resolved against the working
    /// directory at the time of each download.
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Whether bursts already in the cache directory are reused (the default). When disabled,
    /// every request downloads the burst again and replaces the copy in the cache directory.
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Point the downloader at another server, e.g. a mock server in tests.
    #[cfg(test)]
    fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn build(self) -> AsfBurstDownloader {
        let auth_header = self.credentials.map(|(username, password)| {
            format!(
                "Basic {}",
                BASE64_STANDARD.encode(format!("{username}:{password}"))
            )
        });
        // Redirects are followed manually, so that the credentials are only sent to
        // Earthdata Login and every intermediate status code can be inspected.
        let config = ureq::Agent::config_builder()
            .max_redirects(0)
            .http_status_as_error(false)
            .user_agent(USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(120)))
            .timeout_recv_body(Some(Duration::from_secs(30 * 60)))
            .build();
        AsfBurstDownloader {
            agent: ureq::Agent::new_with_config(config),
            auth_header,
            retry_policy: self.retry_policy,
            cache_dir: self.cache_dir,
            cache_enabled: self.cache_enabled,
            base_url: self.base_url,
        }
    }
}

/// Controls how long and how often [`AsfBurstDownloader`] retries a download.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Number of retries after transient failures (network errors, 5xx, truncated or corrupt files).
    pub max_retries: u32,
    /// Backoff before the first retry; doubled after every transient failure.
    pub initial_backoff: Duration,
    /// Upper bound for the exponential backoff.
    pub max_backoff: Duration,
    /// Interval between polls while the server is extracting the burst (HTTP 202).
    pub poll_interval: Duration,
    /// Give up if the burst is still being extracted after this long.
    pub extraction_timeout: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_retries: 5,
            initial_backoff: Duration::from_secs(2),
            max_backoff: Duration::from_secs(60),
            poll_interval: Duration::from_secs(5),
            extraction_timeout: Duration::from_secs(10 * 60),
        }
    }
}

/// Why a single download attempt did not produce the burst.
enum AttemptError {
    /// The server accepted the extraction job (HTTP 202); poll again later.
    Pending,
    /// The session cookie was rejected (HTTP 403); logging in again may help.
    Forbidden(String),
    /// A failure that may go away on retry.
    Transient(AsfDownloadError),
    /// A failure that will not go away on retry.
    Fatal(AsfDownloadError),
}

impl From<io::Error> for AttemptError {
    fn from(err: io::Error) -> Self {
        AttemptError::Fatal(err.into())
    }
}

impl From<ureq::Error> for AttemptError {
    fn from(err: ureq::Error) -> Self {
        if is_transient(&err) {
            AttemptError::Transient(err.into())
        } else {
            AttemptError::Fatal(err.into())
        }
    }
}

/// Downloads Sentinel-1 bursts from ASF into an on-disk cache. Create it with
/// [`AsfBurstDownloader::builder`].
pub struct AsfBurstDownloader {
    agent: ureq::Agent,
    /// `None` when no credentials were configured; only an error once a download is needed.
    auth_header: Option<String>,
    retry_policy: RetryPolicy,
    cache_dir: PathBuf,
    cache_enabled: bool,
    base_url: String,
}

impl AsfBurstDownloader {
    pub fn builder() -> AsfBurstDownloaderBuilder {
        AsfBurstDownloaderBuilder::default()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Where `burst` is stored: `{cache_dir}/{granule}/{subswath}/{pol}/{burst_index}.SAFE`.
    pub fn cached_path(&self, burst: &BurstRequest) -> PathBuf {
        self.cache_dir
            .join(format!("{}.SAFE", burst.relative_path()))
    }

    /// The ASF burst extractor URL of `burst`.
    pub fn burst_url(&self, burst: &BurstRequest) -> String {
        format!("{}/{}.zip", self.base_url, burst.relative_path())
    }

    /// Return the SAFE directory of `burst`, downloading it unless it is already cached.
    ///
    /// When caching is enabled and [`cached_path`](Self::cached_path) already holds a complete
    /// SAFE product, it is returned without any network access or credentials. Otherwise the
    /// burst is downloaded there with [`download_file`](Self::download_file).
    pub fn fetch_burst(&self, burst: &BurstRequest) -> Result<PathBuf, AsfDownloadError> {
        burst.validate()?;
        let path = self.cached_path(burst);
        self.download_file(&self.burst_url(burst), &path)?;
        Ok(path)
    }

    /// [`fetch_burst`](Self::fetch_burst) for several bursts, in order. Stops at the first error.
    pub fn fetch_bursts<'a>(
        &self,
        bursts: impl IntoIterator<Item = &'a BurstRequest>,
    ) -> Result<Vec<PathBuf>, AsfDownloadError> {
        bursts
            .into_iter()
            .map(|burst| self.fetch_burst(burst))
            .collect()
    }

    /// Download a single burst zip from ASF and extract it as a SAFE directory.
    ///
    /// Bursts that are not cached yet are extracted on demand by ASF (HTTP 202); this
    /// function polls until the burst is ready, retries transient failures with
    /// exponential backoff and logs in again if the session cookie expires.
    ///
    /// The archive is extracted into a temporary sibling directory that is only moved to
    /// `output_path` once it holds a complete SAFE product, so `output_path` is never left
    /// empty or half-written.
    ///
    /// If `output_path` already holds a complete SAFE product, the download is skipped when
    /// caching is enabled; otherwise the product is downloaded again and replaces it. An empty
    /// directory is replaced, and any other existing path is an
    /// [`OutputExists`](AsfDownloadError::OutputExists) error.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL of the file to download
    /// * `output_path` - The SAFE directory where the burst will be extracted
    ///
    /// # Returns
    ///
    /// Result indicating success or the reason for failure
    pub fn download_file(&self, url: &str, output_path: &Path) -> Result<(), AsfDownloadError> {
        if output_path.exists() {
            if is_complete_safe(output_path) {
                if self.cache_enabled {
                    info!("Using cached {}", output_path.display());
                    return Ok(());
                }
                // Caching is disabled: replaced once the new download is complete.
            } else if is_empty_dir(output_path) {
                // An empty directory is what failed downloads used to leave behind.
                fs::remove_dir(output_path)?;
            } else {
                return Err(AsfDownloadError::OutputExists(output_path.to_path_buf()));
            }
        }
        if self.auth_header.is_none() {
            return Err(AsfDownloadError::MissingCredentials(format!(
                "Downloading {url} requires Earthdata Login credentials: set \
                 {EARTHDATA_USERNAME_VAR} and {EARTHDATA_PASSWORD_VAR}, or call \
                 AsfBurstDownloaderBuilder::credentials"
            )));
        }
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let policy = &self.retry_policy;
        let started = Instant::now();
        let mut backoff = policy.initial_backoff;
        let mut retries = 0;
        let mut logged_in_again = false;
        loop {
            match self.try_download(url, output_path) {
                Ok(()) => {
                    info!("Downloaded {url} to {}", output_path.display());
                    return Ok(());
                }
                Err(AttemptError::Pending) => {
                    let elapsed = started.elapsed();
                    if elapsed >= policy.extraction_timeout {
                        return Err(AsfDownloadError::ExtractionTimeout(format!(
                            "{url} was still being extracted after {elapsed:?}"
                        )));
                    }
                    info!(
                        "ASF is extracting {url} (HTTP 202), polling again in {:?}",
                        policy.poll_interval
                    );
                    thread::sleep(policy.poll_interval);
                }
                Err(AttemptError::Forbidden(msg)) if !logged_in_again => {
                    warn!("ASF rejected the session ({msg}), logging in again");
                    self.agent.cookie_jar_lock().clear();
                    logged_in_again = true;
                }
                Err(AttemptError::Forbidden(msg)) => {
                    return Err(AsfDownloadError::Forbidden(msg));
                }
                Err(AttemptError::Transient(err)) if retries < policy.max_retries => {
                    retries += 1;
                    warn!(
                        "Download of {url} failed ({err:?}), retry {retries}/{} in {backoff:?}",
                        policy.max_retries
                    );
                    thread::sleep(backoff);
                    backoff = (backoff * 2).min(policy.max_backoff);
                }
                Err(AttemptError::Transient(err) | AttemptError::Fatal(err)) => return Err(err),
            }
        }
    }

    /// A single attempt: request the burst, then download, extract and validate it.
    fn try_download(&self, url: &str, output_path: &Path) -> Result<(), AttemptError> {
        let (final_url, mut resp) = self.get_following_redirects(url)?;
        let status = resp.status().as_u16();
        let is_urs = is_urs_url(&final_url);
        match status {
            200 if is_urs => {
                // EDL showed a page instead of redirecting back, e.g. because the ASF
                // application is not authorized for this account or a EULA is pending.
                return Err(AttemptError::Fatal(AsfDownloadError::AuthenticationError(
                    format!(
                        "Earthdata Login did not redirect back to ASF ({final_url}). Log in at \
                         https://{URS_HOST} and check that the ASF applications are authorized"
                    ),
                )));
            }
            200 => {}
            202 => return Err(AttemptError::Pending),
            401 if is_urs => {
                return Err(AttemptError::Fatal(AsfDownloadError::AuthenticationError(
                    format!(
                        "Earthdata Login rejected the credentials: {}",
                        body_snippet(&mut resp)
                    ),
                )));
            }
            403 => {
                return Err(AttemptError::Forbidden(format!(
                    "HTTP 403 from {final_url}: {}",
                    body_snippet(&mut resp)
                )));
            }
            404 => {
                return Err(AttemptError::Fatal(AsfDownloadError::NotFound(format!(
                    "{url}: {}",
                    body_snippet(&mut resp)
                ))));
            }
            408 | 429 | 500..=599 => {
                return Err(AttemptError::Transient(AsfDownloadError::DownloadFailed(
                    format!(
                        "HTTP {status} from {final_url}: {}",
                        body_snippet(&mut resp)
                    ),
                )));
            }
            _ => {
                return Err(AttemptError::Fatal(AsfDownloadError::DownloadFailed(
                    format!(
                        "HTTP {status} from {final_url}: {}",
                        body_snippet(&mut resp)
                    ),
                )));
            }
        }

        // Stream the archive to disk instead of buffering it in memory.
        let zip_path = TempPath(path_with_suffix(output_path, ".zip.part"));
        let expected_len = resp.body().content_length();
        let mut file = File::create(&zip_path.0)?;
        let written = io::copy(&mut resp.into_body().into_reader(), &mut file).map_err(|err| {
            AttemptError::Transient(AsfDownloadError::DownloadFailed(format!(
                "Error while receiving {url}: {err}"
            )))
        })?;
        drop(file);
        if let Some(expected_len) = expected_len
            && written != expected_len
        {
            return Err(AttemptError::Transient(AsfDownloadError::DownloadFailed(
                format!("Truncated download of {url}: got {written} of {expected_len} bytes"),
            )));
        }
        check_zip_magic(&zip_path.0)?;

        let extract_dir = TempPath(path_with_suffix(output_path, ".part"));
        if extract_dir.0.exists() {
            fs::remove_dir_all(&extract_dir.0)?;
        }
        zip::ZipArchive::new(File::open(&zip_path.0)?)
            .and_then(|mut archive| {
                archive
                    .extract_unwrapped_root_dir(&extract_dir.0, zip::read::root_dir_common_filter)
            })
            .map_err(|err| AttemptError::Transient(err.into()))?;
        if !is_complete_safe(&extract_dir.0) {
            return Err(AttemptError::Fatal(AsfDownloadError::InvalidArchive(
                format!("{url} does not contain manifest.safe and a measurement TIFF"),
            )));
        }
        if output_path.exists() {
            // Only a complete SAFE product gets here (see `download_file`), when caching is
            // disabled.
            fs::remove_dir_all(output_path)?;
        }
        fs::rename(&extract_dir.0, output_path)?;
        Ok(())
    }

    /// GET `url`, following redirects and sending the credentials only to Earthdata Login.
    /// Returns the final URL and its (non-redirect) response.
    fn get_following_redirects(&self, url: &str) -> Result<(Url, Response<Body>), AttemptError> {
        let mut url = Url::parse(url).map_err(|err| AttemptError::Fatal(err.into()))?;
        for _ in 0..=MAX_REDIRECTS {
            let mut request = self.agent.get(url.as_str());
            if is_urs_url(&url)
                && let Some(auth_header) = &self.auth_header
            {
                request = request.header(header::AUTHORIZATION, auth_header);
            }
            let resp = request.call()?;
            if !resp.status().is_redirection() {
                return Ok((url, resp));
            }
            let location = resp
                .headers()
                .get(header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    AttemptError::Transient(AsfDownloadError::DownloadFailed(format!(
                        "HTTP {} from {url} without a Location header",
                        resp.status()
                    )))
                })?;
            url = url
                .join(location)
                .map_err(|err| AttemptError::Fatal(err.into()))?;
        }
        Err(AttemptError::Fatal(AsfDownloadError::AuthenticationError(
            format!(
                "More than {MAX_REDIRECTS} redirects, the Earthdata Login flow is probably looping (last URL: {url})"
            ),
        )))
    }
}

fn is_urs_url(url: &Url) -> bool {
    url.scheme() == "https" && url.host_str() == Some(URS_HOST)
}

fn is_empty_dir(path: &Path) -> bool {
    fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_none())
}

/// Whether `dir` holds a SAFE product with a manifest and at least one measurement TIFF.
fn is_complete_safe(dir: &Path) -> bool {
    let has_manifest = dir.join("manifest.safe").is_file();
    let has_measurement = fs::read_dir(dir.join("measurement")).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            let path = entry.path();
            path.extension().is_some_and(|ext| ext == "tiff")
                && entry.metadata().is_ok_and(|m| m.is_file() && m.len() > 0)
        })
    });
    has_manifest && has_measurement
}

fn check_zip_magic(path: &Path) -> Result<(), AttemptError> {
    let mut head = Vec::with_capacity(256);
    File::open(path)?.take(256).read_to_end(&mut head)?;
    if head.starts_with(ZIP_MAGIC) {
        return Ok(());
    }
    // Not retried as transient: the server answered successfully with something else.
    Err(AttemptError::Fatal(AsfDownloadError::InvalidArchive(
        format!(
            "Expected a zip archive, got: {:?}",
            String::from_utf8_lossy(&head)
        ),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::mock_server::{MockResponse, mock_server};
    use std::io::Write;

    /// A zip laid out like the ones served by ASF: a single top-level SAFE directory.
    fn safe_zip() -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, contents) in [
            ("S1A_TEST.SAFE/manifest.safe", "<manifest/>"),
            (
                "S1A_TEST.SAFE/annotation/s1a-iw3-slc-vv-test.xml",
                "<product/>",
            ),
            (
                "S1A_TEST.SAFE/measurement/s1a-iw3-slc-vv-test.tiff",
                "II*\0data",
            ),
        ] {
            writer.start_file(name, options).unwrap();
            writer.write_all(contents.as_bytes()).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn fast_policy() -> RetryPolicy {
        RetryPolicy {
            max_retries: 2,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(10),
            poll_interval: Duration::from_millis(10),
            extraction_timeout: Duration::from_secs(5),
        }
    }

    fn test_root() -> PathBuf {
        let dir = env::temp_dir().join("psi_insar_rs_asf_burst_download");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A downloader with fake credentials, caching into a fresh directory named `name`.
    fn test_builder(name: &str, policy: RetryPolicy) -> AsfBurstDownloaderBuilder {
        let cache_dir = test_root().join(format!("{name}_cache"));
        let _ = fs::remove_dir_all(&cache_dir);
        AsfBurstDownloader::builder()
            .credentials("user", "pass")
            .cache_dir(cache_dir)
            .retry_policy(policy)
    }

    fn test_downloader(policy: RetryPolicy) -> AsfBurstDownloader {
        test_builder("default", policy).build()
    }

    fn output_dir(name: &str) -> PathBuf {
        let output = test_root().join(format!("{name}.SAFE"));
        let _ = fs::remove_dir_all(&output);
        output
    }

    fn test_burst() -> BurstRequest {
        BurstRequest::new("G", IWSwath::IW3, Polarization::VV, 2)
    }

    #[test]
    fn polls_while_burst_is_being_extracted() {
        let (base, requests) = mock_server(vec![
            MockResponse::json(202, "Accepted"),
            MockResponse::json(202, "Accepted"),
            MockResponse::redirect("/bucket/burst.zip".into()),
            MockResponse::bytes("application/zip", safe_zip()),
        ]);
        let output = output_dir("polls");
        test_downloader(fast_policy())
            .download_file(&format!("{base}/G/IW3/VV/2.zip"), &output)
            .unwrap();
        assert!(is_complete_safe(&output));
        assert_eq!(
            requests.paths(),
            [
                "/G/IW3/VV/2.zip",
                "/G/IW3/VV/2.zip",
                "/G/IW3/VV/2.zip",
                "/bucket/burst.zip"
            ]
        );
    }

    #[test]
    fn extraction_timeout_leaves_no_output() {
        let responses = (0..1000)
            .map(|_| MockResponse::json(202, "Accepted"))
            .collect();
        let (base, _) = mock_server(responses);
        let output = output_dir("timeout");
        let policy = RetryPolicy {
            extraction_timeout: Duration::from_millis(100),
            ..fast_policy()
        };
        let result =
            test_downloader(policy).download_file(&format!("{base}/G/IW3/VV/2.zip"), &output);
        assert!(matches!(
            result,
            Err(AsfDownloadError::ExtractionTimeout(_))
        ));
        assert!(
            !output.exists(),
            "no empty output directory must be left behind"
        );
    }

    #[test]
    fn retries_transient_server_errors() {
        let (base, _) = mock_server(vec![
            MockResponse::json(500, "Internal server error"),
            MockResponse::json(503, "Service Unavailable"),
            MockResponse::bytes("application/zip", safe_zip()),
        ]);
        let output = output_dir("transient");
        test_downloader(fast_policy())
            .download_file(&format!("{base}/G/IW3/VV/2.zip"), &output)
            .unwrap();
        assert!(is_complete_safe(&output));
    }

    #[test]
    fn gives_up_after_max_retries() {
        let responses = (0..3)
            .map(|_| MockResponse::json(500, "Internal server error"))
            .collect();
        let (base, requests) = mock_server(responses);
        let output = output_dir("gives_up");
        let result = test_downloader(fast_policy())
            .download_file(&format!("{base}/G/IW3/VV/2.zip"), &output);
        assert!(matches!(result, Err(AsfDownloadError::DownloadFailed(_))));
        assert_eq!(requests.len(), 3);
        assert!(!output.exists());
    }

    #[test]
    fn not_found_is_not_retried() {
        let (base, requests) = mock_server(vec![MockResponse::json(404, "Not Found")]);
        let output = output_dir("not_found");
        let result = test_downloader(fast_policy())
            .download_file(&format!("{base}/G/IW9/VV/2.zip"), &output);
        assert!(matches!(result, Err(AsfDownloadError::NotFound(_))));
        assert_eq!(requests.len(), 1);
        assert!(!output.exists());
    }

    #[test]
    fn non_zip_body_is_rejected_without_output() {
        let (base, _) = mock_server(vec![MockResponse::json(200, "Accepted")]);
        let output = output_dir("non_zip");
        let result = test_downloader(fast_policy())
            .download_file(&format!("{base}/G/IW3/VV/2.zip"), &output);
        assert!(matches!(result, Err(AsfDownloadError::InvalidArchive(_))));
        assert!(!output.exists());
    }

    #[test]
    fn forbidden_logs_in_again_once() {
        let (base, requests) = mock_server(vec![
            MockResponse::json(403, "Invalid token"),
            MockResponse::bytes("application/zip", safe_zip()),
        ]);
        let output = output_dir("forbidden");
        test_downloader(fast_policy())
            .download_file(&format!("{base}/G/IW3/VV/2.zip"), &output)
            .unwrap();
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn replaces_empty_output_and_skips_complete_output() {
        let (base, requests) =
            mock_server(vec![MockResponse::bytes("application/zip", safe_zip())]);
        let output = output_dir("existing");
        fs::create_dir(&output).unwrap();
        let downloader = test_downloader(fast_policy());
        let url = format!("{base}/G/IW3/VV/2.zip");
        downloader.download_file(&url, &output).unwrap();
        // The second call must not hit the server (which only serves one response).
        downloader.download_file(&url, &output).unwrap();
        assert!(is_complete_safe(&output));
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn credentials_are_only_sent_to_earthdata_login() {
        let urs = Url::parse("https://urs.earthdata.nasa.gov/oauth/authorize?x=1").unwrap();
        let urs_http = Url::parse("http://urs.earthdata.nasa.gov/oauth/authorize").unwrap();
        let asf = Url::parse("https://sentinel1-burst.asf.alaska.edu/G/IW3/VV/2.zip").unwrap();
        let lookalike = Url::parse("https://urs.earthdata.nasa.gov.evil.com/").unwrap();
        assert!(is_urs_url(&urs));
        assert!(!is_urs_url(&urs_http));
        assert!(!is_urs_url(&asf));
        assert!(!is_urs_url(&lookalike));
    }

    #[test]
    fn fetch_burst_uses_cache_layout_and_reuses_cached_bursts() {
        let (base, requests) =
            mock_server(vec![MockResponse::bytes("application/zip", safe_zip())]);
        let downloader = test_builder("fetch_cached", fast_policy())
            .base_url(base)
            .build();
        let burst = test_burst();
        let path = downloader.fetch_burst(&burst).unwrap();
        assert_eq!(path, downloader.cache_dir().join("G/IW3/VV/2.SAFE"));
        assert!(is_complete_safe(&path));
        // The second call must not hit the server (which only serves one response).
        assert_eq!(downloader.fetch_burst(&burst).unwrap(), path);
        assert_eq!(requests.paths(), ["/G/IW3/VV/2.zip"]);
    }

    #[test]
    fn disabled_cache_downloads_again_and_replaces_cached_burst() {
        let (base, requests) = mock_server(vec![
            MockResponse::bytes("application/zip", safe_zip()),
            MockResponse::bytes("application/zip", safe_zip()),
        ]);
        let downloader = test_builder("fetch_uncached", fast_policy())
            .base_url(base)
            .cache(false)
            .build();
        let burst = test_burst();
        let path = downloader.fetch_burst(&burst).unwrap();
        fs::write(path.join("stale-marker"), "").unwrap();
        assert_eq!(downloader.fetch_burst(&burst).unwrap(), path);
        assert!(is_complete_safe(&path));
        assert!(
            !path.join("stale-marker").exists(),
            "cached copy must be replaced"
        );
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn cached_bursts_need_no_credentials() {
        let (base, requests) =
            mock_server(vec![MockResponse::bytes("application/zip", safe_zip())]);
        let builder = test_builder("no_credentials", fast_policy()).base_url(base);
        let cache_dir = builder.cache_dir.clone();
        let burst = test_burst();
        builder.build().fetch_burst(&burst).unwrap();

        let mut builder = AsfBurstDownloader::builder()
            .cache_dir(cache_dir)
            .base_url("http://127.0.0.1:1");
        builder.credentials = None;
        let downloader = builder.build();
        assert!(is_complete_safe(&downloader.fetch_burst(&burst).unwrap()));

        let uncached = BurstRequest::new("G", IWSwath::IW3, Polarization::VV, 3);
        let result = downloader.fetch_burst(&uncached);
        assert!(matches!(
            result,
            Err(AsfDownloadError::MissingCredentials(_))
        ));
        assert!(!downloader.cached_path(&uncached).exists());
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn burst_request_validation_and_urls() {
        let downloader = AsfBurstDownloader::builder().cache_dir("cache").build();
        let burst = BurstRequest::new("S1A_IW_SLC__1SSV_X.SAFE", IWSwath::IW1, Polarization::VH, 7);
        assert_eq!(burst.granule, "S1A_IW_SLC__1SSV_X");
        assert_eq!(
            downloader.burst_url(&burst),
            "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_X/IW1/VH/7.zip"
        );
        assert_eq!(
            downloader.cached_path(&burst),
            Path::new("cache/S1A_IW_SLC__1SSV_X/IW1/VH/7.SAFE")
        );
        for granule in ["", "../escape", "a/b", "x y"] {
            let burst = BurstRequest::new(granule, IWSwath::IW1, Polarization::VV, 0);
            assert!(matches!(
                downloader.fetch_burst(&burst),
                Err(AsfDownloadError::InvalidBurst(_))
            ));
        }
    }

    /// Granule and burst index of the bursts used by `rerun_tests::differential_phase_plot`
    /// (IW3, VV).
    const TEST_BURSTS: [(&str, u32); 7] = [
        (
            "S1A_IW_SLC__1SSV_20150916T122538_20150916T122603_007740_00AC19_8302",
            2,
        ),
        (
            "S1A_IW_SLC__1SSV_20150928T122539_20150928T122606_007915_00B0D8_5407",
            2,
        ),
        (
            "S1A_IW_SLC__1SSV_20151010T122539_20151010T122603_008090_00B578_7501",
            2,
        ),
        (
            "S1A_IW_SLC__1SSV_20151022T122539_20151022T122606_008265_00BA51_5A48",
            2,
        ),
        (
            "S1A_IW_SLC__1SSV_20151103T122539_20151103T122603_008440_00BEE0_AE93",
            2,
        ),
        (
            "S1A_IW_SLC__1SSV_20151115T122533_20151115T122600_008615_00C3B4_8956",
            4,
        ),
        (
            "S1A_IW_SLC__1SSV_20151127T122533_20151127T122557_008790_00C894_14CF",
            4,
        ),
    ];

    // This test requires valid EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables
    // to be set. It downloads every burst in TEST_BURSTS (~140 MB each) with caching disabled,
    // replacing the copies in default_cache_dir(), and fails if any burst could not be fetched.
    // It is ignored by default to prevent unintended network access and file writes during
    // automated tests.
    #[test]
    #[ignore]
    fn test_real_download_with_env_auth() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Info)
            .try_init();

        let downloader = AsfBurstDownloader::builder().cache(false).build();
        let mut failures = Vec::new();
        for (granule, burst_index) in TEST_BURSTS {
            let burst = BurstRequest::new(granule, IWSwath::IW3, Polarization::VV, burst_index);
            println!(
                "Downloading {} -> {}",
                downloader.burst_url(&burst),
                downloader.cached_path(&burst).display()
            );
            match downloader.fetch_burst(&burst) {
                Ok(path) => assert!(
                    is_complete_safe(&path),
                    "{} is not a complete SAFE product",
                    path.display()
                ),
                Err(err) => {
                    eprintln!("Download of {burst:?} failed: {err:?}");
                    failures.push((burst, err));
                }
            }
        }
        assert!(failures.is_empty(), "Failed downloads: {failures:#?}");
    }
}
