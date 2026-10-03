//! ASF Burst Search
//!
//! Finds Sentinel-1 SLC bursts covering a point or an area with the ASF Search API [\[1\]], so
//! that they can be downloaded with [`AsfBurstDownloader`](super::burst_download::AsfBurstDownloader).
//!
//! ```no_run
//! use chrono::{TimeZone, Utc};
//! use psi_insar_rs::datasets::asf::burst_download::Polarization;
//! use psi_insar_rs::datasets::asf::burst_search::{AsfBurstSearch, BurstSearchQuery, SearchArea};
//! use psi_insar_rs::granule_id::IWSwath;
//!
//! let start = Utc.with_ymd_and_hms(2016, 4, 8, 0, 0, 0).unwrap();
//! let end = Utc.with_ymd_and_hms(2016, 4, 21, 0, 0, 0).unwrap();
//! let epicenter = SearchArea::Point {
//!     lat: 32.7906,
//!     lon: 130.7543,
//! };
//! let query = BurstSearchQuery::new(epicenter, start, end, Polarization::VV).subswath(IWSwath::IW1);
//! let bursts = AsfBurstSearch::builder().build().search(&query).unwrap();
//! let requests: Vec<_> = bursts.iter().map(|burst| burst.request()).collect();
//! ```
//!
//! # Requests
//!
//! A single public `GET` request, without credentials:
//!
//! ```text
//! {search url}?platform=SENTINEL-1&processingLevel=BURST&polarization={pol}
//!     &intersectsWith={area}&start={start}&end={end}&output=geojson
//! ```
//!
//! where `{area}` is the WKT of the [`SearchArea`]: `POINT({lon} {lat})` or a `POLYGON` with
//! the corners of a bounding box. The keywords come from [\[1\]]. The response is a GeoJSON `FeatureCollection` with one
//! feature per burst (an empty `features` array when nothing matches), and errors are `400`
//! responses with an `{"error": ...}` body. Each feature's `properties.burst` holds the burst
//! ID, the subswath and the burst index, and `properties.url` is the burst extractor URL,
//! `https://sentinel1-burst.asf.alaska.edu/{granule}/{subswath}/{pol}/{burst_index}.tiff`,
//! which is the only field naming the parent SLC granule. These response details were observed
//! with curl in October 2026.
//!
//! The search also ignores the `beamSwath` keyword for bursts (it returns no results), so
//! [`BurstSearchQuery::subswath`] is applied to the results instead of being sent.
//!
//! Transport errors, `429` and `5xx` responses are retried with exponential backoff.
//!
//! # Caching
//!
//! Each search response is stored as `{cache_dir}/{pol}_{area}_{start}_{end}.geojson`, where
//! `{area}` is `{lat}_{lon}` for a point and `{min lat}_{min lon}_{max lat}_{max lon}` for a
//! bounding box, i.e. named after the query sent to ASF (the subswath filter is applied afterwards, so queries
//! that only differ in subswath share a file). With caching enabled (the default), a query
//! whose response is cached is answered without network access, which lets the bursts found
//! be downloaded and reused offline by
//! [`AsfBurstDownloader`](super::burst_download::AsfBurstDownloader). The cache directory
//! defaults to [`default_cache_dir`] and is set with [`AsfBurstSearchBuilder::cache_dir`].
//!
//! Responses without bursts are not cached, since ASF may not have ingested the requested
//! acquisitions yet. A cached response is never refreshed, so a query whose time range was
//! only partly acquired when it was cached keeps returning the bursts found then; use
//! [`AsfBurstSearchBuilder::cache`]`(false)` to search again and replace the cached response.
//!
//! # Sources
//!
//! 1. ASF Search API keywords: <https://docs.asf.alaska.edu/api/keywords/>
//!
//! [\[1\]]: https://docs.asf.alaska.edu/api/keywords/

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use log::{info, warn};
use serde_json::Value;
use ureq::Body;
use ureq::http::Response;
use url::Url;

use crate::datasets::asf::burst_download::{BurstRequest, Polarization};
use crate::datasets::http::{
    RetryError, TempPath, USER_AGENT, body_snippet, call_with_retries, path_with_suffix,
};
use crate::granule_id::IWSwath;

/// URL of the ASF Search API (parameter search endpoint).
pub const ASF_SEARCH_API_URL: &str = "https://api.daac.asf.alaska.edu/services/search/param";

/// The default cache directory: `asf_search` under [`cache_root`](crate::datasets::cache_root),
/// i.e. `{user cache dir}/psi_insar_rs/asf_search`.
pub fn default_cache_dir() -> PathBuf {
    crate::datasets::cache_root().join("asf_search")
}

/// Error type for ASF search operations
#[derive(Debug, thiserror::Error)]
pub enum AsfSearchError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] ureq::Error),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("HTTP {status} from {url}: {body}")]
    Http {
        status: u16,
        url: String,
        body: String,
    },
    /// The server answered with something unexpected
    #[error("Unexpected response: {0}")]
    InvalidResponse(String),
}

/// The area that the bursts must intersect, in degrees (WGS84).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchArea {
    Point {
        lat: f64,
        lon: f64,
    },
    /// A latitude/longitude box. It must not cross the antimeridian (`min_lon <= max_lon`).
    BoundingBox {
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    },
}

impl SearchArea {
    /// A box of `half_size_lat` × `half_size_lon` degrees on each side of (`lat`, `lon`).
    pub fn around(lat: f64, lon: f64, half_size_lat: f64, half_size_lon: f64) -> Self {
        SearchArea::BoundingBox {
            min_lat: lat - half_size_lat,
            min_lon: lon - half_size_lon,
            max_lat: lat + half_size_lat,
            max_lon: lon + half_size_lon,
        }
    }

    /// Whether (`lat`, `lon`) is the point, or lies in the box (edges included).
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        match *self {
            SearchArea::Point {
                lat: p_lat,
                lon: p_lon,
            } => lat == p_lat && lon == p_lon,
            SearchArea::BoundingBox {
                min_lat,
                min_lon,
                max_lat,
                max_lon,
            } => (min_lat..=max_lat).contains(&lat) && (min_lon..=max_lon).contains(&lon),
        }
    }

    /// The area in WKT, as the `intersectsWith` keyword expects it (longitude first).
    fn wkt(&self) -> String {
        match *self {
            SearchArea::Point { lat, lon } => format!("POINT({lon} {lat})"),
            SearchArea::BoundingBox {
                min_lat,
                min_lon,
                max_lat,
                max_lon,
            } => format!(
                "POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},\
                 {min_lon} {max_lat},{min_lon} {min_lat}))"
            ),
        }
    }

    /// The part of the cache file name that identifies the area.
    fn cache_key(&self) -> String {
        match *self {
            SearchArea::Point { lat, lon } => format!("{lat}_{lon}"),
            SearchArea::BoundingBox {
                min_lat,
                min_lon,
                max_lat,
                max_lon,
            } => format!("{min_lat}_{min_lon}_{max_lat}_{max_lon}"),
        }
    }
}

/// Which bursts to search for.
#[derive(Debug, Clone, PartialEq)]
pub struct BurstSearchQuery {
    /// The area the bursts must intersect.
    pub area: SearchArea,
    /// Start of the acquisition time range.
    pub start: DateTime<Utc>,
    /// End of the acquisition time range.
    pub end: DateTime<Utc>,
    pub polarization: Polarization,
    /// Only bursts of this subswath, if set.
    pub subswath: Option<IWSwath>,
}

impl BurstSearchQuery {
    /// Bursts of any subswath intersecting `area`, acquired between `start` and `end`.
    pub fn new(
        area: SearchArea,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        polarization: Polarization,
    ) -> Self {
        BurstSearchQuery {
            area,
            start,
            end,
            polarization,
            subswath: None,
        }
    }

    /// Only bursts of `subswath`.
    pub fn subswath(mut self, subswath: IWSwath) -> Self {
        self.subswath = Some(subswath);
        self
    }
}

/// A burst found by [`AsfBurstSearch::search`].
#[derive(Debug, Clone, PartialEq)]
pub struct BurstSearchResult {
    /// Parent SLC product name, without the `.SAFE` extension.
    pub granule: String,
    /// ESA burst ID, `{relative orbit}_{burst id}_{subswath}`, e.g. `156_333121_IW1`. Bursts
    /// with the same ID image the same area from the same track, so they form a stack.
    pub full_burst_id: String,
    pub subswath: IWSwath,
    pub polarization: Polarization,
    /// Index of the burst within the subswath of the parent product (0-based).
    pub burst_index: u32,
    pub start_time: DateTime<Utc>,
    /// `ASCENDING` or `DESCENDING`.
    pub flight_direction: String,
    /// Latitude of the burst footprint's center, in degrees.
    pub center_lat: f64,
    /// Longitude of the burst footprint's center, in degrees.
    pub center_lon: f64,
}

impl BurstSearchResult {
    /// The request that downloads this burst with
    /// [`AsfBurstDownloader`](super::burst_download::AsfBurstDownloader).
    pub fn request(&self) -> BurstRequest {
        BurstRequest::new(
            self.granule.clone(),
            self.subswath,
            self.polarization,
            self.burst_index,
        )
    }

    /// Parses a GeoJSON feature of the search response.
    fn from_feature(feature: &Value) -> Result<Self, AsfSearchError> {
        let invalid = |what: &str| {
            AsfSearchError::InvalidResponse(format!("Burst feature with {what}: {feature}"))
        };
        let properties = &feature["properties"];
        let burst = &properties["burst"];
        let str_field = |value: &Value, name: &str| {
            value[name]
                .as_str()
                .ok_or_else(|| invalid(&format!("no `{name}`")))
                .map(str::to_string)
        };
        let f64_field = |name: &str| {
            properties[name]
                .as_f64()
                .ok_or_else(|| invalid(&format!("no `{name}`")))
        };

        let subswath: IWSwath = str_field(burst, "subswath")?
            .parse()
            .map_err(|_| invalid("an invalid `subswath`"))?;
        let polarization: Polarization = str_field(properties, "polarization")?
            .parse()
            .map_err(|_| invalid("an invalid `polarization`"))?;
        let burst_index = burst["burstIndex"]
            .as_u64()
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| invalid("no `burstIndex`"))?;
        let start_time = DateTime::parse_from_rfc3339(&str_field(properties, "startTime")?)
            .map_err(|_| invalid("an invalid `startTime`"))?
            .to_utc();

        // The granule is only named in the extractor URL, whose other path segments must
        // match the burst: `/{granule}/{subswath}/{pol}/{burst_index}.tiff`.
        let url =
            Url::parse(&str_field(properties, "url")?).map_err(|_| invalid("an invalid `url`"))?;
        let segments: Vec<&str> = url.path_segments().into_iter().flatten().collect();
        let expected_tail = format!("{subswath}/{polarization}/{burst_index}.tiff");
        let granule = match segments[..] {
            [granule, swath, pol, file] if format!("{swath}/{pol}/{file}") == expected_tail => {
                granule.to_string()
            }
            _ => return Err(invalid("a `url` that does not match the burst")),
        };

        Ok(BurstSearchResult {
            granule,
            full_burst_id: str_field(burst, "fullBurstID")?,
            subswath,
            polarization,
            burst_index,
            start_time,
            flight_direction: str_field(properties, "flightDirection")?,
            center_lat: f64_field("centerLat")?,
            center_lon: f64_field("centerLon")?,
        })
    }
}

/// Builder for [`AsfBurstSearch`].
pub struct AsfBurstSearchBuilder {
    cache_dir: PathBuf,
    cache_enabled: bool,
    max_retries: u32,
    initial_backoff: Duration,
    search_url: String,
}

impl Default for AsfBurstSearchBuilder {
    fn default() -> Self {
        AsfBurstSearchBuilder {
            cache_dir: default_cache_dir(),
            cache_enabled: true,
            max_retries: 3,
            initial_backoff: Duration::from_secs(2),
            search_url: ASF_SEARCH_API_URL.to_string(),
        }
    }
}

impl AsfBurstSearchBuilder {
    /// Directory where search responses are stored. Defaults to [`default_cache_dir`]. A
    /// relative path is resolved against the working directory at the time of each search.
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }

    /// Whether cached search responses are reused (the default). When disabled, every query
    /// is sent to ASF and its response replaces the cached one.
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
        self.search_url = format!("{base_url}/search");
        self.initial_backoff = Duration::from_millis(10);
        self
    }

    pub fn build(self) -> AsfBurstSearch {
        let config = ureq::Agent::config_builder()
            .https_only(cfg!(not(test)))
            .http_status_as_error(false)
            .user_agent(USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .build();
        AsfBurstSearch {
            agent: ureq::Agent::new_with_config(config),
            cache_dir: self.cache_dir,
            cache_enabled: self.cache_enabled,
            max_retries: self.max_retries,
            initial_backoff: self.initial_backoff,
            search_url: self.search_url,
        }
    }
}

/// Searches Sentinel-1 bursts with the ASF Search API and caches the responses on disk. Create
/// it with [`AsfBurstSearch::builder`].
pub struct AsfBurstSearch {
    agent: ureq::Agent,
    cache_dir: PathBuf,
    cache_enabled: bool,
    max_retries: u32,
    initial_backoff: Duration,
    search_url: String,
}

impl AsfBurstSearch {
    pub fn builder() -> AsfBurstSearchBuilder {
        AsfBurstSearchBuilder::default()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Where the response to `query` is stored:
    /// `{cache_dir}/{pol}_{area}_{start}_{end}.geojson`, with compact UTC times.
    pub fn cached_path(&self, query: &BurstSearchQuery) -> PathBuf {
        let time = |t: DateTime<Utc>| t.format("%Y%m%dT%H%M%SZ");
        self.cache_dir.join(format!(
            "{}_{}_{}_{}.geojson",
            query.polarization,
            query.area.cache_key(),
            time(query.start),
            time(query.end)
        ))
    }

    /// The bursts matching `query`, oldest first. Empty if none match.
    ///
    /// With caching enabled, a cached response to `query` is used without network access.
    pub fn search(
        &self,
        query: &BurstSearchQuery,
    ) -> Result<Vec<BurstSearchResult>, AsfSearchError> {
        let path = self.cached_path(query);
        if self.cache_enabled && path.exists() {
            let cached = fs::read_to_string(&path)
                .map_err(AsfSearchError::from)
                .and_then(|text| parse_response(&text));
            match cached {
                Ok(bursts) => {
                    info!("Using cached search response {}", path.display());
                    return Ok(filter_subswath(bursts, query));
                }
                Err(err) => warn!(
                    "Ignoring invalid cached search response {} ({err}), searching again",
                    path.display()
                ),
            }
        }

        let text = self.request(query)?;
        let bursts = parse_response(&text)?;
        // Only valid responses are cached, and empty ones may be filled in later.
        if !bursts.is_empty() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let part_path = TempPath(path_with_suffix(&path, ".part"));
            fs::write(&part_path.0, &text)?;
            fs::rename(&part_path.0, &path)?;
            info!("Cached search response {}", path.display());
        }
        Ok(filter_subswath(bursts, query))
    }

    /// Sends `query` to ASF and returns the body of the response.
    fn request(&self, query: &BurstSearchQuery) -> Result<String, AsfSearchError> {
        let area = query.area.wkt();
        let start = query.start.to_rfc3339_opts(SecondsFormat::Secs, true);
        let end = query.end.to_rfc3339_opts(SecondsFormat::Secs, true);
        let polarization = query.polarization.to_string();
        let url = &self.search_url;
        let mut resp = call_with_retries(
            "burst search",
            self.max_retries,
            self.initial_backoff,
            || {
                self.agent
                    .get(url)
                    .query("platform", "SENTINEL-1")
                    .query("processingLevel", "BURST")
                    .query("polarization", &polarization)
                    .query("intersectsWith", &area)
                    .query("start", &start)
                    .query("end", &end)
                    .query("output", "geojson")
                    .call()
            },
        )
        .map_err(|err| match err {
            RetryError::Request(err) => err.into(),
            RetryError::Exhausted {
                retries,
                last_error,
            } => AsfSearchError::InvalidResponse(format!(
                "burst search failed after {retries} retries: {last_error}"
            )),
        })?;
        if resp.status() != 200 {
            return Err(http_error(url, &mut resp));
        }
        Ok(resp.body_mut().read_to_string()?)
    }
}

/// All the bursts of a search response, oldest first.
fn parse_response(text: &str) -> Result<Vec<BurstSearchResult>, AsfSearchError> {
    let json: Value = serde_json::from_str(text).map_err(|err| {
        AsfSearchError::InvalidResponse(format!("Search response is not JSON: {err}"))
    })?;
    let features = json["features"].as_array().ok_or_else(|| {
        AsfSearchError::InvalidResponse(format!("Search response without `features`: {json}"))
    })?;
    let mut bursts = features
        .iter()
        .map(BurstSearchResult::from_feature)
        .collect::<Result<Vec<_>, _>>()?;
    bursts.sort_by_key(|burst| burst.start_time);
    Ok(bursts)
}

fn filter_subswath(
    mut bursts: Vec<BurstSearchResult>,
    query: &BurstSearchQuery,
) -> Vec<BurstSearchResult> {
    bursts.retain(|burst| {
        query
            .subswath
            .is_none_or(|subswath| burst.subswath == subswath)
    });
    bursts
}

fn http_error(url: &str, resp: &mut Response<Body>) -> AsfSearchError {
    AsfSearchError::Http {
        status: resp.status().as_u16(),
        url: url.to_string(),
        body: body_snippet(resp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::mock_server::{MockResponse, mock_server};

    fn utc(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().to_utc()
    }

    /// A search feature, trimmed down from an ASF response observed in October 2026.
    fn feature(granule: &str, subswath: &str, burst_index: u32, start_time: &str) -> String {
        format!(
            r#"{{
                "type": "Feature",
                "geometry": {{"type": "Polygon", "coordinates": []}},
                "properties": {{
                    "centerLat": 32.80124,
                    "centerLon": 130.80755,
                    "flightDirection": "ASCENDING",
                    "polarization": "VV",
                    "startTime": "{start_time}",
                    "url": "https://sentinel1-burst.asf.alaska.edu/{granule}/{subswath}/VV/{burst_index}.tiff",
                    "burst": {{
                        "fullBurstID": "156_333121_{subswath}",
                        "burstIndex": {burst_index},
                        "subswath": "{subswath}"
                    }}
                }}
            }}"#
        )
    }

    fn search_response(features: &[String]) -> MockResponse {
        MockResponse::json_body(
            200,
            format!(
                r#"{{"type": "FeatureCollection", "features": [{}]}}"#,
                features.join(",")
            ),
        )
    }

    const REFERENCE: &str = "S1A_IW_SLC__1SSV_20160408T091355_20160408T091430_010728_01001F_83EB";
    const SECONDARY: &str = "S1A_IW_SLC__1SSV_20160420T091355_20160420T091423_010903_010569_F9CE";

    fn kumamoto_query() -> BurstSearchQuery {
        BurstSearchQuery::new(
            SearchArea::Point {
                lat: 32.7906,
                lon: 130.7543,
            },
            utc("2016-04-08T00:00:00Z"),
            utc("2016-04-21T00:00:00Z"),
            Polarization::VV,
        )
    }

    /// A search against the mock server at `base`, with an empty cache directory named `name`.
    fn test_builder(name: &str, base: &str) -> AsfBurstSearchBuilder {
        let cache_dir = std::env::temp_dir()
            .join("psi_insar_rs_asf_burst_search")
            .join(name);
        let _ = fs::remove_dir_all(&cache_dir);
        AsfBurstSearch::builder()
            .cache_dir(cache_dir)
            .max_retries(2)
            .base_url(base)
    }

    #[test]
    fn parses_bursts_oldest_first() {
        let (base, requests) = mock_server(vec![search_response(&[
            feature(SECONDARY, "IW1", 2, "2016-04-20T09:14:02Z"),
            feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z"),
        ])]);
        let bursts = test_builder("parses_bursts_oldest_first", &base)
            .build()
            .search(&kumamoto_query())
            .unwrap();

        assert_eq!(bursts.len(), 2);
        let reference = &bursts[0];
        assert_eq!(reference.granule, REFERENCE);
        assert_eq!(reference.full_burst_id, "156_333121_IW1");
        assert_eq!(reference.start_time, utc("2016-04-08T09:14:02Z"));
        assert_eq!(reference.flight_direction, "ASCENDING");
        assert_eq!(
            reference.request(),
            BurstRequest::new(REFERENCE, IWSwath::IW1, Polarization::VV, 2)
        );
        assert_eq!(bursts[1].granule, SECONDARY);

        let target = &requests.all()[0].target;
        for keyword in [
            "processingLevel=BURST",
            "polarization=VV",
            "start=2016-04-08T00%3A00%3A00Z",
            "output=geojson",
        ] {
            assert!(target.contains(keyword), "{keyword} not in {target}");
        }
        assert!(!target.contains("beamSwath"));
    }

    #[test]
    fn filters_subswath() {
        let (base, _) = mock_server(vec![search_response(&[
            feature(SECONDARY, "IW2", 8, "2016-04-20T21:16:54Z"),
            feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z"),
        ])]);
        let query = kumamoto_query().subswath(IWSwath::IW1);
        let bursts = test_builder("filters_subswath", &base)
            .build()
            .search(&query)
            .unwrap();
        assert_eq!(bursts.len(), 1);
        assert_eq!(bursts[0].subswath, IWSwath::IW1);
    }

    #[test]
    fn empty_search_is_empty() {
        let (base, _) = mock_server(vec![search_response(&[])]);
        let bursts = test_builder("empty_search_is_empty", &base)
            .build()
            .search(&kumamoto_query())
            .unwrap();
        assert!(bursts.is_empty());
    }

    #[test]
    fn mismatched_url_is_invalid() {
        let bad = feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z")
            .replace("/VV/2.tiff", "/VV/3.tiff");
        let (base, _) = mock_server(vec![search_response(&[bad])]);
        let result = test_builder("mismatched_url_is_invalid", &base)
            .build()
            .search(&kumamoto_query());
        assert!(matches!(result, Err(AsfSearchError::InvalidResponse(_))));
    }

    #[test]
    fn bad_request_is_an_http_error() {
        let (base, _) = mock_server(vec![MockResponse::json_body(
            400,
            r#"{"error": {"type": "ERROR", "report": "Invalid wkt"}}"#,
        )]);
        let result = test_builder("bad_request_is_an_http_error", &base)
            .build()
            .search(&kumamoto_query());
        assert!(matches!(
            result,
            Err(AsfSearchError::Http { status: 400, .. })
        ));
    }

    #[test]
    fn retries_transient_errors() {
        let (base, requests) = mock_server(vec![
            MockResponse::json(503, "Service Unavailable"),
            search_response(&[feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z")]),
        ]);
        let bursts = test_builder("retries_transient_errors", &base)
            .build()
            .search(&kumamoto_query())
            .unwrap();
        assert_eq!(bursts.len(), 1);
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn cached_response_needs_no_network() {
        let (base, requests) = mock_server(vec![search_response(&[
            feature(SECONDARY, "IW2", 8, "2016-04-20T21:16:54Z"),
            feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z"),
        ])]);
        let search = test_builder("cached", &base).build();
        let all = search.search(&kumamoto_query()).unwrap();
        assert!(search.cached_path(&kumamoto_query()).is_file());

        // Served from the cache: the mock server has no responses left. The subswath filter is
        // applied to the cached response too.
        assert_eq!(search.search(&kumamoto_query()).unwrap(), all);
        let iw1 = search
            .search(&kumamoto_query().subswath(IWSwath::IW1))
            .unwrap();
        assert_eq!(iw1.len(), 1);
        assert_eq!(iw1[0].granule, REFERENCE);
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn disabled_cache_searches_again_and_replaces_response() {
        let (base, requests) = mock_server(vec![
            search_response(&[feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z")]),
            search_response(&[
                feature(REFERENCE, "IW1", 2, "2016-04-08T09:14:02Z"),
                feature(SECONDARY, "IW1", 2, "2016-04-20T09:14:02Z"),
            ]),
        ]);
        let search = test_builder("disabled_cache", &base).build();
        assert_eq!(search.search(&kumamoto_query()).unwrap().len(), 1);

        let uncached = AsfBurstSearch::builder()
            .cache_dir(search.cache_dir())
            .cache(false)
            .base_url(&base)
            .build();
        assert_eq!(uncached.search(&kumamoto_query()).unwrap().len(), 2);
        assert_eq!(requests.len(), 2);
        // The cached response was replaced by the second one.
        assert_eq!(search.search(&kumamoto_query()).unwrap().len(), 2);
    }

    #[test]
    fn empty_response_is_not_cached() {
        let (base, requests) = mock_server(vec![search_response(&[]), search_response(&[])]);
        let search = test_builder("empty_not_cached", &base).build();
        assert!(search.search(&kumamoto_query()).unwrap().is_empty());
        assert!(!search.cached_path(&kumamoto_query()).exists());
        assert!(search.search(&kumamoto_query()).unwrap().is_empty());
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn invalid_cached_response_is_replaced() {
        let (base, requests) = mock_server(vec![search_response(&[feature(
            REFERENCE,
            "IW1",
            2,
            "2016-04-08T09:14:02Z",
        )])]);
        let search = test_builder("invalid_cache", &base).build();
        let path = search.cached_path(&kumamoto_query());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "{\"type\": \"FeatureCollection\", \"feat").unwrap();

        assert_eq!(search.search(&kumamoto_query()).unwrap().len(), 1);
        assert_eq!(requests.len(), 1);
        assert!(parse_response(&fs::read_to_string(&path).unwrap()).is_ok());
    }

    #[test]
    fn bounding_box_is_sent_as_a_polygon() {
        let (base, requests) = mock_server(vec![search_response(&[feature(
            REFERENCE,
            "IW1",
            2,
            "2016-04-08T09:14:02Z",
        )])]);
        let area = SearchArea::around(32.8, 130.75, 0.2, 0.25);
        let query = BurstSearchQuery {
            area,
            ..kumamoto_query()
        };
        let search = test_builder("bounding_box", &base).build();
        assert_eq!(search.search(&query).unwrap().len(), 1);

        let target = &requests.all()[0].target;
        let wkt = url::form_urlencoded::parse(target.split_once('?').unwrap().1.as_bytes())
            .find(|(key, _)| key == "intersectsWith")
            .unwrap()
            .1
            .into_owned();
        assert_eq!(
            wkt,
            "POLYGON((130.5 32.599999999999994,131 32.599999999999994,131 33,130.5 33,\
             130.5 32.599999999999994))"
        );
        let name = search.cached_path(&query);
        let name = name.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("VV_32.599999999999994_130.5_33_131_"),
            "{name}"
        );
    }

    #[test]
    fn area_contains() {
        let area = SearchArea::around(32.8, 130.75, 0.2, 0.25);
        assert!(area.contains(32.8, 130.75));
        assert!(area.contains(32.99, 130.51));
        assert!(!area.contains(33.01, 130.75));
        assert!(!area.contains(32.8, 130.49));
        let point = SearchArea::Point {
            lat: 32.8,
            lon: 130.75,
        };
        assert!(point.contains(32.8, 130.75));
        assert!(!point.contains(32.8, 130.76));
    }
}
