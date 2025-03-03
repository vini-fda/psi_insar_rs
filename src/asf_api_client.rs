use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Client, Error as ReqwestError, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use url::Url;

/// Error type for ASF API operations
#[derive(Error, Debug)]
pub enum AsfApiError {
    #[error("authentication failed: {0}")]
    AuthenticationError(String),

    #[error("network error: {0}")]
    NetworkError(#[from] ReqwestError),

    #[error("request error: {0}")]
    RequestError(String),

    #[error("download error: {0}")]
    DownloadError(String),

    #[error("io error: {0}")]
    IoError(#[from] io::Error),

    #[error("parse error: {0}")]
    ParseError(String),

    #[error("invalid url: {0}")]
    InvalidUrl(#[from] url::ParseError),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("rate limit exceeded")]
    RateLimitExceeded,
}

/// Represents a search filter for the ASF API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfSearchFilter {
    pub platform: Option<String>,
    pub instrument: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub processing_level: Option<String>,
    pub beam_mode: Option<String>,
    pub polarization: Option<String>,
    pub min_latitude: Option<f64>,
    pub max_latitude: Option<f64>,
    pub min_longitude: Option<f64>,
    pub max_longitude: Option<f64>,
    pub polygon: Option<Vec<(f64, f64)>>,
    pub intersects_point: Option<(f64, f64)>,
    pub max_results: Option<usize>,
    pub product_type: Option<String>,
    pub relativeorbit: Option<String>,
    pub frame: Option<String>,
}

impl AsfSearchFilter {
    pub fn to_query_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        if let Some(ref platform) = self.platform {
            params.insert("platform".to_string(), platform.clone());
        }

        if let Some(ref instrument) = self.instrument {
            params.insert("instrument".to_string(), instrument.clone());
        }

        if let Some(start_date) = self.start_date {
            params.insert("start".to_string(), start_date.to_rfc3339());
        }

        if let Some(end_date) = self.end_date {
            params.insert("end".to_string(), end_date.to_rfc3339());
        }

        if let Some(ref processing_level) = self.processing_level {
            params.insert("processingLevel".to_string(), processing_level.clone());
        }

        if let Some(ref beam_mode) = self.beam_mode {
            params.insert("beamMode".to_string(), beam_mode.clone());
        }

        if let Some(ref polarization) = self.polarization {
            params.insert("polarization".to_string(), polarization.clone());
        }

        if let Some(min_latitude) = self.min_latitude {
            params.insert("minLatitude".to_string(), min_latitude.to_string());
        }

        if let Some(max_latitude) = self.max_latitude {
            params.insert("maxLatitude".to_string(), max_latitude.to_string());
        }

        if let Some(min_longitude) = self.min_longitude {
            params.insert("minLongitude".to_string(), min_longitude.to_string());
        }

        if let Some(max_longitude) = self.max_longitude {
            params.insert("maxLongitude".to_string(), max_longitude.to_string());
        }

        if let Some(ref polygon) = self.polygon {
            let polygon_str = polygon
                .iter()
                .map(|(lon, lat)| format!("{},{}", lon, lat))
                .collect::<Vec<String>>()
                .join(",");
            params.insert("polygon".to_string(), polygon_str);
        }

        if let Some((lon, lat)) = self.intersects_point {
            params.insert(
                "intersectsWith".to_string(),
                format!("POINT({} {})", lon, lat),
            );
        }

        if let Some(max_results) = self.max_results {
            params.insert("maxResults".to_string(), max_results.to_string());
        }

        if let Some(ref product_type) = self.product_type {
            params.insert("productType".to_string(), product_type.clone());
        }

        if let Some(ref relativeorbit) = self.relativeorbit {
            params.insert("relativeOrbit".to_string(), relativeorbit.clone());
        }

        if let Some(ref frame) = self.frame {
            params.insert("frame".to_string(), frame.clone());
        }

        params
    }
}

/// Represents information about a dataset in ASF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfDataset {
    pub id: String,
    pub dataset_id: String,
    pub granule_name: String,
    pub url: String,
    pub bytes: Option<u64>,
    pub beam_mode: Option<String>,
    pub polarization: Option<String>,
    pub acquisition_date: Option<DateTime<Utc>>,
    pub path: Option<u32>,
    pub frame: Option<u32>,
    pub orbit_number: Option<u32>,
    pub processing_level: Option<String>,
    pub processing_date: Option<DateTime<Utc>>,
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Represents a file in a HyP3 job result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfHyp3File {
    pub filename: String,
    pub size: u64,
    pub url: String,
    pub hash: Option<String>,
}

/// Represents parameters for a HyP3 job in ASF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfHyp3JobParameters {
    pub granules: Vec<String>,
    #[serde(flatten)]
    pub additional_parameters: HashMap<String, Value>,
}

/// Represents a HyP3 job in ASF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfHyp3Job {
    pub job_id: String,
    pub status_code: String,
    pub request_time: DateTime<Utc>,
    pub user_id: String,
    pub job_type: String,
    pub job_parameters: AsfHyp3JobParameters,
    pub files: Option<Vec<AsfHyp3File>>,
    pub browse_images: Option<Vec<String>>,
    pub thumbnail_images: Option<Vec<String>>,
    pub expiration_time: Option<DateTime<Utc>>,
}

/// Represents the response from ASF API search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfSearchResponse {
    pub results: Vec<AsfDataset>,
}

/// Represents the response from ASF HyP3 API jobs query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfHyp3JobsResponse {
    pub jobs: Vec<AsfHyp3Job>,
    pub next: Option<String>,
}

/// Credentials for ASF API authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsfCredentials {
    pub username: String,
    pub password: String,
}

/// Client for interacting with the ASF API
#[derive(Clone)]
pub struct AsfApiClient {
    client: Client,
    credentials: Option<AsfCredentials>,
    api_base_url: String,
    hyp3_base_url: String,
}

impl AsfApiClient {
    /// Create a new ASF API client
    pub fn new() -> Result<Self, AsfApiError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;

        Ok(Self {
            client,
            credentials: None,
            api_base_url: "https://api.daac.asf.alaska.edu".to_string(),
            hyp3_base_url: "https://hyp3-api.asf.alaska.edu/api".to_string(),
        })
    }

    /// Set credentials for ASF API authentication
    pub fn with_credentials(mut self, username: &str, password: &str) -> Self {
        self.credentials = Some(AsfCredentials {
            username: username.to_string(),
            password: password.to_string(),
        });
        self
    }

    /// Search for datasets using the ASF API
    pub async fn search(&self, filter: &AsfSearchFilter) -> Result<Vec<AsfDataset>, AsfApiError> {
        let url = format!("{}/services/search/param", self.api_base_url);
        let params = filter.to_query_params();

        let mut request = self.client.get(&url);

        // Add query parameters
        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }

        // Add basic auth if credentials are set
        if let Some(ref creds) = self.credentials {
            request = request.basic_auth(&creds.username, Some(&creds.password));
        }

        let response: reqwest::Response = request.send().await?;

        match response.status() {
            StatusCode::OK => {
                let search_response: AsfSearchResponse = response.json().await?;
                Ok(search_response.results)
            }
            StatusCode::TOO_MANY_REQUESTS => Err(AsfApiError::RateLimitExceeded),
            _ => {
                let status = response.status();
                let error_text = response.text().await?;
                Err(AsfApiError::RequestError(format!(
                    "Search request failed: {} - {}",
                    status, error_text
                )))
            }
        }
    }

    /// Download a dataset from ASF
    pub async fn download_dataset(
        &self,
        dataset: &AsfDataset,
        output_dir: &Path,
        show_progress: bool,
    ) -> Result<PathBuf, AsfApiError> {
        // Create the output directory if it doesn't exist
        if !output_dir.exists() {
            fs::create_dir_all(output_dir)?;
        }

        // Determine the output file path
        let url = Url::parse(&dataset.url)?;
        let url_path = url.path();
        let filename = url_path
            .split('/')
            .last()
            .ok_or_else(|| AsfApiError::InvalidUrl(url::ParseError::EmptyHost))?;
        let output_path = output_dir.join(filename);

        // Start the download
        let mut request = self.client.get(&dataset.url);

        // Add basic auth if credentials are set
        if let Some(ref creds) = self.credentials {
            request = request.basic_auth(&creds.username, Some(&creds.password));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(AsfApiError::DownloadError(format!(
                "Download failed: {} - {}",
                status, error_text
            )));
        }

        // Get the content length for progress reporting
        let content_length = response.content_length().unwrap_or(0);

        // Setup progress bar if requested
        let progress_bar = if show_progress && content_length > 0 {
            let pb = ProgressBar::new(content_length);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .expect("Failed to set progress bar style")
                .progress_chars("#>-"));
            Some(pb)
        } else {
            None
        };

        // Create the output file
        let mut output_file = File::create(&output_path)?;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            output_file.write_all(&chunk)?;

            if let Some(ref pb) = progress_bar {
                downloaded += chunk.len() as u64;
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = progress_bar {
            pb.finish_with_message("Download complete");
        }

        Ok(output_path)
    }

    /// Submit a HyP3 job
    pub async fn submit_hyp3_job(
        &self,
        job_type: &str,
        granules: Vec<String>,
        parameters: HashMap<String, Value>,
    ) -> Result<AsfHyp3Job, AsfApiError> {
        let url = format!("{}/v3/jobs", self.hyp3_base_url);

        let mut job_params = HashMap::new();
        job_params.insert("granules".to_string(), json!(granules));

        for (key, value) in parameters {
            job_params.insert(key, value);
        }

        let job_request = json!({
            "job_type": job_type,
            "job_parameters": job_params
        });

        let mut request = self.client.post(&url).json(&job_request);

        // Add basic auth if credentials are set
        if let Some(ref creds) = self.credentials {
            request = request.basic_auth(&creds.username, Some(&creds.password));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(AsfApiError::RequestError(format!(
                "HyP3 job submission failed: {} - {}",
                status, error_text
            )));
        }

        let job: AsfHyp3Job = response.json().await?;
        Ok(job)
    }

    /// Get a HyP3 job by ID
    pub async fn get_hyp3_job(&self, job_id: &str) -> Result<AsfHyp3Job, AsfApiError> {
        let url = format!("{}/v3/jobs/{}", self.hyp3_base_url, job_id);

        let mut request = self.client.get(&url);

        // Add basic auth if credentials are set
        if let Some(ref creds) = self.credentials {
            request = request.basic_auth(&creds.username, Some(&creds.password));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(AsfApiError::RequestError(format!(
                "Failed to get HyP3 job {}: {} - {}",
                job_id, status, error_text
            )));
        }

        let job: AsfHyp3Job = response.json().await?;
        Ok(job)
    }

    /// List HyP3 jobs
    pub async fn list_hyp3_jobs(
        &self,
        status: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<AsfHyp3Job>, AsfApiError> {
        let mut url = format!("{}/v3/jobs", self.hyp3_base_url);

        let mut query_params = vec![];
        if let Some(status_value) = status {
            query_params.push(format!("status={}", status_value));
        }

        if let Some(limit_value) = limit {
            query_params.push(format!("limit={}", limit_value));
        }

        if !query_params.is_empty() {
            url = format!("{}?{}", url, query_params.join("&"));
        }

        let mut request = self.client.get(&url);

        // Add basic auth if credentials are set
        if let Some(ref creds) = self.credentials {
            request = request.basic_auth(&creds.username, Some(&creds.password));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(AsfApiError::RequestError(format!(
                "Failed to list HyP3 jobs: {} - {}",
                status, error_text
            )));
        }

        let mut jobs_response: AsfHyp3JobsResponse = response.json().await?;
        let mut all_jobs = jobs_response.jobs;

        // Handle pagination if there are more results
        let mut next_url = jobs_response.next;
        while let Some(url) = next_url {
            let mut request = self.client.get(&url);

            // Add basic auth if credentials are set
            if let Some(ref creds) = self.credentials {
                request = request.basic_auth(&creds.username, Some(&creds.password));
            }

            let response = request.send().await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await?;
                return Err(AsfApiError::RequestError(format!(
                    "Failed to get next page of HyP3 jobs: {} - {}",
                    status, error_text
                )));
            }

            let jobs_response: AsfHyp3JobsResponse = response.json().await?;
            all_jobs.extend(jobs_response.jobs);
            next_url = jobs_response.next;
        }

        Ok(all_jobs)
    }

    /// Download a HyP3 job result file
    pub async fn download_hyp3_job_file(
        &self,
        file: &AsfHyp3File,
        output_dir: &Path,
        show_progress: bool,
    ) -> Result<PathBuf, AsfApiError> {
        // Create the output directory if it doesn't exist
        if !output_dir.exists() {
            fs::create_dir_all(output_dir)?;
        }

        // Determine the output file path
        let output_path = output_dir.join(&file.filename);

        // Start the download
        let mut request = self.client.get(&file.url);

        // Add basic auth if credentials are set
        if let Some(ref creds) = self.credentials {
            request = request.basic_auth(&creds.username, Some(&creds.password));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(AsfApiError::DownloadError(format!(
                "Download failed: {} - {}",
                status, error_text
            )));
        }

        // Setup progress bar if requested
        let progress_bar = if show_progress {
            let pb = ProgressBar::new(file.size);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .expect("Failed to set progress bar style")
                .progress_chars("#>-"));
            Some(pb)
        } else {
            None
        };

        // Create the output file
        let mut output_file = File::create(&output_path)?;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            output_file.write_all(&chunk)?;

            if let Some(ref pb) = progress_bar {
                downloaded += chunk.len() as u64;
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = progress_bar {
            pb.finish_with_message("Download complete");
        }

        Ok(output_path)
    }

    /// Download all files from a HyP3 job
    pub async fn download_hyp3_job_files(
        &self,
        job: &AsfHyp3Job,
        output_dir: &Path,
        show_progress: bool,
    ) -> Result<Vec<PathBuf>, AsfApiError> {
        if job.files.is_none() || job.files.as_ref().unwrap().is_empty() {
            return Err(AsfApiError::DownloadError(
                "No files available for this job".to_string(),
            ));
        }

        let mut downloaded_files = Vec::new();

        for file in job.files.as_ref().unwrap() {
            let file_path = self
                .download_hyp3_job_file(file, output_dir, show_progress)
                .await?;
            downloaded_files.push(file_path);
        }

        Ok(downloaded_files)
    }

    /// Wait for a HyP3 job to complete
    pub async fn wait_for_hyp3_job(
        &self,
        job_id: &str,
        polling_interval_secs: u64,
        timeout_secs: Option<u64>,
    ) -> Result<AsfHyp3Job, AsfApiError> {
        let start_time = std::time::Instant::now();
        let timeout_duration = timeout_secs.map(Duration::from_secs);

        loop {
            // Check if we've exceeded the timeout
            if let Some(timeout) = timeout_duration {
                if start_time.elapsed() > timeout {
                    return Err(AsfApiError::RequestError(format!(
                        "Timeout waiting for job {} to complete",
                        job_id
                    )));
                }
            }

            // Get the current job status
            let job = self.get_hyp3_job(job_id).await?;

            match job.status_code.as_str() {
                "SUCCEEDED" => return Ok(job),
                "RUNNING" | "PENDING" => {
                    tokio::time::sleep(Duration::from_secs(polling_interval_secs)).await;
                    continue;
                }
                "FAILED" => {
                    return Err(AsfApiError::RequestError(format!("Job {} failed", job_id)));
                }
                _ => {
                    return Err(AsfApiError::RequestError(format!(
                        "Job {} has unknown status: {}",
                        job_id, job.status_code
                    )));
                }
            }
        }
    }

    /// Submit an InSAR processing job using HyP3
    pub async fn submit_insar_job(
        &self,
        reference_granule: &str,
        secondary_granule: &str,
        dem_matching: bool,
        include_dem: bool,
        include_inc_map: bool,
        include_displacement_maps: bool,
        include_look_vectors: bool,
    ) -> Result<AsfHyp3Job, AsfApiError> {
        let mut parameters = HashMap::new();
        parameters.insert("reference_granule".to_string(), json!(reference_granule));
        parameters.insert("secondary_granule".to_string(), json!(secondary_granule));
        parameters.insert("dem_matching".to_string(), json!(dem_matching));
        parameters.insert("include_dem".to_string(), json!(include_dem));
        parameters.insert("include_inc_map".to_string(), json!(include_inc_map));
        parameters.insert(
            "include_displacement_maps".to_string(),
            json!(include_displacement_maps),
        );
        parameters.insert(
            "include_look_vectors".to_string(),
            json!(include_look_vectors),
        );

        self.submit_hyp3_job("INSAR_GAMMA", vec![], parameters)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_search_filter_to_query_params() {
        let filter = AsfSearchFilter {
            platform: Some("Sentinel-1".to_string()),
            instrument: Some("C-SAR".to_string()),
            start_date: None,
            end_date: None,
            processing_level: Some("SLC".to_string()),
            beam_mode: Some("IW".to_string()),
            polarization: Some("DV".to_string()),
            min_latitude: Some(37.0),
            max_latitude: Some(38.0),
            min_longitude: Some(-123.0),
            max_longitude: Some(-122.0),
            polygon: None,
            intersects_point: None,
            max_results: Some(10),
            product_type: Some("SLC".to_string()),
            relativeorbit: None,
            frame: None,
        };

        let params = filter.to_query_params();

        assert_eq!(params.get("platform").unwrap(), "Sentinel-1");
        assert_eq!(params.get("instrument").unwrap(), "C-SAR");
        assert_eq!(params.get("processingLevel").unwrap(), "SLC");
        assert_eq!(params.get("beamMode").unwrap(), "IW");
        assert_eq!(params.get("polarization").unwrap(), "DV");
        assert_eq!(params.get("minLatitude").unwrap(), "37");
        assert_eq!(params.get("maxLatitude").unwrap(), "38");
        assert_eq!(params.get("minLongitude").unwrap(), "-123");
        assert_eq!(params.get("maxLongitude").unwrap(), "-122");
        assert_eq!(params.get("maxResults").unwrap(), "10");
        assert_eq!(params.get("productType").unwrap(), "SLC");
    }

    // Note: The following tests require actual credentials and would make real API calls
    // They are commented out to prevent unintended API usage and to pass CI builds

    /*
    #[test]
    fn test_search_for_sentinel1_data() {
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            let client = AsfApiClient::new().unwrap()
                .with_credentials("YOUR_ASF_USERNAME", "YOUR_ASF_PASSWORD");

            let filter = AsfSearchFilter {
                platform: Some("Sentinel-1".to_string()),
                processing_level: Some("SLC".to_string()),
                beam_mode: Some("IW".to_string()),
                min_latitude: Some(37.0),
                max_latitude: Some(38.0),
                min_longitude: Some(-123.0),
                max_longitude: Some(-122.0),
                max_results: Some(5),
                ..Default::default()
            };

            let results = client.search(&filter).await.unwrap();
            assert!(!results.is_empty());

            for dataset in &results {
                println!("{}: {}", dataset.granule_name, dataset.url);
            }
        });
    }
    */
}
