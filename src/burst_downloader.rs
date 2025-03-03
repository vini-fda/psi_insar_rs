use crate::asf_api_client::{AsfApiClient, AsfApiError, AsfSearchFilter};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::io::Seek;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::task;
use zip::ZipArchive;

/// Error type for Sentinel-1 burst downloading operations
#[derive(Error, Debug)]
pub enum BurstDownloadError {
    #[error("invalid burst identifier format: {0}")]
    InvalidBurstFormat(String),

    #[error("ASF API error: {0}")]
    AsfApiError(#[from] AsfApiError),

    #[error("io error: {0}")]
    IoError(#[from] io::Error),

    #[error("burst not found in data: {0}")]
    BurstNotFound(String),

    #[error("failed to extract burst from archive: {0}")]
    ExtractionError(String),

    #[error("XML parsing error: {0}")]
    XmlParsingError(String),

    #[error("external command error: {0}")]
    CommandError(String),

    #[error("failed to find SLC product for burst: {0}")]
    ProductNotFound(String),
}

/// Represents a Sentinel-1 burst identifier
///
/// Format: S1_ORBIT_SWH_DATETIME_POL_BURSTID-BURST
/// Example: S1_305967_IW3_20151022T122546_VV_5A48-BURST
///
/// Source: https://github.com/AlexeyPechnikov/pygmtsar/blob/master/pygmtsar/pygmtsar/ASF.py
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurstIdentifier {
    pub satellite: String,
    pub orbit: String,
    pub subswath: String,
    pub datetime: DateTime<Utc>,
    pub polarization: String,
    pub burst_id: String,
    pub raw_id: String,
}

impl BurstIdentifier {
    /// Parse a burst identifier from string
    pub fn parse(s: &str) -> Result<Self, BurstDownloadError> {
        lazy_static! {
            static ref BURST_RE: Regex =
                Regex::new(r"S1_(\d+)_(IW\d)_(\d{8}T\d{6})_([HV]{2})_([0-9A-F]+)-BURST").unwrap();
        }

        if let Some(captures) = BURST_RE.captures(s) {
            let orbit = captures.get(1).unwrap().as_str().to_string();
            let subswath = captures.get(2).unwrap().as_str().to_string();
            let datetime_str = captures.get(3).unwrap().as_str();
            let polarization = captures.get(4).unwrap().as_str().to_string();
            let burst_id = captures.get(5).unwrap().as_str().to_string();

            // Parse datetime
            let datetime = NaiveDateTime::parse_from_str(datetime_str, "%Y%m%dT%H%M%S")
                .map_err(|e| BurstDownloadError::InvalidBurstFormat(e.to_string()))?;
            let datetime = Utc.from_utc_datetime(&datetime);

            Ok(BurstIdentifier {
                satellite: "S1".to_string(),
                orbit,
                subswath,
                datetime,
                polarization,
                burst_id,
                raw_id: s.to_string(),
            })
        } else {
            Err(BurstDownloadError::InvalidBurstFormat(s.to_string()))
        }
    }

    /// Get the subswath number (1, 2, or 3)
    pub fn subswath_number(&self) -> Result<u8, BurstDownloadError> {
        if self.subswath.len() < 3 {
            return Err(BurstDownloadError::InvalidBurstFormat(format!(
                "Invalid subswath: {}",
                self.subswath
            )));
        }

        let num_str = &self.subswath[2..];
        num_str
            .parse::<u8>()
            .map_err(|e| BurstDownloadError::InvalidBurstFormat(e.to_string()))
    }

    /// Create a search filter to find the SLC product containing this burst
    pub fn to_search_filter(&self) -> AsfSearchFilter {
        // Create a time window around the burst time (±1 minute to account for slight time differences)
        let start_date = self.datetime - chrono::Duration::minutes(1);
        let end_date = self.datetime + chrono::Duration::minutes(1);

        AsfSearchFilter {
            platform: Some("Sentinel-1".to_string()),
            product_type: Some("SLC".to_string()),
            processing_level: Some("SLC".to_string()),
            beam_mode: Some("IW".to_string()),
            polarization: Some(self.polarization.clone()),
            start_date: Some(start_date),
            end_date: Some(end_date),
            // Use max_results to limit the search
            max_results: Some(10),
            // All other fields are None/default
            ..Default::default()
        }
    }
}

impl fmt::Display for BurstIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S1_{}_{}_{}_{}_{}-BURST",
            self.orbit,
            self.subswath,
            self.datetime.format("%Y%m%dT%H%M%S"),
            self.polarization,
            self.burst_id
        )
    }
}

/// Structure to organize burst downloads
///
/// Source: https://github.com/AlexeyPechnikov/pygmtsar/blob/master/pygmtsar/pygmtsar/ASF.py
pub struct BurstDownloader {
    client: AsfApiClient,
    output_dir: PathBuf,
    temp_dir: PathBuf,
    use_gdal: bool,
}

impl BurstDownloader {
    /// Create a new burst downloader
    pub fn new(client: AsfApiClient, output_dir: PathBuf) -> Result<Self, BurstDownloadError> {
        let temp_dir = output_dir.join("temp");
        fs::create_dir_all(&temp_dir)?;

        Ok(Self {
            client,
            output_dir,
            temp_dir,
            use_gdal: false, // Default to not using GDAL
        })
    }

    /// Enable GDAL usage for burst extraction (if available)
    pub fn with_gdal(mut self, use_gdal: bool) -> Self {
        self.use_gdal = use_gdal;
        self
    }

    /// Parse multiple burst identifiers from a string with one ID per line
    pub fn parse_burst_list(burst_list: &str) -> Result<Vec<BurstIdentifier>, BurstDownloadError> {
        let mut bursts = Vec::new();

        for line in burst_list.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let burst = BurstIdentifier::parse(line)?;
            bursts.push(burst);
        }

        Ok(bursts)
    }

    /// Download a single burst
    pub async fn download_burst(
        &self,
        burst: &BurstIdentifier,
    ) -> Result<PathBuf, BurstDownloadError> {
        // Create search filter for this burst
        let filter = burst.to_search_filter();

        // Search for SLC products matching this burst
        let datasets = self.client.search(&filter).await?;

        if datasets.is_empty() {
            return Err(BurstDownloadError::ProductNotFound(burst.raw_id.clone()));
        }

        // Create directory for this burst
        let burst_dir = self.output_dir.join(format!("{}", burst));
        fs::create_dir_all(&burst_dir)?;

        // Choose the first dataset (most relevant based on search criteria)
        let dataset = &datasets[0];
        println!(
            "Found SLC product: {} for burst: {}",
            dataset.granule_name, burst
        );

        // Download the SLC zip file to temp directory
        let zip_path = self
            .client
            .download_dataset(dataset, &self.temp_dir, true)
            .await?;
        println!("Downloaded SLC product to: {:?}", zip_path);

        // Extract burst data from the zip file
        let burst_path = self
            .extract_burst_from_zip(&zip_path, burst, &burst_dir)
            .await?;

        // Optionally clean up temp files
        // fs::remove_file(zip_path)?;

        Ok(burst_path)
    }

    /// Download multiple bursts in parallel
    pub async fn download_bursts(
        &self,
        bursts: &[BurstIdentifier],
    ) -> Result<Vec<PathBuf>, BurstDownloadError> {
        let multi_progress = MultiProgress::new();
        let progress_style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Failed to create progress style")
            .progress_chars("#>-");

        let total_progress = multi_progress.add(ProgressBar::new(bursts.len() as u64));
        total_progress.set_style(progress_style.clone());
        total_progress.set_message("Downloading bursts");

        let mut tasks = Vec::new();
        let mut results = Vec::new();

        for burst in bursts {
            let burst_clone = burst.clone();
            let client_clone = self.client.clone();
            let output_dir_clone = self.output_dir.clone();
            let temp_dir_clone = self.temp_dir.clone();
            let use_gdal = self.use_gdal;
            let progress = multi_progress.add(ProgressBar::new(3)); // 3 steps: search, download, extract
            progress.set_style(progress_style.clone());
            progress.set_message(format!("Processing {}", burst));

            // Spawn task for each burst
            let task = task::spawn(async move {
                // Create search filter for this burst
                let filter = burst_clone.to_search_filter();

                // Step 1: Search
                progress.set_message(format!("Searching for {}", burst_clone));
                let datasets = client_clone.search(&filter).await?;
                progress.inc(1);

                if datasets.is_empty() {
                    return Err(BurstDownloadError::ProductNotFound(
                        burst_clone.raw_id.clone(),
                    ));
                }

                // Create directory for this burst
                let burst_dir = output_dir_clone.join(format!("{}", burst_clone));
                fs::create_dir_all(&burst_dir)?;

                // Choose the first dataset
                let dataset = &datasets[0];

                // Step 2: Download
                progress.set_message(format!("Downloading product for {}", burst_clone));
                let zip_path = client_clone
                    .download_dataset(dataset, &temp_dir_clone, false)
                    .await?;
                progress.inc(1);

                // Step 3: Extract burst
                progress.set_message(format!("Extracting burst {}", burst_clone));

                // This part extracts the burst data from the downloaded zip
                let burst_data_path = if use_gdal {
                    // GDAL method would be implemented here
                    // For now, use the simple extraction method
                    extract_burst_simple(&zip_path, &burst_clone, &burst_dir).await?
                } else {
                    extract_burst_simple(&zip_path, &burst_clone, &burst_dir).await?
                };

                progress.inc(1);
                progress.finish_with_message(format!("Completed {}", burst_clone));

                Ok::<PathBuf, BurstDownloadError>(burst_data_path)
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            match task.await {
                Ok(result) => match result {
                    Ok(path) => {
                        results.push(path);
                        total_progress.inc(1);
                    }
                    Err(e) => {
                        total_progress.inc(1);
                        return Err(e);
                    }
                },
                Err(e) => {
                    total_progress.inc(1);
                    return Err(BurstDownloadError::CommandError(format!(
                        "Task failed: {}",
                        e
                    )));
                }
            }
        }

        total_progress.finish_with_message(format!("Downloaded {} bursts", results.len()));

        Ok(results)
    }

    /// Extract burst data from a Sentinel-1 SLC zip file
    async fn extract_burst_from_zip(
        &self,
        zip_path: &Path,
        burst: &BurstIdentifier,
        output_dir: &Path,
    ) -> Result<PathBuf, BurstDownloadError> {
        if self.use_gdal {
            // GDAL implementation would go here
            // For now, fall back to simple extraction
            extract_burst_simple(zip_path, burst, output_dir).await
        } else {
            extract_burst_simple(zip_path, burst, output_dir).await
        }
    }
}

/// Simple burst extraction from Sentinel-1 SLC zip file
///
/// This is a simplified version that extracts the burst data. In a production environment,
/// you would want a more sophisticated approach using libraries like SNAP or GDAL.
///
/// Source: Based on PyGMTSAR approach at https://github.com/AlexeyPechnikov/pygmtsar/
async fn extract_burst_simple(
    zip_path: &Path,
    burst: &BurstIdentifier,
    output_dir: &Path,
) -> Result<PathBuf, BurstDownloadError> {
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;

    // Determine subswath number (IW1, IW2, IW3)
    let subswath_num = burst.subswath_number()?;

    // Open zip file
    let zip_file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(zip_file)
        .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?;

    // Find annotation XML file for the subswath
    let annotation_pattern = format!("annotation/s1.-iw{}-slc", subswath_num);
    let mut annotation_path = None;

    for i in 0..archive.len() {
        let file_path = archive
            .by_index(i)
            .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?
            .name()
            .to_string();

        if file_path.contains(&annotation_pattern)
            && file_path.contains(&burst.polarization.to_lowercase())
        {
            annotation_path = Some(file_path);
            break;
        }
    }

    let annotation_path = annotation_path.ok_or_else(|| {
        BurstDownloadError::BurstNotFound(format!("No annotation XML found for {}", burst))
    })?;

    // Extract annotation XML
    let annotation_output_path = output_dir.join("annotation.xml");
    {
        let mut annotation_file = archive
            .by_name(&annotation_path)
            .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?;
        let mut output_file = File::create(&annotation_output_path)?;
        io::copy(&mut annotation_file, &mut output_file)?;
    }

    // Parse XML to find burst information
    let xml_content = fs::read_to_string(&annotation_output_path)?;

    // Simple parsing to find burst ID and byte position
    // This is a simplified approach. In a real implementation, you would use proper XML parsing
    let swath_tag = format!("<swath>IW{}</swath>", subswath_num);
    if !xml_content.contains(&swath_tag) {
        return Err(BurstDownloadError::BurstNotFound(format!(
            "Subswath {} not found in XML",
            subswath_num
        )));
    }

    // Find burst data based on burst ID
    // This is a simplified approach that works for Sentinel-1 annotation XMLs
    let mut burst_data_found = false;
    let mut burst_start = 0u64;
    let mut burst_end = 0u64;
    let mut lines = 0u32;
    let mut samples = 0u32;

    // Look for the burst ID tag or equivalent identifier
    let burst_id_pattern = format!("<burstId>{}</burstId>", burst.burst_id);
    if xml_content.contains(&burst_id_pattern) {
        burst_data_found = true;

        // Find sample and line information
        if let Some(samples_pos) = xml_content.find("<samplesPerBurst>") {
            if let Some(samples_end) = xml_content[samples_pos..].find("</samplesPerBurst>") {
                let samples_str = &xml_content[samples_pos + 17..samples_pos + samples_end];
                samples = samples_str.parse().unwrap_or(0);
            }
        }

        if let Some(lines_pos) = xml_content.find("<linesPerBurst>") {
            if let Some(lines_end) = xml_content[lines_pos..].find("</linesPerBurst>") {
                let lines_str = &xml_content[lines_pos + 15..lines_pos + lines_end];
                lines = lines_str.parse().unwrap_or(0);
            }
        }

        // Find byte position information
        if let Some(start_pos) = xml_content.find("<byteOffset>") {
            if let Some(start_end) = xml_content[start_pos..].find("</byteOffset>") {
                let start_str = &xml_content[start_pos + 12..start_pos + start_end];
                burst_start = start_str.parse().unwrap_or(0);
            }
        }

        // Calculate end position
        let complex_sample_size = 4; // Each complex sample is 4 bytes (2 bytes real + 2 bytes imaginary)
        burst_end = burst_start + (lines as u64 * samples as u64 * complex_sample_size as u64);
    }

    if !burst_data_found || lines == 0 || samples == 0 {
        return Err(BurstDownloadError::BurstNotFound(format!(
            "Burst ID {} not found in XML or invalid metadata",
            burst.burst_id
        )));
    }

    // Find measurement data file for the subswath
    let measurement_pattern = format!("measurement/s1.-iw{}-slc", subswath_num);
    let mut measurement_path = None;

    for i in 0..archive.len() {
        let file_path = archive
            .by_index(i)
            .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?
            .name()
            .to_string();

        if file_path.contains(&measurement_pattern)
            && file_path.contains(&burst.polarization.to_lowercase())
        {
            measurement_path = Some(file_path);
            break;
        }
    }

    let measurement_path = measurement_path.ok_or_else(|| {
        BurstDownloadError::BurstNotFound(format!("No measurement data found for {}", burst))
    })?;

    // Extract burst data from measurement file
    let burst_data_path = output_dir.join(format!("burst_data_{}x{}.raw", samples, lines));
    {
        let mut measurement_file = archive
            .by_name_seek(&measurement_path)
            .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?;

        // Seek to burst start position
        measurement_file
            .seek(std::io::SeekFrom::Start(burst_start))
            .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?;

        // Create output file
        let mut output_file = File::create(&burst_data_path)?;

        // Calculate how many bytes to read
        let bytes_to_read = burst_end - burst_start;

        // Read burst data to output file
        let mut buffer = vec![0u8; std::cmp::min(bytes_to_read, 1024 * 1024) as usize]; // 1MB buffer or smaller
        let mut remaining = bytes_to_read;

        while remaining > 0 {
            let buf_size = std::cmp::min(remaining, buffer.len() as u64) as usize;
            let bytes_read = measurement_file
                .read(&mut buffer[0..buf_size])
                .map_err(|e| BurstDownloadError::ExtractionError(e.to_string()))?;

            if bytes_read == 0 {
                break; // End of file
            }

            output_file.write_all(&buffer[0..bytes_read])?;
            remaining -= bytes_read as u64;
        }
    }

    // Create metadata file
    let metadata_path = output_dir.join("burst_metadata.json");
    let metadata = serde_json::json!({
        "burst_id": burst.burst_id,
        "subswath": burst.subswath,
        "polarization": burst.polarization,
        "datetime": burst.datetime.to_rfc3339(),
        "orbit": burst.orbit,
        "lines": lines,
        "samples": samples,
        "data_path": burst_data_path.file_name().unwrap().to_string_lossy(),
        "format": "complex16",  // 16-bit complex (2 bytes real + 2 bytes imaginary)
        "byte_order": "little_endian"
    });

    let mut metadata_file = File::create(&metadata_path)?;
    let metadata_str = serde_json::to_string_pretty(&metadata).map_err(|e| {
        BurstDownloadError::ExtractionError(format!("Failed to serialize metadata: {}", e))
    })?;
    metadata_file.write_all(metadata_str.as_bytes())?;

    Ok(burst_data_path)
}

/// Default implementation for AsfSearchFilter
impl Default for AsfSearchFilter {
    fn default() -> Self {
        Self {
            platform: None,
            instrument: None,
            start_date: None,
            end_date: None,
            processing_level: None,
            beam_mode: None,
            polarization: None,
            min_latitude: None,
            max_latitude: None,
            min_longitude: None,
            max_longitude: None,
            polygon: None,
            intersects_point: None,
            max_results: None,
            product_type: None,
            relativeorbit: None,
            frame: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_burst_identifier() {
        let burst_str = "S1_305967_IW3_20151022T122546_VV_5A48-BURST";
        let burst = BurstIdentifier::parse(burst_str).unwrap();

        assert_eq!(burst.satellite, "S1");
        assert_eq!(burst.orbit, "305967");
        assert_eq!(burst.subswath, "IW3");
        assert_eq!(burst.polarization, "VV");
        assert_eq!(burst.burst_id, "5A48");
        assert_eq!(burst.raw_id, burst_str);

        // Format back to string
        let formatted = format!("{}", burst);
        assert_eq!(formatted, burst_str);
    }

    #[test]
    fn test_parse_burst_list() {
        let burst_list = r#"
        S1_305967_IW3_20151022T122546_VV_5A48-BURST
        S1_305967_IW3_20151010T122546_VV_7501-BURST
        S1_305967_IW3_20150928T122546_VV_5407-BURST
        "#;

        let bursts = BurstDownloader::parse_burst_list(burst_list).unwrap();
        assert_eq!(bursts.len(), 3);

        assert_eq!(bursts[0].orbit, "305967");
        assert_eq!(bursts[0].burst_id, "5A48");

        assert_eq!(bursts[1].orbit, "305967");
        assert_eq!(bursts[1].burst_id, "7501");

        assert_eq!(bursts[2].orbit, "305967");
        assert_eq!(bursts[2].burst_id, "5407");
    }
}
