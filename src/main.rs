use std::error::Error;
use std::path::PathBuf;

use psi_insar_rs::asf_api_client::AsfApiClient;
use psi_insar_rs::burst_downloader::{BurstDownloader, BurstIdentifier};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Set up ASF credentials (use environment variables in production)
    let username = "GoogleColab2023";
    let password = "GoogleColab_2023";

    // Create ASF API client
    let client = AsfApiClient::new()?.with_credentials(username, password);

    // Define output directory
    let output_dir = PathBuf::from("data/mexico_city_bursts");

    // Create burst downloader
    let downloader = BurstDownloader::new(client, output_dir)?;

    // Define burst list - Mexico City bursts
    let bursts_list = r#"
    S1_305967_IW3_20151022T122546_VV_5A48-BURST
    S1_305967_IW3_20151010T122546_VV_7501-BURST
    S1_305967_IW3_20150928T122546_VV_5407-BURST
    "#;

    // Parse burst identifiers
    let bursts = BurstDownloader::parse_burst_list(bursts_list)?;
    println!("Parsed {} burst identifiers", bursts.len());

    for burst in &bursts {
        println!("Burst: {}", burst);
    }

    // Download the bursts
    println!("Starting download of {} bursts", bursts.len());
    let burst_paths = downloader.download_bursts(&bursts).await?;

    // Print results
    println!(
        "Download complete. Downloaded {} bursts:",
        burst_paths.len()
    );
    for path in &burst_paths {
        println!("  - {:?}", path);
    }

    // Generate a simple report
    generate_report(&bursts, &burst_paths)?;

    Ok(())
}

fn generate_report(bursts: &[BurstIdentifier], paths: &[PathBuf]) -> Result<(), Box<dyn Error>> {
    println!("\nMexico City Bursts Report:");
    println!("===========================");

    for (i, (burst, path)) in bursts.iter().zip(paths.iter()).enumerate() {
        println!("Burst #{}", i + 1);
        println!("  ID:          {}", burst);
        println!("  Orbit:       {}", burst.orbit);
        println!("  Subswath:    {}", burst.subswath);
        println!("  Date:        {}", burst.datetime);
        println!("  Polarization: {}", burst.polarization);
        println!("  Saved to:    {:?}", path);
        println!();
    }

    println!("Total: {} bursts", bursts.len());
    println!("===========================");

    Ok(())
}
