//! External data used by the ignored tests, fetched through the dataset caches: downloaded on
//! the first run (which needs the credentials of each service) and read from the user cache
//! directory afterwards (see [`cache_root`](crate::datasets::cache_root)).

use crate::{
    datasets::{
        asf::burst_download::{AsfBurstDownloader, BurstRequest, Polarization},
        opentopography::dem_download::{CopernicusDemType, OpenTopographyDemDownloader},
    },
    dem::DEM,
    granule_id::IWSwath,
    interferometry::bounding_box_from_stack,
    sentinel::Sentinel1SlcIWSwath,
};

/// The 2015 Sentinel-1A stack of burst `143_305967_IW3` (track 143, IW3/VV, near Mexico City):
/// for each acquisition date, the SLC granule and the index of the burst in its IW3 subswath,
/// as found with the ASF Search API.
pub(crate) const STACK_143_305967_IW3: [(&str, &str, u32); 7] = [
    (
        "20150916",
        "S1A_IW_SLC__1SSV_20150916T122538_20150916T122603_007740_00AC19_8302",
        2,
    ),
    (
        "20150928",
        "S1A_IW_SLC__1SSV_20150928T122539_20150928T122606_007915_00B0D8_5407",
        2,
    ),
    (
        "20151010",
        "S1A_IW_SLC__1SSV_20151010T122539_20151010T122603_008090_00B578_7501",
        2,
    ),
    (
        "20151022",
        "S1A_IW_SLC__1SSV_20151022T122539_20151022T122606_008265_00BA51_5A48",
        2,
    ),
    (
        "20151103",
        "S1A_IW_SLC__1SSV_20151103T122539_20151103T122603_008440_00BEE0_AE93",
        2,
    ),
    (
        "20151115",
        "S1A_IW_SLC__1SSV_20151115T122533_20151115T122600_008615_00C3B4_8956",
        4,
    ),
    (
        "20151127",
        "S1A_IW_SLC__1SSV_20151127T122533_20151127T122557_008790_00C894_14CF",
        4,
    ),
];

/// Loads the single-burst SAFE product of `request`, downloaded from ASF on the first run
/// (requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD) and read from the cache afterwards.
pub(crate) fn load_burst(request: &BurstRequest) -> Sentinel1SlcIWSwath {
    let safe_dir = AsfBurstDownloader::builder()
        .build()
        .fetch_burst(request)
        .unwrap_or_else(|err| panic!("Could not fetch {request:?}: {err:?}"));
    Sentinel1SlcIWSwath::load_swath_from_directory(request.subswath, &safe_dir)
        .unwrap_or_else(|err| panic!("Could not load {}: {err}", safe_dir.display()))
}

/// Burst `143_305967_IW3` acquired on `date` (`YYYYMMDD`, one of [`STACK_143_305967_IW3`]),
/// loaded with [`load_burst`].
pub(crate) fn burst_143_305967_iw3(date: &str) -> Sentinel1SlcIWSwath {
    let (_, granule, burst_index) = STACK_143_305967_IW3
        .iter()
        .find(|(stack_date, _, _)| *stack_date == date)
        .unwrap_or_else(|| panic!("No acquisition of burst 143_305967_IW3 on {date}"));
    load_burst(&BurstRequest::new(
        *granule,
        IWSwath::IW3,
        Polarization::VV,
        *burst_index,
    ))
}

/// The Copernicus DEM covering `bounds` (`[min lat, max lat, min lon, max lon]`), downloaded
/// from OpenTopography on the first run (requires OPENTOPOGRAPHY_API_KEY) and read from the
/// cache afterwards.
pub(crate) fn fetch_dem(bounds: [f64; 4], dem_type: CopernicusDemType) -> DEM {
    OpenTopographyDemDownloader::builder()
        .build()
        .fetch_dem(bounds, dem_type)
        .unwrap_or_else(|err| panic!("Could not fetch the {dem_type} DEM for {bounds:?}: {err}"))
}

/// The Copernicus DEM covering all the `swaths`, with a margin (see
/// [`bounding_box_from_stack`]), fetched with [`fetch_dem`].
pub(crate) fn dem_covering<'a>(
    swaths: impl IntoIterator<Item = &'a Sentinel1SlcIWSwath>,
    dem_type: CopernicusDemType,
) -> DEM {
    fetch_dem(bounding_box_from_stack(swaths), dem_type)
}
