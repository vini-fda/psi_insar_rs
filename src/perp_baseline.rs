//! Perpendicular Baseline (Repeat-pass interferometry)
//!
//! The baseline vector `B` is the difference between the satellite positions (`S_1` and `S_2`)
//! for the two acquisitions:
//!
//! `B = S_2 - S_1`
//!
//! The "perpendicular baseline" (denoted `B⊥`) refers to the magnitude of the component of this
//! baseline vector `B` that is perpendicular to the look direction vector `L` (line-of-sight
//! from the satellite to a point on the ground).
//!
//! This calculation involves decomposing the baseline vector `B` into two orthogonal components:
//! one parallel to the look direction `L`, and one perpendicular to `L`. The magnitude of the
//! perpendicular component is `B⊥`.
//!
//! To calculate `B⊥`:
//!
//! 1.  Let `L̂` be the unit vector in the look direction. This is obtained by normalizing the
//!     look vector `L`:
//!     `L̂ = L / ||L||`
//!     where `||L||` is the norm (magnitude) of `L`.
//!
//! 2.  The scalar projection of the baseline vector `B` onto the unit look vector `L̂` gives
//!     the length of the component of `B` that is parallel to `L`. Let this be `B_parallel_scalar`:
//!     `B_parallel_scalar = B ⋅ L̂`
//!     where `(B ⋅ L̂)` is the dot product of `B` and `L̂`.
//!
//! 3.  The baseline vector `B`, its component parallel to `L` (with magnitude `|B_parallel_scalar|`),
//!     and its component perpendicular to `L` (with magnitude `B⊥`) form a right-angled triangle.
//!     By the Pythagorean theorem:
//!     `||B||² = (B_parallel_scalar)² + (B⊥)²`
//!     Substituting `B_parallel_scalar = B ⋅ L̂`, we get:
//!     `||B||² = (B ⋅ L̂)² + (B⊥)²`
//!
//! 4.  Solving for `B⊥` yields the formula for the magnitude of the perpendicular baseline:
//!
//! Algebraically, the perpendicular baseline is given by:
//!
//! B⊥ = sqrt(||B||² - (B ⋅ L̂)²)
//!
//! where:
//!   - `||B||` is the norm (magnitude) of the baseline vector `B`. Consequently, `||B||²`
//!     is its squared magnitude.
//!   - `L̂` (L-hat) is the unit vector in the look direction (i.e., `L / ||L||`).
//!   - `(B ⋅ L̂)` is the dot product of the baseline vector `B` and the unit look vector `L̂`.
//!     The term `(B ⋅ L̂)²` is the squared magnitude of the component of `B` parallel to `L`.
//!
//! Key considerations for implementation:
//!   - `S_1` and `S_2` are the 3D position vectors of the satellite sensor at the times of the
//!     two acquisitions.
//!   - `L` is the look vector, typically from an average satellite position to the target area
//!     (e.g., scene center) on the ground.
//!   - All vectors (`S_1`, `S_2`, `B`, `L`) must be defined in a consistent 3D Cartesian
//!     coordinate system (e.g., Earth-Centered, Earth-Fixed - ECEF).
//!   - `B⊥` is a fundamental parameter in interferometric processing, directly impacting
//!     the sensitivity to topography/deformation and the degree of spatial decorrelation.

// # --- How inputs would typically be obtained (conceptual) ---
// # master_orbit_data = load_master_orbit()
// # slave_orbit_data = load_slave_orbit()
// # master_image_metadata = load_master_metadata()

// # For a given (line, pixel, height_guess):
// #   master_time_at_line = master_image_metadata.line_to_azimuth_time(line)
// #   master_sat_pos_xyz = master_orbit_data.get_position_at_time(master_time_at_line)
// #
// #   ground_point_xyz = master_orbit_data.project_pixel_to_ground(line, pixel, height_guess, master_image_metadata)
// #
// #   # To find corresponding slave satellite position, one needs to determine the slave's
// #   # imaging time for that specific ground_point_xyz. This often involves an
// #   # iterative process or using slave image metadata if already co-registered.
// #   # Let's assume slave_sat_pos_xyz for the same ground_point_xyz is found.
// #   slave_time_for_ground_point = slave_orbit_data.get_time_for_ground_point(ground_point_xyz, slave_image_metadata)
// #   slave_sat_pos_xyz = slave_orbit_data.get_position_at_time(slave_time_for_ground_point)
// #
// #   # Then call the function:
// #   # baseline_info = calculate_baseline_components_at_point(
// #   #     master_sat_pos_xyz,
// #   #     slave_sat_pos_xyz,
// #   #     ground_point_xyz
// #   # )
// #   # bperp_value = baseline_info['B_perpendicular_signed']

use nalgebra::Vector3;
use rustfft::num_traits::Zero;
use spade::{DelaunayTriangulation, HasPosition, Triangulation};

use crate::{
    constants::SENTINEL_1_WAVELENGTH,
    dem::DEM,
    geodesy::{geodetic_to_ecef, local_normal},
    metadata::annotation_xml::GeolocationGrid,
    satellite_orbit::{RadarCoords, pixel_coords_to_radar_coords},
    sentinel::Sentinel1SlcBurst,
};

/// Calculate the perpendicular baseline between two satellite positions, using a ground point.
///
/// # Arguments
///
/// * `s_1` - The position of the reference satellite.
/// * `s_2` - The position of the secondary satellite.
/// * `p` - The position of the ground point.
///
/// # Returns
///
/// The perpendicular baseline between the two satellites, at the given ground point.
///
/// # Example
///
/// ```ignore
/// use nalgebra::Vector3;
/// use perp_baseline::perp_baseline;
///
/// let s_1 = Vector3::new(1.0, 2.0, 3.0);
/// let s_2 = Vector3::new(4.0, 5.0, 6.0);
/// let p = Vector3::new(7.0, 8.0, 9.0);
///
/// let bperp = perp_baseline(&s_1, &s_2, &p);
///
/// assert_eq!(bperp, 10.0);
/// ```
///
/// # Panics
///
/// Panics if the ground point is the same as the reference satellite position.
///
pub fn perp_baseline(s_1: &Vector3<f64>, s_2: &Vector3<f64>, p: &Vector3<f64>) -> f64 {
    let b = s_2 - s_1;
    let l = (p - s_1).normalize();
    let b_parallel = b.dot(&l);
    let b_perp = (b.norm() - b_parallel).abs();
    b_perp
}

pub fn perp_baseline_from_pixel_index(
    primary_burst: &Sentinel1SlcBurst,
    secondary_burst: &Sentinel1SlcBurst,
    azimuth_index: f64,
    slant_range_index: f64,
    dem: &DEM,
) -> f64 {
    let osh_primary = primary_burst.orbital_state_history();
    let annotation_1 = &primary_burst.metadata;
    let radar_coords_1 =
        pixel_coords_to_radar_coords(azimuth_index, slant_range_index, annotation_1);
    let ground_target_pos = osh_primary.find_ground_target(radar_coords_1, dem);
    let (s_1, _) = osh_primary.interp_pos_vel(radar_coords_1.time);
    // find corresponding ground target on secondary acquisition
    let osh_secondary = secondary_burst.orbital_state_history();
    let radar_coords_2 = osh_secondary.find_zero_doppler_state(ground_target_pos);
    let (s_2, _) = osh_secondary.interp_pos_vel(radar_coords_2.time);
    perp_baseline(&s_1, &s_2, &ground_target_pos)
}

/// Calculates the incidence angle theta
pub fn theta_from_pixel_index(
    primary_burst: &Sentinel1SlcBurst,
    azimuth_index: f64,
    slant_range_index: f64,
    dem: &DEM,
) -> f64 {
    let osh_primary = primary_burst.orbital_state_history();
    let annotation_1 = &primary_burst.metadata;
    let radar_coords = pixel_coords_to_radar_coords(azimuth_index, slant_range_index, annotation_1);
    let mut current_min = std::f64::MAX;
    let mut ground_target_pos_geodetic = (0.0, 0.0);
    let mut ground_target_pos = Vector3::<f64>::zero();
    let (sat_pos, sat_vel) = &osh_primary.interp_pos_vel(radar_coords.time);
    let sat_vel_hat = sat_vel.normalize();
    for (_, _, lat, lon, height) in dem.indexed_lat_lon_height() {
        ground_target_pos_geodetic = (lat, lon);
        let ground_pos = Vector3::from(geodetic_to_ecef(lat, lon, height));
        let val = (ground_pos - sat_pos).dot(&sat_vel_hat).abs();
        if val < current_min {
            current_min = val;
            ground_target_pos = ground_pos;
        }
    }
    let (s_1, _) = osh_primary.interp_pos_vel(radar_coords.time);
    let l = (s_1 - ground_target_pos).normalize();
    let normal = Vector3::from(local_normal(
        ground_target_pos_geodetic.0,
        ground_target_pos_geodetic.1,
    ));
    let theta = l.dot(&normal).acos();
    theta
}

#[derive(Clone, Copy, Debug)]
pub struct ExactMapping {
    pub azimuth_index: f64,
    pub slant_range_index: f64,
    pub lat: f64,
    pub lon: f64,
    pub height: f64,
}

impl HasPosition for ExactMapping {
    type Scalar = f64;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        [self.azimuth_index, self.slant_range_index].into()
    }
}

pub struct GeolocationGridInterpolator {
    triangulation: DelaunayTriangulation<ExactMapping>,
}

impl GeolocationGridInterpolator {
    pub fn new(grid: &GeolocationGrid) -> Self {
        let points = &grid.geolocation_grid_point_list.geolocation_grid_point;
        let mut triangulation = DelaunayTriangulation::<ExactMapping>::new();
        for p in points {
            let mapping = ExactMapping {
                azimuth_index: p.line as f64,
                slant_range_index: p.pixel as f64,
                lat: p.latitude,
                lon: p.longitude,
                height: p.height,
            };
            triangulation
                .insert(mapping)
                .expect("Failed to insert mapping");
        }
        Self { triangulation }
    }

    pub fn interpolate(&self, azimuth_index: f64, slant_range_index: f64) -> [Option<f64>; 3] {
        let nn = self.triangulation.natural_neighbor();

        let interp_lat =
            nn.interpolate(|v| v.data().lat, [azimuth_index, slant_range_index].into());
        let interp_lon =
            nn.interpolate(|v| v.data().lon, [azimuth_index, slant_range_index].into());
        let interp_height = nn.interpolate(
            |v| v.data().height,
            [azimuth_index, slant_range_index].into(),
        );
        [interp_lat, interp_lon, interp_height]
    }
}

pub fn flat_earth_dphi_efficient(
    primary_burst: &Sentinel1SlcBurst,
    secondary_burst: &Sentinel1SlcBurst,
    azimuth_index: f64,
    slant_range_index: f64,
    dem: &DEM,
) -> f64 {
    let osh_primary = primary_burst.orbital_state_history();
    let annotation_1 = &primary_burst.metadata;
    let radar_coords_1 =
        pixel_coords_to_radar_coords(azimuth_index, slant_range_index, annotation_1);
    let mut current_min = std::f64::MAX;
    let mut ground_target_lat = 0.0;
    let mut ground_target_lon = 0.0;
    let mut ground_target_pos = Vector3::<f64>::zero();
    let (sat_pos, sat_vel) = &osh_primary.interp_pos_vel(radar_coords_1.time);
    let sat_vel_hat = sat_vel.normalize();
    // 2D root finding to find the ground target position
    for (_, _, lat, lon, height) in dem.indexed_lat_lon_height() {
        ground_target_lat = lat;
        ground_target_lon = lon;
        let ground_pos = Vector3::from(geodetic_to_ecef(lat, lon, height));
        let val = (ground_pos - sat_pos).dot(&sat_vel_hat).abs();
        if val < current_min {
            current_min = val;
            ground_target_pos = ground_pos;
        }
    }
    println!("min_dot = {}", current_min);
    let (s_1, _) = osh_primary.interp_pos_vel(radar_coords_1.time);
    // find corresponding ground target on secondary acquisition
    let osh_secondary = secondary_burst.orbital_state_history();
    let radar_coords_2 = osh_secondary.find_zero_doppler_state(ground_target_pos);
    let (s_2, _) = osh_secondary.interp_pos_vel(radar_coords_2.time);
    let bperp = perp_baseline(&s_1, &s_2, &ground_target_pos);
    let r = (s_1 - ground_target_pos).norm();
    let l = (s_1 - ground_target_pos).normalize();
    let normal = Vector3::from(local_normal(ground_target_lat, ground_target_lon));
    let theta = l.dot(&normal).acos();
    let s = annotation_1
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let dphi = (4.0 * std::f64::consts::PI * bperp * s) / (r * SENTINEL_1_WAVELENGTH * theta.tan());
    dphi
}

pub fn flat_earth_dphi(
    primary_burst: &Sentinel1SlcBurst,
    secondary_burst: &Sentinel1SlcBurst,
    azimuth_index: f64,
    slant_range_index: f64,
    dem: &DEM,
) -> f64 {
    let osh_primary = primary_burst.orbital_state_history();
    let annotation_1 = &primary_burst.metadata;
    let radar_coords_1 =
        pixel_coords_to_radar_coords(azimuth_index, slant_range_index, annotation_1);
    let mut current_min = std::f64::MAX;
    let mut ground_target_lat = 0.0;
    let mut ground_target_lon = 0.0;
    let mut ground_target_pos = Vector3::<f64>::zero();
    let (sat_pos, sat_vel) = &osh_primary.interp_pos_vel(radar_coords_1.time);
    let sat_vel_hat = sat_vel.normalize();
    // 2D root finding to find the ground target position
    for (_, _, lat, lon, height) in dem.indexed_lat_lon_height() {
        ground_target_lat = lat;
        ground_target_lon = lon;
        let ground_pos = Vector3::from(geodetic_to_ecef(lat, lon, height));
        let val = (ground_pos - sat_pos).dot(&sat_vel_hat).abs();
        if val < current_min {
            current_min = val;
            ground_target_pos = ground_pos;
        }
    }
    println!("min_dot = {}", current_min);
    let (s_1, _) = osh_primary.interp_pos_vel(radar_coords_1.time);
    // find corresponding ground target on secondary acquisition
    let osh_secondary = secondary_burst.orbital_state_history();
    let radar_coords_2 = osh_secondary.find_zero_doppler_state(ground_target_pos);
    let (s_2, _) = osh_secondary.interp_pos_vel(radar_coords_2.time);
    let bperp = perp_baseline(&s_1, &s_2, &ground_target_pos);
    let r = (s_1 - ground_target_pos).norm();
    let l = (s_1 - ground_target_pos).normalize();
    let normal = Vector3::from(local_normal(ground_target_lat, ground_target_lon));
    let theta = l.dot(&normal).acos();
    let s = annotation_1
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let dphi = (4.0 * std::f64::consts::PI * bperp * s) / (r * SENTINEL_1_WAVELENGTH * theta.tan());
    dphi
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rerun::Image;

    use super::*;

    #[test]
    fn test_perp_baseline_from_pixel_index() {
        let primary = Sentinel1SlcBurst::load_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
            "S1A_IW_SLC__1SVV_20151022T122546_20151022T122549_008265_00BA51_422D",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
            "S1A_IW_SLC__1SVV_20151010T122546_20151010T122550_008090_00B578_BFAD",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let [range_size, az_size] = primary.data.raster_size();
        for i in 0..az_size {
            for j in 0..1 {
                let theta = theta_from_pixel_index(&primary, i as f64, j as f64, &dem);
                println!(
                    "azimuth_index: {}, slant_range_index: {}, theta: {}",
                    i,
                    j,
                    theta.to_degrees()
                );
            }
        }

        // let data = Array2::<f64>::from_shape_fn((az_size, 1), |(az_index, _)| {
        //     perp_baseline_from_pixel_index(&primary, &secondary, az_index as f64, 100.0, &dem)
        // });
        // let rr = rerun::RecordingStreamBuilder::new("test_perp_baseline_from_pixel_index")
        //     .connect_grpc()
        //     .expect("Could not connect to local Rerun instance.");
        // let img = Image::from_color_model_and_tensor(rerun::ColorModel::L, data).unwrap();

        // rr.log_static("bperp", &img).unwrap();
    }

    #[test]
    fn test_flat_earth_dphi() {
        let primary = Sentinel1SlcBurst::load_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
            "S1A_IW_SLC__1SVV_20151022T122546_20151022T122549_008265_00BA51_422D",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
            "S1A_IW_SLC__1SVV_20151010T122546_20151010T122550_008090_00B578_BFAD",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let [range_size, az_size] = primary.data.raster_size();
        for i in 0..az_size {
            for j in 0..1 {
                let dphi = flat_earth_dphi(&primary, &secondary, i as f64, j as f64, &dem);
                println!(
                    "azimuth_index: {}, slant_range_index: {}, dphi: {}",
                    i,
                    j,
                    dphi.to_degrees()
                );
            }
        }
    }
}
