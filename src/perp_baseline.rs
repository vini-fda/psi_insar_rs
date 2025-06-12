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

use nalgebra::{Complex, Vector3};
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use rustfft::num_traits::Zero;
use spade::{DelaunayTriangulation, HasPosition, Triangulation};
use std::iter::Map;

use crate::{
    constants::SENTINEL_1_WAVELENGTH,
    coregistration::{
        deramping::DerampSlcBurst,
        interpolation2d::{KnabSincKernel, interpolate_2d},
        warp_function::WarpFunction,
    },
    dem::DEM,
    geodesy::{geodetic_to_ecef, local_normal},
    metadata::annotation_xml::{GeolocationGrid, SlcProductAnnotation},
    satellite_orbit::{
        OrbitalStateHistory, RadarCoords, pixel_coords_to_radar_coords,
        radar_coords_to_pixel_coords,
    },
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
/// - B⊥ > 0: If secondary satellite is farther from the ground track than the reference satellite.
/// - B⊥ < 0: If secondary satellite is closer to the ground track than the reference satellite.
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
    let b_parallel = b.dot(&l) * l;
    let b_perp = (b - b_parallel).norm();
    if (s_1 - p).norm() < (s_2 - p).norm() {
        b_perp
    } else {
        -b_perp
    }
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

#[derive(Clone, Copy, Debug)]
pub struct FlatEarthComponents {
    pub azimuth_index: f64,
    pub slant_range_index: f64,
    pub theta: f64,
    pub bperp: f64,
    pub r: f64,
}

impl HasPosition for FlatEarthComponents {
    type Scalar = f64;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        [self.azimuth_index, self.slant_range_index].into()
    }
}

/// Interpolates `theta` and `bperp` for a given pixel index.
///
/// The underlying data is a sparse grid of points, which are interpolated to get the values at the
/// pixel index.
pub struct FlatEarthComponentsInterpolator {
    triangulation: DelaunayTriangulation<FlatEarthComponents>,
    azimuth_size: usize,
    slant_range_size: usize,
    slant_range_pixel_spacing: f64,
}

impl FlatEarthComponentsInterpolator {
    pub fn from_grid_params(
        primary_burst: &Sentinel1SlcBurst,
        secondary_burst: &Sentinel1SlcBurst,
        dem: &DEM,
        azimuth_samples: usize,
        slant_range_samples: usize,
    ) -> Self {
        let [slant_range_size, azimuth_size] = primary_burst.data.raster_size();
        let azimuth_spacing = azimuth_size as f64 / azimuth_samples as f64;
        let slant_range_spacing = slant_range_size as f64 / slant_range_samples as f64;
        let slant_range_pixel_spacing = primary_burst
            .metadata
            .image_annotation
            .image_information
            .range_pixel_spacing;
        let mut points = Vec::new();
        for i in 0..azimuth_samples {
            for j in 0..slant_range_samples {
                let azimuth_index = i as f64 * azimuth_spacing;
                let slant_range_index = j as f64 * slant_range_spacing;
                let [theta, bperp, r] = Self::compute_components_precisely(
                    primary_burst,
                    secondary_burst,
                    azimuth_index,
                    slant_range_index,
                    dem,
                );
                points.push(FlatEarthComponents {
                    azimuth_index,
                    slant_range_index,
                    theta,
                    bperp,
                    r,
                });
            }
        }
        Self::new(
            points,
            azimuth_size,
            slant_range_size,
            slant_range_pixel_spacing,
        )
    }

    pub fn new(
        points: impl IntoIterator<Item = impl Into<FlatEarthComponents>>,
        azimuth_size: usize,
        slant_range_size: usize,
        slant_range_pixel_spacing: f64,
    ) -> Self {
        let mut triangulation = DelaunayTriangulation::<FlatEarthComponents>::new();
        for p in points {
            triangulation
                .insert(p.into())
                .expect("Failed to insert mapping");
        }
        Self {
            triangulation,
            azimuth_size,
            slant_range_size,
            slant_range_pixel_spacing,
        }
    }

    /// Interpolates `theta`, `bperp` and `r` for a given pixel index.
    ///
    /// Note: this method is not recommended for consecutive interpolations. Instead, use
    /// `interpolate_many` with a batch of pixel indices.
    pub fn interpolate(&self, azimuth_index: f64, slant_range_index: f64) -> [Option<f64>; 3] {
        let nn = self.triangulation.natural_neighbor();

        let interp_theta = nn.interpolate(
            |v| v.data().theta,
            [azimuth_index, slant_range_index].into(),
        );
        let interp_bperp = nn.interpolate(
            |v| v.data().bperp,
            [azimuth_index, slant_range_index].into(),
        );
        let interp_r = nn.interpolate(|v| v.data().r, [azimuth_index, slant_range_index].into());
        [interp_theta, interp_bperp, interp_r]
    }

    /// Efficiently interpolates `theta` and `bperp` for a given set of pixel indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of pixel indices, where each index is an array of [azimuth index, slant range index].
    ///
    /// # Returns
    ///
    /// A vector of `[Option<f64>; 2]`, where each value contains the interpolated `theta` and `bperp` values,
    /// or `None` if the point is outside the convex hull of the triangulation.
    pub fn interpolate_many(
        &self,
        indices: impl IntoIterator<Item = [f64; 2]>,
    ) -> Vec<[Option<f64>; 3]> {
        let nn = self.triangulation.natural_neighbor();
        indices
            .into_iter()
            .map(|[azimuth_index, slant_range_index]| {
                let interp_theta = nn.interpolate(
                    |v| v.data().theta,
                    [azimuth_index, slant_range_index].into(),
                );
                let interp_bperp = nn.interpolate(
                    |v| v.data().bperp,
                    [azimuth_index, slant_range_index].into(),
                );
                let interp_r =
                    nn.interpolate(|v| v.data().r, [azimuth_index, slant_range_index].into());
                [interp_theta, interp_bperp, interp_r]
            })
            .collect()
    }

    /// Calculates the approximate flat earth phase array by accumulating the phase difference `dphi`
    /// between pixels that are adjacent in the slant range direction.
    pub fn calculate_array(&self) -> Array2<f64> {
        let azimuth_size = self.azimuth_size;
        let slant_range_size = self.slant_range_size;
        let s = self.slant_range_pixel_spacing;
        let mut array = Array2::<f64>::zeros((azimuth_size, slant_range_size));
        let nn = self.triangulation.natural_neighbor();
        for i in 0..azimuth_size {
            for j in 0..(slant_range_size - 1) {
                let i_f64 = i as f64;
                let j_f64 = j as f64;
                let theta = nn
                    .interpolate(|v| v.data().theta, [i_f64, j_f64].into())
                    .unwrap_or_default();
                let bperp = nn
                    .interpolate(|v| v.data().bperp, [i_f64, j_f64].into())
                    .unwrap_or_default();
                let r = nn
                    .interpolate(|v| v.data().r, [i_f64, j_f64].into())
                    .unwrap_or_default();

                let dphi = (4.0 * std::f64::consts::PI * bperp * s)
                    / (r * SENTINEL_1_WAVELENGTH * theta.tan());
                array[[i, j + 1]] = array[[i, j]] + dphi;
            }
        }
        array
    }

    /// Computes `theta`, `bperp` and `r` for a given pixel index.
    ///
    /// # Arguments
    ///
    /// - `primary_burst` - The primary burst.
    /// - `secondary_burst` - The secondary burst.
    /// - `azimuth_index` - The azimuth index.
    /// - `slant_range_index` - The slant range index.
    /// - `dem` - The digital elevation model.
    ///
    /// # Returns
    ///
    /// A vector of `[f64; 3]`, where each value contains the computed `theta`, `bperp` and `r` values.
    fn compute_components_precisely(
        primary_burst: &Sentinel1SlcBurst,
        secondary_burst: &Sentinel1SlcBurst,
        azimuth_index: f64,
        slant_range_index: f64,
        dem: &DEM,
    ) -> [f64; 3] {
        let osh_1 = primary_burst.orbital_state_history();
        let annotation_1 = &primary_burst.metadata;
        let radar_coords_1 =
            pixel_coords_to_radar_coords(azimuth_index, slant_range_index, annotation_1);
        let mut current_min = std::f64::MAX;
        let mut ground_target_lat = 0.0;
        let mut ground_target_lon = 0.0;
        let mut ground_target_pos = Vector3::<f64>::zero();
        let (sat_pos, sat_vel) = &osh_1.interp_pos_vel(radar_coords_1.time);
        let distance_to_target = radar_coords_1.distance_to_target;
        const MAX_DISTANCE_TO_TARGET_DIFF: f64 = 20.0;
        let sat_vel_hat = sat_vel.normalize();
        // 2D root finding to find the ground target position
        for (_, _, lat, lon, height) in dem.indexed_lat_lon_height() {
            let ground_pos = Vector3::from(geodetic_to_ecef(lat, lon, height));
            let val = (ground_pos - sat_pos).dot(&sat_vel_hat).abs();
            let r = (ground_pos - sat_pos).norm();
            let distance_to_target_diff = (r - distance_to_target).abs();
            if distance_to_target_diff > MAX_DISTANCE_TO_TARGET_DIFF {
                continue;
            }
            if val < current_min {
                current_min = val;
                ground_target_pos = ground_pos;
                ground_target_lat = lat;
                ground_target_lon = lon;
            }
        }

        let (s_1, _) = osh_1.interp_pos_vel(radar_coords_1.time);
        // find corresponding ground target on secondary acquisition
        let osh_2 = secondary_burst.orbital_state_history();
        let radar_coords_2 = osh_2.find_zero_doppler_state(ground_target_pos);
        let (s_2, _) = osh_2.interp_pos_vel(radar_coords_2.time);
        let bperp = perp_baseline(&s_1, &s_2, &ground_target_pos);
        let r = (s_1 - ground_target_pos).norm();
        let l = (s_1 - ground_target_pos).normalize();
        let normal = Vector3::from(local_normal(ground_target_lat, ground_target_lon));
        let theta = l.dot(&normal).acos();
        [theta, bperp, r]
    }
}

pub fn flat_earth_dphi(
    primary_burst: &Sentinel1SlcBurst,
    secondary_burst: &Sentinel1SlcBurst,
    azimuth_index: f64,
    slant_range_index: f64,
    dem: &DEM,
) -> f64 {
    let osh_1 = primary_burst.orbital_state_history();
    let annotation_1 = &primary_burst.metadata;
    let radar_coords_1 =
        pixel_coords_to_radar_coords(azimuth_index, slant_range_index, annotation_1);
    let mut current_min = std::f64::MAX;
    let mut ground_target_lat = 0.0;
    let mut ground_target_lon = 0.0;
    let mut ground_target_pos = Vector3::<f64>::zero();
    let (sat_pos, sat_vel) = &osh_1.interp_pos_vel(radar_coords_1.time);
    let distance_to_target = radar_coords_1.distance_to_target;
    const MAX_DISTANCE_TO_TARGET_DIFF: f64 = 20.0;
    let sat_vel_hat = sat_vel.normalize();
    // 2D root finding to find the ground target position
    for (_, _, lat, lon, height) in dem.indexed_lat_lon_height() {
        let ground_pos = Vector3::from(geodetic_to_ecef(lat, lon, height));
        let val = (ground_pos - sat_pos).dot(&sat_vel_hat).abs();
        let r = (ground_pos - sat_pos).norm();
        let distance_to_target_diff = (r - distance_to_target).abs();
        if distance_to_target_diff > MAX_DISTANCE_TO_TARGET_DIFF {
            continue;
        }
        if val < current_min {
            current_min = val;
            ground_target_pos = ground_pos;
            ground_target_lat = lat;
            ground_target_lon = lon;
        }
    }

    let (s_1, _) = osh_1.interp_pos_vel(radar_coords_1.time);
    // find corresponding ground target on secondary acquisition
    let osh_2 = secondary_burst.orbital_state_history();
    let radar_coords_2 = osh_2.find_zero_doppler_state(ground_target_pos);
    let (s_2, _) = osh_2.interp_pos_vel(radar_coords_2.time);
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
    println!(
        "azimuth_index: {}, slant_range_index: {}, dphi: {}, bperp: {}, theta: {}",
        azimuth_index,
        slant_range_index,
        dphi.to_degrees(),
        bperp,
        theta.to_degrees()
    );

    dphi
}

// -- Enchanced Delaunay Warp Function --

pub struct EnhancedDelaunayWarpFunction {
    pub triangulation: DelaunayTriangulation<WarpFunctionExactMapping>,
}

/// A point which contains a single exact mapping of the reference coordinates to the secondary coordinates.
#[derive(Debug, Clone, Copy)]
pub struct WarpFunctionExactMapping {
    pub reference_coords: [f64; 2],
    pub secondary_coords: [f64; 2],
    pub lat: f64,
    pub lon: f64,
}

impl HasPosition for WarpFunctionExactMapping {
    type Scalar = f64;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        self.reference_coords.into()
    }
}

impl EnhancedDelaunayWarpFunction {
    /// Computes the warp function \rho between two SLC images, in the domain of the reference image.
    pub fn new(reference: &Sentinel1SlcBurst, secondary: &Sentinel1SlcBurst, dem: &DEM) -> Self {
        let [azimuth_size, slant_range_size] = reference.data.raster_size();
        let ref_osh = reference.orbital_state_history();
        let sec_osh = secondary.orbital_state_history();
        let radar_coords = |ground_target_pos: Vector3<f64>,
                            osh: &OrbitalStateHistory,
                            annotation: &SlcProductAnnotation|
         -> [f64; 2] {
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);

            radar_coords_to_pixel_coords(zero_doppler, annotation)
        };
        let mut triangulation: DelaunayTriangulation<_> = DelaunayTriangulation::new();
        for (_, _, lat, lon, _) in dem.indexed_lat_lon_height() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let rc_ref = radar_coords(pos.into(), &ref_osh, &reference.metadata);
            let rc_sec = radar_coords(pos.into(), &sec_osh, &secondary.metadata);

            if (rc_ref[0] >= 0.0 && rc_ref[0] < slant_range_size as f64)
                && (rc_ref[1] >= 0.0 && rc_ref[1] < azimuth_size as f64)
            {
                let mapping = WarpFunctionExactMapping {
                    reference_coords: rc_ref,
                    secondary_coords: rc_sec,
                    lat,
                    lon,
                };
                triangulation
                    .insert(mapping)
                    .expect("Failed to insert mapping");
            }
        }

        Self { triangulation }
    }

    /// Maps multiple reference image coordinates to their corresponding coordinates in the secondary image.
    ///
    /// This method efficiently computes the mapped coordinates for multiple points by reusing the
    /// natural neighbor interpolation object. It's more efficient than calling `map()` multiple times
    /// as it avoids recreating the interpolation object for each point.
    ///
    /// # Arguments
    ///
    /// * `ref_coords` - An iterator of 2D arrays, each containing [azimuth, range] coordinates in the reference image
    ///
    /// # Returns
    ///
    /// An iterator over optional coordinates, where each element is:
    /// * `Some([azimuth, range])` - The coordinates in the secondary image that correspond to the
    ///   input reference coordinates
    /// * `None` - If the corresponding input coordinates are outside the convex hull of the triangulation
    pub fn map<'a, I>(
        &'a self,
        ref_coords: I,
    ) -> Map<I::IntoIter, impl FnMut(I::Item) -> Option<[f64; 2]> + 'a>
    where
        I: IntoIterator<Item = [f64; 2]>,
        I::IntoIter: 'a,
    {
        let nn = self.triangulation.natural_neighbor();
        ref_coords.into_iter().map(move |point| {
            let compute_mapped_coord = |dimension: usize| {
                nn.interpolate(|v| v.data().secondary_coords[dimension], point.into())
            };
            let mapped_azimuth = compute_mapped_coord(0)?;
            let mapped_range = compute_mapped_coord(1)?;

            Some([mapped_azimuth, mapped_range])
        })
    }

    // uses rayon IntoParallelIterator trait
    pub fn map_parallel<'a, I>(
        &'a self,
        ref_coords: I,
    ) -> rayon::iter::MapInit<
        I::Iter,
        impl Fn() -> spade::NaturalNeighbor<'a, DelaunayTriangulation<WarpFunctionExactMapping>>,
        impl Fn(
            &mut spade::NaturalNeighbor<'a, DelaunayTriangulation<WarpFunctionExactMapping>>,
            [f64; 2],
        ) -> Option<[f64; 2]>,
    >
    where
        I: IntoParallelIterator<Item = [f64; 2]>,
        I::Iter: 'a,
    {
        ref_coords.into_par_iter().map_init(
            || self.triangulation.natural_neighbor(),
            move |nn, point| {
                let compute_mapped_coord = |dimension: usize| {
                    nn.interpolate(|v| v.data().secondary_coords[dimension], point.into())
                };
                let mapped_azimuth = compute_mapped_coord(0)?;
                let mapped_range = compute_mapped_coord(1)?;

                Some([mapped_azimuth, mapped_range])
            },
        )
    }
}

pub fn coregister_and_remove_flat_phase(
    reference: &Sentinel1SlcBurst,
    secondary: &Sentinel1SlcBurst,
    dem: &DEM,
) -> Array2<f32> {
    log::info!("Computing warp function");
    let start_time = std::time::Instant::now();
    let warp_function = EnhancedDelaunayWarpFunction::new(reference, secondary, dem);
    let end_time = std::time::Instant::now();
    log::info!("Time taken: {:?}", end_time - start_time);

    let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();
    let mut coregistered_secondary_img = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));

    // The indices in the domain of the reference image
    let indices = (0..ref_slant_range_dim)
        .flat_map(|ref_rg| (0..ref_azimuth_dim).map(move |ref_az| [ref_az, ref_rg]));
    let indices_usize: Vec<[usize; 2]> = indices.clone().collect();

    let kernel = KnabSincKernel::default();
    let deramp = DerampSlcBurst::new();

    log::info!("Deramping reference and secondary images");
    let start_time = std::time::Instant::now();
    let reference_img = deramp.apply_forward(&reference);
    let secondary_img = deramp.apply_forward(&secondary);
    let end_time = std::time::Instant::now();
    log::info!("Time taken: {:?}", end_time - start_time);

    log::info!("Resampling secondary image to reference image via warp function");
    let start_time = std::time::Instant::now();
    coregistered_secondary_img
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each_init(
            || warp_function.triangulation.natural_neighbor(),
            |nn, (ref_az, mut row)| {
                for (ref_rg, value) in row.iter_mut().enumerate() {
                    let ref_coords = [ref_az as f64, ref_rg as f64];
                    let compute_mapped_coord = |dimension: usize| {
                        nn.interpolate(|v| v.data().secondary_coords[dimension], ref_coords.into())
                    };
                    if let (Some(sec_az), Some(sec_rg)) =
                        (compute_mapped_coord(0), compute_mapped_coord(1))
                    {
                        let v = interpolate_2d(
                            secondary_img.view(),
                            sec_az as f32,
                            sec_rg as f32,
                            &kernel,
                        );
                        *value = v;
                    }
                }
            },
        );
    let end_time = std::time::Instant::now();
    log::info!("Time taken: {:?}", end_time - start_time);

    let osh_1 = reference.orbital_state_history();
    let annotation_1 = &reference.metadata;
    let s = annotation_1
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let osh_2 = secondary.orbital_state_history();
    let annotation_2 = &secondary.metadata;
    let nn = warp_function.triangulation.natural_neighbor();
    let mut phase_diff = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));
    for i in 0..ref_azimuth_dim {
        for j in 0..ref_slant_range_dim {
            phase_diff[[i, j]] =
                reference_img[[i, j]].arg() - coregistered_secondary_img[[i, j]].arg();
        }
    }

    log::info!("Removing topographic phase");
    let start_time = std::time::Instant::now();
    for ref_az in 0..ref_azimuth_dim {
        let mut accumulated_dphi = 0.0;
        let mut current_height = None;
        for ref_rg in 0..ref_slant_range_dim {
            let ref_coords = [ref_az as f64, ref_rg as f64];
            // Helper function to compute mapped coordinate for a given dimension
            let compute_mapped_coord = |dimension: usize| {
                nn.interpolate(|v| v.data().secondary_coords[dimension], ref_coords.into())
            };
            let sec_az = compute_mapped_coord(0);
            let sec_rg = compute_mapped_coord(1);

            // Now compute the ground target position
            let ground_target_lat = nn.interpolate(|v| v.data().lat, ref_coords.into());
            let ground_target_lon = nn.interpolate(|v| v.data().lon, ref_coords.into());

            // if any are None, skip
            if sec_az.is_none()
                || sec_rg.is_none()
                || ground_target_lat.is_none()
                || ground_target_lon.is_none()
            {
                continue;
            }
            let sec_az = sec_az.unwrap();
            let sec_rg = sec_rg.unwrap();
            let ground_target_lat = ground_target_lat.unwrap();
            let ground_target_lon = ground_target_lon.unwrap();

            let ground_target_pos =
                Vector3::from(dem.get_ecef_at_lat_lon(ground_target_lat, ground_target_lon));

            let radar_coords_1 =
                pixel_coords_to_radar_coords(ref_az as f64, ref_rg as f64, annotation_1);
            let radar_coords_2 =
                pixel_coords_to_radar_coords(sec_az as f64, sec_rg as f64, annotation_2);
            let (s_1, _) = osh_1.interp_pos_vel(radar_coords_1.time);
            let (s_2, _) = osh_2.interp_pos_vel(radar_coords_2.time);

            let bperp = perp_baseline(&s_1, &s_2, &ground_target_pos);
            let l = (s_1 - ground_target_pos).normalize();
            let r = (s_1 - ground_target_pos).norm();
            let normal = Vector3::from(local_normal(ground_target_lat, ground_target_lon));
            let theta = l.dot(&normal).acos();

            phase_diff[[ref_az, ref_rg]] -= accumulated_dphi as f32;

            let height = dem.get_height_at_lat_lon(ground_target_lat, ground_target_lon);

            if let Some(current_height) = current_height {
                let height_diff = height - current_height;
                let height_diff_dphi = (4.0 * std::f64::consts::PI * bperp * height_diff)
                    / (r * SENTINEL_1_WAVELENGTH * theta.sin());
                accumulated_dphi -= height_diff_dphi;
            }

            current_height = Some(height);

            let dphi = (4.0 * std::f64::consts::PI * bperp * s)
                / (r * SENTINEL_1_WAVELENGTH * theta.tan());
            accumulated_dphi += dphi;
        }
    }
    let end_time = std::time::Instant::now();
    log::info!("Time taken: {:?}", end_time - start_time);

    phase_diff
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, s};
    use rerun::{Image, RecordingStream};

    use crate::{
        coregistration::{
            deramping::DerampSlcBurst, warp_function::resample_secondary_to_reference,
        },
        dem::CopernicusDemType,
        visualization::cubehelix_colormap,
    };

    use super::*;

    fn init_logger() {
        //Records logged during cargo test will not be captured by the test harness by default.
        // The Builder::is_test method can be used in unit tests to ensure logs will be captured
        let _ = env_logger::init();
    }

    #[test]
    fn test_perp_baseline_from_pixel_index() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
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
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_flat_earth_dphi")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let mut dem_corners = dem.corners_lat_lon().to_vec();
        let first = dem_corners.first().unwrap();
        dem_corners.push(*first);
        rr.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
        let [rg_size, az_size] = primary.data.raster_size();
        let data = Array2::<f64>::from_shape_fn((az_size, rg_size / 10), |(az_index, rg_index)| {
            flat_earth_dphi(&primary, &secondary, az_index as f64, rg_index as f64, &dem)
        });
        let vector = data.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase =
                    (phase + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
                cubehelix_colormap(normalized_phase as f32).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [rg_size as u32 / 10, az_size as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log_static("dphi", &rr_image).unwrap();
    }

    #[test]
    fn test_interpolated_flat_earth_dphi() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_interpolated_flat_earth_dphi")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let mut dem_corners = dem.corners_lat_lon().to_vec();
        let first = dem_corners.first().unwrap();
        dem_corners.push(*first);
        rr.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
        let interpolator =
            FlatEarthComponentsInterpolator::from_grid_params(&primary, &secondary, &dem, 140, 20);
        let wrap_phase = |phase: f64| {
            phase - 2.0 * std::f64::consts::PI * (phase / (2.0 * std::f64::consts::PI)).floor()
        };
        let data = interpolator
            .calculate_array()
            .map(|phase| wrap_phase(*phase));
        let data = data.slice(s![.., 0..data.dim().1 / 2]).to_owned();
        let (az_size, rg_size) = data.dim();
        let vector = data.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase = phase / (2.0 * std::f64::consts::PI);
                cubehelix_colormap(normalized_phase as f32).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [rg_size as u32, az_size as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log_static("dphi", &rr_image).unwrap();
    }

    #[test]
    fn test_interpolated_flat_earth_removal() {
        init_logger();
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let [min_lat, max_lat, min_lon, max_lon] = &primary
            .metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .get_bounding_box_lat_lon();
        let offset_lat = 0.05;
        let offset_lon = 0.05;
        let bounds = [
            min_lat - offset_lat,
            max_lat + offset_lat,
            min_lon - offset_lon,
            max_lon + offset_lon,
        ];
        let dem = DEM::download_dem(bounds, CopernicusDemType::Cop30);
        let rr = rerun::RecordingStreamBuilder::new("test_interpolated_flat_earth_removal")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let wrap_phase = |phase: f32| {
            phase - 2.0 * std::f32::consts::PI * (phase / (2.0 * std::f32::consts::PI)).floor()
        };
        let phase = coregister_and_remove_flat_phase(&primary, &secondary, &dem)
            .map(|phase| wrap_phase(*phase) as f32);
        let phase = phase.slice(s![.., 0..phase.dim().1 / 2]).to_owned();
        let (az_size, rg_size) = phase.dim();
        let vector = phase.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase = phase / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [rg_size as u32, az_size as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log_static("phase", &rr_image).unwrap();
    }

    #[test]
    fn get_bounding_box_lat_lon() {
        let burst = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let metadata = &burst.metadata;
        let [min_lat, max_lat, min_lon, max_lon] = &metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .get_bounding_box_lat_lon();
        let offset_lat = 0.005;
        let offset_lon = 0.005;
        let bounds = [
            min_lat - offset_lat,
            max_lat + offset_lat,
            min_lon - offset_lon,
            max_lon + offset_lon,
        ];
        let dem = DEM::download_dem(bounds, CopernicusDemType::Cop30);
        println!("dem: {:?}", dem.corners_lat_lon());
    }
}
