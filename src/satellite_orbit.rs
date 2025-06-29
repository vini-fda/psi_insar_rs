use std::path::Path;

use crate::{
    constants::C_LIGHT,
    dem::DEM,
    geodesy::geodetic_to_ecef,
    interpolation::{
        unit_derivative_interval_cubic_hermite_spline_interpolation,
        unit_interval_cubic_hermite_spline_interpolation,
        unit_second_derivative_interval_cubic_hermite_spline_interpolation,
    },
    metadata::{
        annotation_xml::{OrbitList, SlcProductAnnotation},
        orbit_xml::{EarthExplorerFile, ListOfOsvs},
    },
};
use chrono::{DateTime, TimeDelta, Utc};
use nalgebra::Vector3;
use num_traits::Float;
use rustfft::num_traits::Zero;

/// The radar coordinates of a ground target captured by a satellite.
///
/// - `time`: The zero-Doppler time.
/// - `distance_to_target`: The distance from the satellite to the ground target.
#[derive(Clone, Copy, Debug)]
pub struct RadarCoords {
    /// The zero-Doppler time.
    pub time: DateTime<Utc>,
    /// The distance from the satellite to the ground target.
    pub distance_to_target: f64,
}

/// A collection of orbital state vectors (position and velocity) over time
///
/// This structure stores a time series of orbital states and provides methods
/// for interpolating state values at arbitrary times and computing related
/// orbital parameters.
pub struct OrbitalStateHistory {
    pub time: Vec<DateTime<Utc>>,
    pub position: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
}

impl OrbitalStateHistory {
    /// From a Precise Orbit Ephemerides file, and a timeframe (start_time, end_time)
    pub fn from_poe_timeframe<P: AsRef<Path>>(
        path: P,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Self {
        let eef = EarthExplorerFile::open(path);
        let osvs = eef.data_block.list_of_osvs.osv;
        // osv.utc
        let first_index: usize = osvs.iter().rposition(|osv| osv.utc <= start_time).unwrap();
        let last_index: usize = osvs.iter().position(|osv| osv.utc >= end_time).unwrap();
        let n = (last_index + 1) - first_index;
        let mut time = Vec::with_capacity(n);
        let mut position = Vec::with_capacity(n);
        let mut velocity = Vec::with_capacity(n);

        for osv in &osvs[first_index..=last_index] {
            time.push(osv.utc);
            position.push([osv.x, osv.y, osv.z].into());
            velocity.push([osv.vx, osv.vy, osv.vz].into());
        }
        Self {
            time,
            position,
            velocity,
        }
    }
    #[inline(always)]
    pub fn interp_pos_vel(&self, t: DateTime<Utc>) -> (Vector3<f64>, Vector3<f64>) {
        let time: &[DateTime<Utc>] = self.time.as_slice();
        let pos: &[Vector3<f64>] = self.position.as_slice();
        let vel: &[Vector3<f64>] = self.velocity.as_slice();
        assert!(time.len() >= 2);
        // Try to find "t" in the slice "time"
        // 1. If you can find it, return the corresponding position, velocity pair
        match time.binary_search(&t) {
            Ok(i) => return (pos[i], vel[i]), // exact match
            Err(_) => { /* continue to interpolation */ }
        }
        // 2. Otherwise, search for the previous and next position & velocity (panics if out of bounds)
        let i = time
            .windows(2)
            .position(|w| w[0] <= t && t <= w[1])
            .expect("Interpolation time out of bounds");

        let t_prev = time[i];
        let t_next = time[i + 1];
        let p_prev = pos[i];
        let p_next = pos[i + 1];
        let v_prev = vel[i];
        let v_next = vel[i + 1];
        // 3. With "t_prev" and "t_next", perform interpolation
        let total_dt = (t_next - t_prev).num_nanoseconds().unwrap() as f64;
        let dt = (t - t_prev).num_nanoseconds().unwrap() as f64;
        let alpha = dt / total_dt;

        let total_dt_sec = total_dt * 1e-9;
        // cubic hermite interpolation
        let pos_interp = unit_interval_cubic_hermite_spline_interpolation(
            p_prev,
            v_prev,
            p_next,
            v_next,
            total_dt_sec,
            alpha,
        );
        // Derivative of Hermite spline w.r.t. time
        let vel_interp = unit_derivative_interval_cubic_hermite_spline_interpolation(
            p_prev,
            v_prev,
            p_next,
            v_next,
            total_dt_sec,
            alpha,
        );

        (pos_interp, vel_interp)
    }

    #[inline(always)]
    pub fn interp_pos(&self, t: DateTime<Utc>) -> Vector3<f64> {
        let time: &[DateTime<Utc>] = self.time.as_slice();
        let pos: &[Vector3<f64>] = self.position.as_slice();
        assert!(time.len() >= 2);
        // Try to find "t" in the slice "time"
        // 1. If you can find it, return the corresponding position, velocity pair
        match time.binary_search(&t) {
            Ok(i) => return pos[i], // exact match
            Err(_) => { /* continue to interpolation */ }
        }
        let vel: &[Vector3<f64>] = self.velocity.as_slice();
        // 2. Otherwise, search for the previous and next position & velocity (panics if out of bounds)
        let i = time
            .windows(2)
            .position(|w| w[0] <= t && t <= w[1])
            .expect("Interpolation time out of bounds");

        let t_prev = time[i];
        let t_next = time[i + 1];
        let p_prev = pos[i];
        let p_next = pos[i + 1];
        let v_prev = vel[i];
        let v_next = vel[i + 1];
        // 3. With "t_prev" and "t_next", perform interpolation
        let total_dt = (t_next - t_prev).num_nanoseconds().unwrap() as f64;
        let dt = (t - t_prev).num_nanoseconds().unwrap() as f64;
        let alpha = dt / total_dt;

        let total_dt_sec = total_dt * 1e-9;
        // cubic hermite interpolation

        unit_interval_cubic_hermite_spline_interpolation(
            p_prev,
            v_prev,
            p_next,
            v_next,
            total_dt_sec,
            alpha,
        )
    }

    /// Calculate the zero-Doppler state (time and distance to target) for a given ground target and satellite trajectory.
    ///
    /// The zero-Doppler time is the time `t` such that the satellite's velocity vector
    /// is perpendicular to the vector pointing from the satellite to the ground target:
    ///
    ///     v(t) · (ground_target_pos - s(t)) = 0
    ///
    /// This condition implies a dot product of zero between the velocity vector and
    /// the look vector, indicating orthogonality.
    ///
    /// With the time "t" calculated, we can also obtain the slant range distance to the ground target.
    pub fn find_zero_doppler_state(&self, ground_target_pos: Vector3<f64>) -> RadarCoords {
        const NUM_BISECTION_ITER: usize = 64;
        const TOLERANCE: f64 = 1e-12;
        let time = self.time.as_slice();
        assert!(time.len() >= 2);

        let f = |t: DateTime<Utc>| {
            let (sat_pos, sat_vel) = self.interp_pos_vel(t);
            let normalized_displacement = (ground_target_pos - sat_pos).normalize();
            sat_vel.dot(&normalized_displacement)
        };

        let state = |t: DateTime<Utc>| {
            let (sat_pos, _) = self.interp_pos_vel(t);
            let distance_to_target = (ground_target_pos - sat_pos).norm();
            RadarCoords {
                time: t,
                distance_to_target,
            }
        };

        // Step 1: Search for a sign change across time intervals
        for i in 0..time.len() - 1 {
            let t0 = time[i];
            let t1 = time[i + 1];
            let f0 = f(t0);
            let f1 = f(t1);

            if f0 * f1 <= 0.0 {
                // Step 2: Narrow down using bisection over datetime
                let mut left = t0;
                let mut right = t1;
                for _ in 0..NUM_BISECTION_ITER {
                    let mid = left + (right - left) / 2;
                    let fm = f(mid);

                    if fm.abs() < TOLERANCE {
                        return state(mid);
                    } else if f0 * fm < 0.0 {
                        right = mid;
                    } else {
                        left = mid;
                    }
                }
                let half = (right - left) / 2;
                let mid = left + half;
                return state(mid);
            }
        }

        panic!("Zero-Doppler point not found in trajectory window.");
    }

    /// Calculate the ground target position for a given zero-Doppler state (time and distance to target) and satellite trajectory.
    ///
    /// The zero-Doppler time is the time `t` such that the satellite's velocity vector
    /// is perpendicular to the vector pointing from the satellite to the ground target:
    ///
    ///     v(t) · (ground_target_pos - s(t)) = 0
    ///
    /// This condition implies a dot product of zero between the velocity vector and
    /// the look vector, indicating orthogonality.
    ///
    /// With the time "t" calculated, we can also obtain the slant range distance to the ground target.
    pub fn find_ground_target(&self, radar_coords: RadarCoords, dem: &DEM) -> Vector3<f64> {
        let mut current_min = std::f64::MAX;
        let mut optimal_ground_pos = Vector3::<f64>::zero();
        let (sat_pos, sat_vel) = self.interp_pos_vel(radar_coords.time);
        let sat_vel_hat = sat_vel.normalize();
        for (lat, lon, height) in dem.lat_lon_height_iter() {
            let ground_pos = Vector3::from(geodetic_to_ecef(lat, lon, height));
            let val = (ground_pos - sat_pos).dot(&sat_vel_hat).abs();
            if val < current_min {
                current_min = val;
                optimal_ground_pos = ground_pos;
            }
        }
        optimal_ground_pos
    }

    /// Creates an upsampled version of Self, inserting k >= 1 samples inbetween every two samples in the original
    pub fn interp_n(&self, k: usize) -> Self {
        assert!(k >= 1);
        let n = self.time.len();
        let n_ups = k * (n - 1) + n;
        let mut time_ups = vec![DateTime::<Utc>::default(); n_ups];
        for i in 0..(n - 1) {
            time_ups[(k + 1) * i] = self.time[i];
            let delta_t = (self.time[i + 1] - self.time[i]).num_nanoseconds().unwrap() as f64
                / (k as f64 + 1.0);
            for j in 1..=k {
                let total_delta_t = delta_t * j as f64;
                let total_delta_t = TimeDelta::nanoseconds(total_delta_t.round() as i64);
                time_ups[(k + 1) * i + j] = self.time[i] + total_delta_t;
            }
        }
        time_ups[(k + 1) * (n - 1)] = self.time[n - 1];
        let mut position_ups = Vec::with_capacity(n_ups);
        let mut velocity_ups = Vec::with_capacity(n_ups);
        for t in &time_ups {
            let (pos, vel) = self.interp_pos_vel(*t);
            position_ups.push(pos);
            velocity_ups.push(vel);
        }
        Self {
            time: time_ups,
            position: position_ups,
            velocity: velocity_ups,
        }
    }
}

impl From<OrbitList> for OrbitalStateHistory {
    fn from(list: OrbitList) -> Self {
        Self::from(&list)
    }
}

impl From<&OrbitList> for OrbitalStateHistory {
    fn from(orbit_list: &OrbitList) -> Self {
        let mut time = Vec::with_capacity(orbit_list.count as usize);
        let mut position = Vec::with_capacity(orbit_list.count as usize);
        let mut velocity = Vec::with_capacity(orbit_list.count as usize);

        for orbit in orbit_list.orbit.iter() {
            time.push(orbit.time);
            position.push(orbit.position.into());
            velocity.push(orbit.velocity.into());
        }

        OrbitalStateHistory {
            time,
            position,
            velocity,
        }
    }
}

impl From<ListOfOsvs> for OrbitalStateHistory {
    fn from(list: ListOfOsvs) -> Self {
        Self::from(&list)
    }
}

impl From<&ListOfOsvs> for OrbitalStateHistory {
    fn from(osv_list: &ListOfOsvs) -> Self {
        let mut time = Vec::with_capacity(osv_list.count);
        let mut position = Vec::with_capacity(osv_list.count);
        let mut velocity = Vec::with_capacity(osv_list.count);

        for osv in osv_list.osv.iter() {
            time.push(osv.utc);
            position.push([osv.x, osv.y, osv.z].into());
            velocity.push([osv.vx, osv.vy, osv.vz].into());
        }

        OrbitalStateHistory {
            time,
            position,
            velocity,
        }
    }
}

/// A collection of orbital state vectors (position and velocity) over time.
///
/// In this struct, unlike `OrbitalStateHistory`, the time points are represented as seconds since the start time (from metadata).
///
/// This structure stores a time series of orbital states and provides methods
/// for interpolating state values at arbitrary times and computing related
/// orbital parameters.
pub struct ContinuousOrbitalStateHistory {
    pub time: Vec<f64>,
    pub position: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    near_edge_slant_range: f64,
    range_spacing: f64,
    azimuth_time_interval: f64,
}

impl ContinuousOrbitalStateHistory {
    /// From a Orbital State History
    pub fn from_osh(
        osh: &OrbitalStateHistory,
        start_time: DateTime<Utc>,
        annotation: &SlcProductAnnotation,
    ) -> Self {
        let time = osh
            .time
            .iter()
            .map(|&t_datetime| {
                t_datetime
                    .signed_duration_since(start_time)
                    .as_seconds_f64()
            })
            .collect();
        let slant_range_time = annotation
            .image_annotation
            .image_information
            .slant_range_time;

        Self {
            time,
            near_edge_slant_range: 0.5 * C_LIGHT * slant_range_time,
            range_spacing: annotation
                .image_annotation
                .image_information
                .range_pixel_spacing,
            azimuth_time_interval: annotation
                .image_annotation
                .image_information
                .azimuth_time_interval,
            position: osh.position.clone(),
            velocity: osh.velocity.clone(),
        }
    }

    /// Interpolate p(t) and v(t) at
    #[inline(always)]
    pub fn interp_pos_vel(&self, t: f64) -> (Vector3<f64>, Vector3<f64>) {
        let time: &[f64] = self.time.as_slice();
        let pos: &[Vector3<f64>] = self.position.as_slice();
        let vel: &[Vector3<f64>] = self.velocity.as_slice();
        assert!(time.len() >= 2);
        // 1. Search for the previous and next position & velocity (panics if out of bounds)
        let i = time
            .windows(2)
            .position(|w| w[0] <= t && t <= w[1])
            .expect("Interpolation time out of bounds");

        let t_prev = time[i];
        let t_next = time[i + 1];
        let p_prev = pos[i];
        let p_next = pos[i + 1];
        let v_prev = vel[i];
        let v_next = vel[i + 1];
        // 2. With "t_prev" and "t_next", perform interpolation
        let total_dt = t_next - t_prev;
        let dt = t - t_prev;
        let alpha = dt / total_dt;

        let total_dt_sec = total_dt;
        // cubic hermite interpolation
        let pos_interp = unit_interval_cubic_hermite_spline_interpolation(
            p_prev,
            v_prev,
            p_next,
            v_next,
            total_dt_sec,
            alpha,
        );
        // Derivative of Hermite spline w.r.t. time
        let vel_interp = unit_derivative_interval_cubic_hermite_spline_interpolation(
            p_prev,
            v_prev,
            p_next,
            v_next,
            total_dt_sec,
            alpha,
        );

        (pos_interp, vel_interp)
    }

    #[inline(always)]
    pub fn interp_pos(&self, t: f64) -> Vector3<f64> {
        let time: &[f64] = self.time.as_slice();
        let pos: &[Vector3<f64>] = self.position.as_slice();
        assert!(time.len() >= 2);
        let vel: &[Vector3<f64>] = self.velocity.as_slice();
        // 1. Search for the previous and next position & velocity (panics if out of bounds)
        let i = time
            .windows(2)
            .position(|w| w[0] <= t && t <= w[1])
            .expect("Interpolation time out of bounds");

        let t_prev = time[i];
        let t_next = time[i + 1];
        let p_prev = pos[i];
        let p_next = pos[i + 1];
        let v_prev = vel[i];
        let v_next = vel[i + 1];
        // 2. With "t_prev" and "t_next", perform interpolation
        let total_dt = t_next - t_prev;
        let dt = t - t_prev;
        let alpha = dt / total_dt;

        let total_dt_sec = total_dt;
        // cubic hermite interpolation

        unit_interval_cubic_hermite_spline_interpolation(
            p_prev,
            v_prev,
            p_next,
            v_next,
            total_dt_sec,
            alpha,
        )
    }

    /// Interpolate p(t), v(t), and a(t) at time t
    #[inline(always)]
    pub fn interp_pos_vel_acc(&self, t: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let time: &[f64] = self.time.as_slice();
        let pos: &[Vector3<f64>] = self.position.as_slice();
        let vel: &[Vector3<f64>] = self.velocity.as_slice();
        assert!(time.len() >= 2);

        // Search for the previous and next position & velocity (panics if out of bounds)
        let i = time
            .windows(2)
            .position(|w| w[0] <= t && t <= w[1])
            .expect("Interpolation time out of bounds");

        let t_prev = time[i];
        let t_next = time[i + 1];
        let p_prev = pos[i];
        let p_next = pos[i + 1];
        let v_prev = vel[i];
        let v_next = vel[i + 1];

        // With "t_prev" and "t_next", perform interpolation
        let total_dt = t_next - t_prev;
        let dt = t - t_prev;
        let alpha = dt / total_dt;

        // cubic hermite interpolation for position
        let pos_interp = unit_interval_cubic_hermite_spline_interpolation(
            p_prev, v_prev, p_next, v_next, total_dt, alpha,
        );

        // First derivative of Hermite spline w.r.t. time (velocity)
        let vel_interp = unit_derivative_interval_cubic_hermite_spline_interpolation(
            p_prev, v_prev, p_next, v_next, total_dt, alpha,
        );

        // Second derivative of Hermite spline w.r.t. time (acceleration)
        let acc_interp = unit_second_derivative_interval_cubic_hermite_spline_interpolation(
            p_prev, v_prev, p_next, v_next, total_dt, alpha,
        );

        (pos_interp, vel_interp, acc_interp)
    }

    /// Calculate the zero-Doppler state (time and distance to target) for a given ground target and satellite trajectory.
    ///
    /// The zero-Doppler time is the time `t` such that the satellite's velocity vector
    /// is perpendicular to the vector pointing from the satellite to the ground target:
    ///
    ///     v(t) · (ground_target_pos - s(t)) = 0
    ///
    /// This condition implies a dot product of zero between the velocity vector and
    /// the look vector, indicating orthogonality.
    ///
    /// With the time "t" calculated, we can also obtain the slant range distance to the ground target.
    pub fn find_zero_doppler_state(&self, ground_target_pos: Vector3<f64>) -> [f64; 2] {
        const NUM_BISECTION_ITER: usize = 32;
        const TOLERANCE: f64 = 1e-9;
        let time = self.time.as_slice();
        assert!(time.len() >= 2);

        let f = |t: f64| {
            let (sat_pos, sat_vel) = self.interp_pos_vel(t);
            let normalized_displacement = (ground_target_pos - sat_pos).normalize();
            sat_vel.normalize().dot(&normalized_displacement)
        };

        // Step 1: Search for a sign change across time intervals
        let mut distance_to_target = f64::NAN;
        let mut delta_time_secs = f64::NAN;
        'outer: for i in 0..time.len() - 1 {
            let t0 = time[i];
            let t1 = time[i + 1];
            let f0 = f(t0);
            let f1 = f(t1);

            if f0 * f1 <= 0.0 {
                // Step 2: Narrow down using bisection over time
                let mut left = t0;
                let mut right = t1;
                for _ in 0..NUM_BISECTION_ITER {
                    let mid = left + (right - left) / 2.0;
                    let fm = f(mid);

                    if fm.abs() < TOLERANCE {
                        let sat_pos = self.interp_pos(mid);
                        distance_to_target = (ground_target_pos - sat_pos).norm();
                        delta_time_secs = mid;
                        break 'outer;
                    } else if f0 * fm < 0.0 {
                        right = mid;
                    } else {
                        left = mid;
                    }
                }
                let half = (right - left) / 2.0;
                let mid = left + half;
                let sat_pos = self.interp_pos(mid);
                distance_to_target = (ground_target_pos - sat_pos).norm();
                delta_time_secs = mid;
                break 'outer;
            }
        }

        if distance_to_target.is_nan() || delta_time_secs.is_nan() {
            panic!("NaN found in bisection");
        }

        // Calculation using azimuth_time_interval (similar to linear interp)
        // This is what the official SNAP microwave toolbox performs:
        // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-io/src/main/java/eu/esa/sar/io/sentinel1/Sentinel1Level1Directory.java
        // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/support/SARPosition.java
        let azimuth_index = delta_time_secs / self.azimuth_time_interval;

        // Column calculation:
        let slant_range_index =
            (distance_to_target - self.near_edge_slant_range) / self.range_spacing;
        [azimuth_index, slant_range_index]
    }

    /// Calculate the zero-Doppler state using Newton-Raphson method.
    ///
    /// This is an alternative implementation to the bisection method that can converge
    /// faster but requires computing the derivative (using satellite acceleration).
    ///
    /// The zero-Doppler time is the time `t` such that:
    ///     v(t) · (ground_target_pos - s(t)) = 0
    pub fn find_zero_doppler_state_newton_raphson(
        &self,
        ground_target_pos: Vector3<f64>,
    ) -> [f64; 2] {
        const MAX_ITER: usize = 10;
        const TOLERANCE: f64 = 1e-12;
        let time = self.time.as_slice();
        assert!(time.len() >= 2);

        // Function f(t) = v(t) · (ground_target_pos - s(t))
        // We don't normalize here to simplify the derivative
        let f = |t: f64| {
            let (sat_pos, sat_vel) = self.interp_pos_vel(t);
            sat_vel.dot(&(ground_target_pos - sat_pos))
        };

        // Derivative f'(t) = a(t) · (ground_target_pos - s(t)) - v(t) · v(t)
        let f_prime = |t: f64| {
            let (sat_pos, sat_vel, sat_acc) = self.interp_pos_vel_acc(t);
            sat_acc.dot(&(ground_target_pos - sat_pos)) - sat_vel.dot(&sat_vel)
        };

        // Step 1: Find a good initial guess by searching for a sign change
        let mut t_guess = f64::NAN;
        for i in 0..time.len() - 1 {
            let t0 = time[i];
            let t1 = time[i + 1];
            let f0 = f(t0);
            let f1 = f(t1);

            if f0 * f1 <= 0.0 {
                // Use the midpoint as initial guess
                t_guess = (t0 + t1) / 2.0;
                break;
            }
        }

        if t_guess.is_nan() {
            panic!("No sign change found - zero-Doppler point may not exist in trajectory window");
        }

        // Step 2: Newton-Raphson iteration
        let mut t = t_guess;
        for iter in 0..MAX_ITER {
            let f_val = f(t);

            if f_val.abs() < TOLERANCE {
                break;
            }

            let f_prime_val = f_prime(t);

            if f_prime_val.abs() < 1e-15 {
                // Derivative too small, fall back to bisection for this step
                // This is rare but can happen at inflection points
                break;
            }

            let dt = -f_val / f_prime_val;
            t += dt;

            // Ensure t stays within bounds
            if t < time[0] {
                t = time[0];
            } else if t > time[time.len() - 1] {
                t = time[time.len() - 1];
            }

            if dt.abs() < 1e-12 {
                break;
            }
        }

        // Calculate final results
        let sat_pos = self.interp_pos(t);
        let distance_to_target = (ground_target_pos - sat_pos).norm();
        let delta_time_secs = t;

        // Convert to pixel indices
        let azimuth_index = delta_time_secs / self.azimuth_time_interval;
        let slant_range_index =
            (distance_to_target - self.near_edge_slant_range) / self.range_spacing;

        [azimuth_index, slant_range_index]
    }
}

/// Converts the radar coordinates (azimuth time and slant range distance to target) to the pixel coordinates (azimuth index and slant range index) in the SLC image.
///
/// # Returns
/// An array containing the azimuth index and the slant range index, both as `f32`.
///
/// # Panics
/// Panics if the zero-Doppler time is outside the range of the SLC image.
#[inline(always)]
pub fn radar_coords_to_pixel_coords<T: Float>(
    zero_doppler: RadarCoords,
    annotation: &SlcProductAnnotation,
) -> [T; 2] {
    let slant_range_time = annotation
        .image_annotation
        .image_information
        .slant_range_time;
    let near_edge_slant_range = 0.5 * C_LIGHT * slant_range_time;
    let t_start = annotation
        .image_annotation
        .image_information
        .product_first_line_utc_time;

    let range_spacing = annotation
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let azimuth_time_interval = annotation
        .image_annotation
        .image_information
        .azimuth_time_interval;

    // Time difference in seconds
    let delta_time = zero_doppler.time.signed_duration_since(t_start);
    let delta_time_secs = delta_time.num_microseconds().unwrap() as f64 * 1.0e-6;

    // Calculation using azimuth_time_interval (similar to linear interp)
    // This is what the official SNAP microwave toolbox performs:
    // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-io/src/main/java/eu/esa/sar/io/sentinel1/Sentinel1Level1Directory.java
    // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/support/SARPosition.java
    let azimuth_index = delta_time_secs / azimuth_time_interval;

    // Column calculation:
    let slant_range_index =
        (zero_doppler.distance_to_target - near_edge_slant_range) / range_spacing;
    [
        T::from(azimuth_index).unwrap_or_else(|| {
            panic!(
                "could not convert azimuth_index to {}",
                std::any::type_name::<T>()
            )
        }),
        T::from(slant_range_index).unwrap_or_else(|| {
            panic!(
                "could not convert slant_range_index to {}",
                std::any::type_name::<T>()
            )
        }),
    ]
}

/// Converts the pixel coordinates (azimuth index and slant range index) to the radar coordinates (azimuth time and slant range distance to target).
///
/// # Returns
/// The radar coordinates.
///
/// # Panics
/// Panics if either the azimuth index or the slant range index is out of bounds.
#[inline(always)]
pub fn pixel_coords_to_radar_coords(
    azimuth_index: f64,
    slant_range_index: f64,
    annotation: &SlcProductAnnotation,
) -> RadarCoords {
    let slant_range_time = annotation
        .image_annotation
        .image_information
        .slant_range_time;
    let near_edge_slant_range = 0.5 * C_LIGHT * slant_range_time;

    let t_start = annotation
        .image_annotation
        .image_information
        .product_first_line_utc_time;

    let range_spacing = annotation
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let azimuth_time_interval = annotation
        .image_annotation
        .image_information
        .azimuth_time_interval;

    // Reverse azimuth index calculation to get time
    let delta_time_secs = azimuth_index * azimuth_time_interval;
    let delta_time_nanos = (delta_time_secs * 1_000_000_000.0).round() as i64;
    let zero_doppler_time = t_start + TimeDelta::nanoseconds(delta_time_nanos);

    // Reverse slant range index calculation to get distance
    let distance_to_target = slant_range_index * range_spacing + near_edge_slant_range;

    RadarCoords {
        time: zero_doppler_time,
        distance_to_target,
    }
}

pub fn zero_doppler_time(azimuth_index: f64, annotation: &SlcProductAnnotation) -> DateTime<Utc> {
    let t_start = annotation
        .image_annotation
        .image_information
        .product_first_line_utc_time;

    let azimuth_time_interval = annotation
        .image_annotation
        .image_information
        .azimuth_time_interval;

    // Reverse azimuth index calculation to get time
    let delta_time_secs = azimuth_index * azimuth_time_interval;
    let delta_time_nanos = (delta_time_secs * 1_000_000_000.0).round() as i64;

    t_start + TimeDelta::nanoseconds(delta_time_nanos)
}

#[cfg(test)]
mod manual_tests_satellite_orbit {
    use std::f32::consts::PI;

    use nalgebra::{Matrix3, Unit, Vector3};
    use rerun::Color;

    use super::OrbitalStateHistory;
    use crate::{
        dem::DEM,
        geodesy::{geodetic_to_ecef, local_normal},
        metadata::annotation_xml::{OrbitList, SlcProductAnnotation},
        satellite_orbit::radar_coords_to_pixel_coords,
    };

    /// Reads the first `<orbitList>` element found in the XML file at `path`.
    fn read_orbit_list_from_file(path: &str) -> OrbitList {
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);

        let xml_de = &mut quick_xml::de::Deserializer::from_reader(reader);
        // Parse the XML into our Product struct
        let result: Result<SlcProductAnnotation, _> = serde_path_to_error::deserialize(xml_de);
        let slc_product_annotation = match result {
            Ok(val) => val,
            Err(err) => {
                let path = err.path().to_string();
                panic!("Error parsing XML\nError path: {path}\nError: {err}");
            }
        };
        slc_product_annotation.general_annotation.orbit_list
    }

    #[test]
    fn read_orbit_list() {
        let orbit_list = read_orbit_list_from_file("src/metadata/test_data/annotation_example.xml");
        // println!("orbitList = {:?}", orbit_list);
    }

    #[test]
    fn find_zero_doppler_from_orbit_list() {
        let annotation =
            SlcProductAnnotation::open("src/metadata/test_data/annotation_example.xml");
        // let orbit_list = read_orbit_list_from_file("src/metadata/test_data/annotation_example.xml");
        let start_time = annotation.ads_header.start_time;
        let end_time = annotation.ads_header.stop_time;
        let osh = OrbitalStateHistory::from_poe_timeframe("orbit.EOF", start_time, end_time);
        let dem = DEM::open_file("dem.tif");
        let [lat, lon] = [19.49831428810679, -98.59301000370277];
        let pos = dem.get_ecef_at_lat_lon(lat, lon);

        let zero_doppler = osh.find_zero_doppler_state(pos.into());
        println!("Zero-Doppler time = {zero_doppler:?}");
        let [row, col]: [f32; 2] = radar_coords_to_pixel_coords(zero_doppler, &annotation);
        println!("Found pixel at {row}, {col}");
    }

    #[test]
    #[ignore]
    fn test_backgeocoding() {
        let rr = rerun::RecordingStreamBuilder::new("test_backgeocoding")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let annotation =
            SlcProductAnnotation::open("src/metadata/test_data/annotation_example.xml");
        let osh = OrbitalStateHistory::from(&annotation.general_annotation.orbit_list);
        let dem = DEM::open_file("dem.tif");

        let mut points = vec![];
        for (lat, lon) in dem.lat_lon_iter() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let ground_target_pos = Vector3::<f64>::from(pos);
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
            let [azimuth_idx, slant_range_idx] =
                radar_coords_to_pixel_coords(zero_doppler, &annotation);

            points.push([slant_range_idx, azimuth_idx]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors(dem.vertex_colors())
            .with_radii([10.0]);
        rr.log_static("backgeocoded_points", &points).unwrap();

        // Log ground control points
        let gcps = &annotation
            .geolocation_grid
            .geolocation_grid_point_list
            .geolocation_grid_point;

        let mut points = vec![];
        for gcp in gcps {
            let pos = geodetic_to_ecef(gcp.latitude, gcp.longitude, gcp.height);
            let zero_doppler = osh.find_zero_doppler_state(pos.into());
            let [azimuth_idx, slant_range_idx] =
                radar_coords_to_pixel_coords(zero_doppler, &annotation);

            points.push([slant_range_idx, azimuth_idx]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors([rerun::Color::from_rgb(255, 122, 100)])
            .with_radii([30.0]);
        rr.log_static("backgeocoded_gcps", &points).unwrap();

        // part 2
        let annotation = SlcProductAnnotation::open(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE/annotation/s1a-iw3-slc-vv-20151010t122546-20151010t122550-008090-00b578-001.xml",
        );
        let osh = OrbitalStateHistory::from(&annotation.general_annotation.orbit_list);
        let dem = DEM::open_file("dem.tif");

        let mut points = vec![];
        for (lat, lon) in dem.lat_lon_iter() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let zero_doppler = osh.find_zero_doppler_state(pos.into());
            let [azimuth_idx, slant_range_idx] =
                radar_coords_to_pixel_coords(zero_doppler, &annotation);

            points.push([slant_range_idx, azimuth_idx]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors(dem.vertex_colors().iter().map(|&n| n & 0x0000FFFF))
            .with_radii([10.0]);
        rr.log_static("backgeocoded_points_2", &points).unwrap();

        let mut points = vec![];
        for gcp in gcps {
            let pos = geodetic_to_ecef(gcp.latitude, gcp.longitude, gcp.height);
            let zero_doppler = osh.find_zero_doppler_state(pos.into());
            let [azimuth_idx, slant_range_idx] =
                radar_coords_to_pixel_coords(zero_doppler, &annotation);

            points.push([slant_range_idx, azimuth_idx]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors([rerun::Color::from_rgb(122, 255, 100)])
            .with_radii([30.0]);
        rr.log_static("backgeocoded_gcps_2", &points).unwrap();
    }

    #[test]
    #[ignore]
    fn simple() {
        let rec = rerun::RecordingStreamBuilder::new("simple_test_satellite_orbit")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let annotation =
            SlcProductAnnotation::open("src/metadata/test_data/annotation_example.xml");
        let orbit_list = &annotation.general_annotation.orbit_list;
        let orbital_history = OrbitalStateHistory::from(orbit_list).interp_n(4);
        let points = orbital_history
            .position
            .iter()
            .map(|v| v.map(|x| x as f32).data.0[0]);

        rec.log_static(
            "orbital_positions",
            &rerun::Points3D::new(points).with_radii([1000.0]),
        )
        .unwrap();

        // Record time-series of satellite position
        let n = orbital_history.time.len();
        for k in 0..n {
            let sat_pos = orbital_history.position[k];
            let vel = orbital_history.velocity[k];
            let time = orbital_history.time[k];
            let look_rot3x3 = look_rotation_from_velocity_and_position(
                vel.map(|c| c as f32),
                sat_pos.map(|c| c as f32),
            );
            rec.set_time_nanos("satellite_time", time.timestamp_nanos_opt().unwrap());

            // log normals as arrows
            let mut normals = vec![];
            let mut sat_look_vectors = vec![];
            let mut positions = vec![];
            for gcp in &annotation
                .geolocation_grid
                .geolocation_grid_point_list
                .geolocation_grid_point
            {
                let gcp_pos =
                    geodetic_to_ecef(gcp.latitude, gcp.longitude, gcp.height).map(|val| val as f32);
                let gcp_pos_vec3 = Vector3::<f32>::from(gcp_pos);
                let l = (sat_pos.map(|c| c as f32) - gcp_pos_vec3).normalize();
                sat_look_vectors.push([l.x * 1000.0, l.y * 1000.0, l.z * 1000.0]);
                let normal = local_normal(gcp.latitude, gcp.longitude).map(|val| val as f32);
                let normal_vec3 = Vector3::<f32>::from(normal);
                let theta = l.dot(&normal_vec3).acos();
                let log_title = format!("theta for gcp {}, {}", gcp.latitude, gcp.longitude);
                rec.log(
                    log_title,
                    &rerun::TextLog::new(format!("{}", theta.to_degrees())),
                )
                .unwrap();
                normals.push(normal.map(|c| c * 1000.0));
                positions.push(gcp_pos);
            }
            rec.log(
                "geo_normals",
                &rerun::Arrows3D::from_vectors(normals).with_origins(positions.clone()),
            )
            .unwrap();
            rec.log(
                "satellite_look_vectors",
                &rerun::Arrows3D::from_vectors(sat_look_vectors).with_origins(positions),
            )
            .unwrap();
            let sat_pos =
                rerun::Position3D::new(sat_pos.x as f32, sat_pos.y as f32, sat_pos.z as f32);
            rec.log(
                "satellite_position",
                &rerun::Points3D::new([sat_pos])
                    .with_colors([Color::WHITE])
                    .with_radii([1500.0]),
            )
            .unwrap();
            let arrow_vel =
                rerun::Arrows3D::from_vectors([(vel.x as f32, vel.y as f32, vel.z as f32)])
                    .with_origins([sat_pos]);
            rec.log("satellite_velocity", &arrow_vel).unwrap();

            // Pinhole camera
            let ground_target = geodetic_to_ecef(19.49831428810679, -98.59301000370277, 0.0);
            let ground_target_vec3 = Vector3::<f32>::from(ground_target.map(|val| val as f32));
            let pos_vec3 = Vector3::<f32>::new(sat_pos.x(), sat_pos.y(), sat_pos.z());
            let rot3x3 = look_at_ground_target(pos_vec3, ground_target_vec3);
            rec.log(
                "universe/camera",
                &rerun::Transform3D::from_translation_mat3x3(
                    [sat_pos.x(), sat_pos.y(), sat_pos.z()],
                    rerun::Mat3x3(rot3x3.data.0.as_flattened().try_into().unwrap()),
                ),
            )
            .unwrap();
            let focal_length = (pos_vec3 - ground_target_vec3).norm();
            rec.log(
                "universe/camera",
                &rerun::Pinhole::from_focal_length_and_resolution(
                    [focal_length, focal_length],
                    [23739., 1507.],
                ),
            )
            .unwrap();
            // TODO: Pinhole camera

            // rec.log(
            //     "universe/camera",
            //     &rerun::Pinhole::new(intrinsics)
            //         // See https://github.com/google-research-datasets/Objectron/issues/39 for coordinate systems
            //         .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            //         .with_resolution(resolution),
            // )
            // .unwrap();
        }

        const EARTH_RADIUS: f32 = 6_378_137.0;
        rec.log_static("universe", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())
            .unwrap();
        let asset = rerun::Asset3D::from_file("earth.glb").unwrap();

        rec.log_static(
            "universe/earth",
            &rerun::Transform3D::from_rotation_scale(
                rerun::RotationAxisAngle::new(
                    [1.0, -1.0, -1.0],
                    rerun::Angle::from_radians(2.0 * PI / 3.0),
                ),
                rerun::Scale3D::from(EARTH_RADIUS / 500.0),
            ),
        )
        .unwrap();
        rec.log_static("universe/earth", &asset).unwrap();

        // X Y Z arrows
        const ARROW_LENGTH: f32 = 1.2 * EARTH_RADIUS;
        let arrow_x = rerun::Arrows3D::from_vectors([(ARROW_LENGTH, 0.0, 0.0)])
            .with_origins([(0.0, 0.0, 0.0)]);
        rec.log_static("universe/x", &arrow_x).unwrap();
        let arrow_y = rerun::Arrows3D::from_vectors([(0.0, ARROW_LENGTH, 0.0)])
            .with_origins([(0.0, 0.0, 0.0)]);
        rec.log_static("universe/y", &arrow_y).unwrap();
        let arrow_z = rerun::Arrows3D::from_vectors([(0.0, 0.0, ARROW_LENGTH)])
            .with_origins([(0.0, 0.0, 0.0)]);
        rec.log_static("universe/z", &arrow_z).unwrap();

        // DEM
        let dem = DEM::open_file("dem.tif");
        let vertex_positions: Vec<[f32; 3]> = dem.vertex_positions();
        let vertex_normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; dem.len()];
        let vertex_colors: Vec<u32> = dem.vertex_colors();
        let triangle_indices = dem.triangle_indices();
        rec.log_static(
            "dem_mesh3d",
            &rerun::Mesh3D::new(vertex_positions)
                .with_vertex_normals(vertex_normals)
                .with_vertex_colors(vertex_colors)
                .with_triangle_indices(triangle_indices),
        )
        .unwrap();

        // XYZ OF GROUND CONTROL POINTS
        let gcps = &annotation
            .geolocation_grid
            .geolocation_grid_point_list
            .geolocation_grid_point;
        let radar_xyz = gcps.iter().map(|gcp| {
            println!("gcp height = {}", gcp.height);
            geodetic_to_ecef(gcp.latitude, gcp.longitude, gcp.height).map(|val| val as f32)
        });

        rec.log_static("geo_points", &rerun::Points3D::new(radar_xyz))
            .unwrap();
    }

    fn look_rotation_from_velocity_and_position(v: Vector3<f32>, p: Vector3<f32>) -> Matrix3<f32> {
        let forward = Unit::new_normalize(v);
        let radial = Unit::new_normalize(p);

        // Remove component of radial in the direction of forward (Gram-Schmidt)
        let up_raw = radial.into_inner();
        let up = Unit::new_normalize(up_raw - forward.into_inner() * up_raw.dot(&forward));

        // Right vector
        let right = Unit::new_normalize(up.cross(&forward));

        // Compose rotation matrix from right, up, forward as columns

        Matrix3::from_columns(&[right.into_inner(), up.into_inner(), forward.into_inner()])
    }

    /// Constructs a rotation matrix that orients an object at `p_sat` to point toward `p_target`.
    pub fn look_at_ground_target(p_sat: Vector3<f32>, p_target: Vector3<f32>) -> Matrix3<f32> {
        let forward_vec = p_target - p_sat;
        let forward = Unit::new_normalize(forward_vec);

        let up_raw = Unit::new_normalize(p_sat).into_inner(); // Radial from Earth center
        let up = Unit::new_normalize(up_raw - forward.into_inner() * up_raw.dot(&forward));
        let right = Unit::new_normalize(up.cross(&forward));

        Matrix3::from_columns(&[right.into_inner(), up.into_inner(), forward.into_inner()])
    }
}
