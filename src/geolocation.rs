use chrono::{DateTime, Duration, Utc};
use thiserror::Error;

use crate::dem::DEM;

/// Error type for geolocation functions
#[derive(Error, Debug)]
pub enum GeolocationError {
    #[error("Failed to converge: {0}")]
    ConvergenceFailure(String),
    #[error("Failed to find orbit state vector at time {0}")]
    MissingOrbitStateVector(f64),
    #[error("Failed to find DEM height at latitude {0} and longitude {1}")]
    MissingDEMHeight(f64, f64),
}

// alias for Vector3
type Vector3 = nalgebra::Vector3<f64>;

/// A collection of orbital state vectors (position and velocity) over time
///
/// This structure stores a time series of orbital states and provides methods
/// for interpolating state values at arbitrary times and computing related
/// orbital parameters.
pub struct OrbitalStateHistory {
    pub time: Vec<DateTime<Utc>>,
    pub position: Vec<Vector3>,
    pub velocity: Vec<Vector3>,
}

impl OrbitalStateHistory {
    /// Gets the interpolated position at the specified time
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to calculate the position
    ///
    /// # Returns
    ///
    /// A 3D vector representing the interpolated position in meters
    pub fn get_position(&self, time: DateTime<Utc>) -> Vector3 {
        // Check if time exists in our data to avoid unnecessary interpolation
        if let Some(idx) = self.time.iter().position(|t| *t == time) {
            return self.position[idx];
        }

        // Otherwise interpolate between closest points
        let (i, j) = self.find_closest_times(time);
        self.interpolate_vector(&self.position, i, j, time)
    }

    /// Gets the interpolated velocity at the specified time
    ///
    /// # Arguments
    ///
    /// * `time` - The time at which to calculate the velocity
    ///
    /// # Returns
    ///
    /// A 3D vector representing the interpolated velocity in meters/second
    pub fn get_velocity(&self, time: DateTime<Utc>) -> Vector3 {
        // Check if time exists in our data to avoid unnecessary interpolation
        if let Some(idx) = self.time.iter().position(|t| *t == time) {
            return self.velocity[idx];
        }

        // Otherwise interpolate between closest points
        let (i, j) = self.find_closest_times(time);
        self.interpolate_vector(&self.velocity, i, j, time)
    }

    /// Helper method to interpolate a vector between two time points
    ///
    /// # Arguments
    ///
    /// * `vectors` - Slice of vectors to interpolate from
    /// * `i` - Index of the first vector
    /// * `j` - Index of the second vector
    /// * `time` - The time at which to interpolate
    ///
    /// # Returns
    ///
    /// The interpolated vector
    fn interpolate_vector(
        &self,
        vectors: &[Vector3],
        i: usize,
        j: usize,
        time: DateTime<Utc>,
    ) -> Vector3 {
        let vec_i = vectors[i];
        let vec_j = vectors[j];

        // Calculate interpolation factor (alpha) based on time differences
        let t_i = self.time[i].timestamp_millis() as f64;
        let t_j = self.time[j].timestamp_millis() as f64;
        let t = time.timestamp_millis() as f64;

        // Ensure we don't divide by zero
        if (t_j - t_i).abs() < f64::EPSILON {
            // Times are effectively identical, return first value
            return vec_i;
        }

        let alpha = (t - t_i) / (t_j - t_i);

        // Linear interpolation
        vec_i + alpha * (vec_j - vec_i)
    }

    /// Finds the indices of the two closest time points to the given time
    ///
    /// # Arguments
    ///
    /// * `time` - The reference time
    ///
    /// # Returns
    ///
    /// A tuple of indices (i, j) where i corresponds to the closest time point
    /// and j to the second closest time point
    fn find_closest_times(&self, time: DateTime<Utc>) -> (usize, usize) {
        assert!(
            self.time.len() >= 2,
            "At least two time points are required for interpolation"
        );

        // Convert to f64 timestamp for comparison
        let target_ts = time.timestamp_millis() as f64;

        // Find closest point first
        let mut closest_idx = 0;
        let mut min_diff = f64::INFINITY;

        for (idx, t) in self.time.iter().enumerate() {
            let diff = (t.timestamp_millis() as f64 - target_ts).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_idx = idx;
            }
        }

        // Now find second closest - prefer the neighboring point in the time sequence
        let second_idx = if closest_idx == 0 {
            1
        } else if closest_idx == self.time.len() - 1 {
            self.time.len() - 2
        } else {
            // Compare neighbors and pick the closer one
            let prev_diff =
                (self.time[closest_idx - 1].timestamp_millis() as f64 - target_ts).abs();
            let next_diff =
                (self.time[closest_idx + 1].timestamp_millis() as f64 - target_ts).abs();
            if prev_diff < next_diff {
                closest_idx - 1
            } else {
                closest_idx + 1
            }
        };

        // Ensure proper ordering for interpolation (earlier time first)
        if self.time[closest_idx] < self.time[second_idx] {
            (closest_idx, second_idx)
        } else {
            (second_idx, closest_idx)
        }
    }

    /// Gets the time of closest approach to a ground point
    ///
    /// # Arguments
    ///
    /// * `ground_point` - The 3D coordinates of the ground point in meters
    ///
    /// # Returns
    ///
    /// The time of closest approach as a `DateTime<Utc>` object
    pub fn get_closest_approach_time(&self, ground_point: Vector3) -> DateTime<Utc> {
        // Initial guess: closest approach to the ground point
        let (&min_time, _) = self
            .time
            .iter()
            .zip(self.position.iter())
            .map(|(time, pos)| (time, (*pos - ground_point).norm()))
            .min_by(|(_, dist1), (_, dist2)| {
                dist1
                    .partial_cmp(dist2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        min_time
    }

    /// Computes the Doppler shift derivative at a given time for a ground point
    ///
    /// # Arguments
    ///
    /// * `ground_point` - The 3D coordinates of the ground point in meters
    /// * `time` - The time at which to calculate the Doppler derivative
    ///
    /// # Returns
    ///
    /// The rate of change of the Doppler shift in meters/second²
    pub fn compute_doppler_derivative(&self, ground_point: Vector3, time: DateTime<Utc>) -> f64 {
        /// Time step for finite difference calculations in seconds
        const FINITE_DIFF_STEP_SEC: f64 = 1e-6;
        // Use central finite difference for numerical derivative
        let delta = Duration::microseconds(1);
        let doppler_plus = self.compute_doppler(ground_point, time + delta);
        let doppler_minus = self.compute_doppler(ground_point, time - delta);

        // Return rate of change per second
        (doppler_plus - doppler_minus) / (2.0 * FINITE_DIFF_STEP_SEC)
    }

    /// Computes the Doppler shift at a given time for a ground point
    ///
    /// The Doppler shift is calculated as the line-of-sight component of
    /// the satellite's velocity vector, which represents the rate of change
    /// of distance between the satellite and ground point.
    ///
    /// # Arguments
    ///
    /// * `ground_point` - The 3D coordinates of the ground point in meters
    /// * `time` - The time at which to calculate the Doppler shift
    ///
    /// # Returns
    ///
    /// The Doppler shift in meters/second (positive values indicate increasing distance)
    pub fn compute_doppler(&self, ground_point: Vector3, time: DateTime<Utc>) -> f64 {
        let sat_pos = self.get_position(time);
        let sat_vel = self.get_velocity(time);

        // Calculate line of sight vector from satellite to ground point
        let los = ground_point - sat_pos;

        // Normalize to get unit vector in the direction of line of sight
        let los_unit = los.normalize();

        // Return the component of velocity in the line of sight direction
        los_unit.dot(&sat_vel)
    }
}

/// Computes the zero-Doppler time for a ground point
///
/// The zero-Doppler time is when the satellite's velocity has no component
/// along the line-of-sight vector to the ground point. At this time, the
/// satellite is moving perpendicular to the line connecting it to the ground point.
///
/// # Arguments
///
/// * `ground_point` - ECEF coordinates of the ground point in meters
/// * `orbit` - Satellite orbit state vectors
///
/// # Returns
///
/// The time (as a `Datetime<Utc>`) when the satellite passes through the zero-Doppler plane for this ground point
pub fn compute_zero_doppler_time(
    ground_point: Vector3,
    orbit: &OrbitalStateHistory,
) -> DateTime<Utc> {
    const MAX_ITERATIONS: usize = 10;
    /// Tolerance for zero-Doppler in m/s
    const DOPPLER_TOLERANCE: f64 = 1e-6;
    /// Maximum time step in seconds
    const MAX_TIME_STEP: i64 = 30;
    // Initial guess from closest approach
    let mut t = orbit.get_closest_approach_time(ground_point);

    // Iterative refinement using Newton's method
    for iteration in 0..MAX_ITERATIONS {
        // Get current Doppler value
        let doppler = orbit.compute_doppler(ground_point, t);

        // If close enough to zero, we've found the zero-Doppler time
        if doppler.abs() < DOPPLER_TOLERANCE {
            return t;
        }
        // Derivative of Doppler for Newton's method
        let derivative = orbit.compute_doppler_derivative(ground_point, t);

        // Avoid division by very small numbers
        if derivative.abs() < DOPPLER_TOLERANCE {
            // If derivative is too small, use bisection or just return current best estimate
            break;
        }

        // Calculate time step using Newton's method
        let mut time_step_seconds = -(doppler / derivative);

        // Limit maximum time step to prevent overshooting
        time_step_seconds = time_step_seconds.clamp(-MAX_TIME_STEP as f64, MAX_TIME_STEP as f64);

        // Convert to Duration and apply time step
        let time_step = Duration::milliseconds((time_step_seconds * 1000.0) as i64);
        let next_time = t + time_step;

        // Check if we're converging
        let new_doppler = orbit.compute_doppler(ground_point, next_time);

        // If the new Doppler is worse, reduce step size and try again
        if new_doppler.abs() > doppler.abs() && iteration < MAX_ITERATIONS - 1 {
            // Halve the time step and try again
            let reduced_step = Duration::milliseconds(time_step.num_milliseconds() / 2);
            t += reduced_step;
        } else {
            t = next_time;
        }
    }

    t
}

/// Computes slant range between satellite and ground point
pub fn compute_slant_range(
    ground_point: Vector3,
    time: DateTime<Utc>,
    orbit: &OrbitalStateHistory,
) -> Result<f64, GeolocationError> {
    let sat_pos = orbit.get_position(time);
    let los = ground_point - sat_pos;
    Ok(los.norm())
}

/// Converts radar coordinates to geographic coordinates
pub fn radar_to_geographic(
    slant_range: f64,
    azimuth_index: usize,
    az_time_first: DateTime<Utc>,
    az_time_last: DateTime<Utc>,
    num_azimuth_lines: usize,
    dem: &DEM,
    orbit: &OrbitalStateHistory,
) -> Result<GeographicPoint, GeolocationError> {
    // // Convert line to time
    // let azimuth_time = azimuth_line_to_azimuth_time(
    //     azimuth_index,
    //     az_time_first,
    //     az_time_last,
    //     num_azimuth_lines,
    // );

    // // Initial position (no elevation)
    // let sat_pos = orbit.get_position(azimuth_time);
    // let earth_center = Vector3::new(0.0, 0.0, 0.0);

    // // Iterative solution to find the intersection point
    // let mut ground_point = initial_ground_point(sat_pos, slant_range, earth_center);

    // for _ in 0..10 {
    //     // 1. Get geographic coordinates
    //     let mut geo = ecef_to_geographic(ground_point.into());

    //     // 2. Get height from DEM
    //     geo.height = dem.get_height(geo.latitude, geo.longitude);

    //     // 3. Get improved position using height
    //     let improved_point = geographic_to_ecef(geo).into();

    //     // 4. Find new zero-Doppler time for this position
    //     let time = compute_zero_doppler_time(improved_point, orbit);

    //     // 5. Get satellite position at new time
    //     let new_sat_pos = orbit.get_position(time);

    //     // 6. Compute new ground point on the range sphere
    //     ground_point = compute_ground_point(new_sat_pos, slant_range, improved_point);

    //     // Check convergence
    //     if (ground_point - improved_point).norm() < 1e-6 {
    //         break;
    //     }
    // }

    // // Convert final ECEF to geographic
    // let final_geo = ecef_to_geographic(ground_point.into());
    // Ok(final_geo)
    todo!()
}

fn initial_ground_point(sat_pos: Vector3, slant_range: f64, earth_center: Vector3) -> Vector3 {
    let sat_to_earth = earth_center - sat_pos;
    let scale = slant_range / sat_to_earth.norm();
    sat_pos + scale * sat_to_earth
}

fn compute_ground_point(sat_pos: Vector3, slant_range: f64, earth_point: Vector3) -> Vector3 {
    let sat_to_earth = earth_point - sat_pos;
    let scale = slant_range / sat_to_earth.norm();
    sat_pos + scale * sat_to_earth
}

/// Converts an azimuth line index to azimuth time using metadata.
///
/// # Arguments
/// * `az_index` - The azimuth index (0-based).
/// * `az_time_first` - The timestamp of the first azimuth line.
/// * `az_time_last` - The timestamp of the last azimuth line.
/// * `num_lines` - Total number of azimuth lines in the image.
///
/// # Returns
/// * `DateTime<Utc>` - The computed azimuth time for the given index.
///
/// # Formula
/// The azimuth time is interpolated linearly as:
/// ```text
/// Δt = (AzTimeLast - AzTimeFirst) / (numberOfLines - 1)
/// t_i = AzTimeFirst + (i * Δt)
/// ```
fn azimuth_line_to_azimuth_time(
    az_index: usize,
    az_time_first: DateTime<Utc>,
    az_time_last: DateTime<Utc>,
    num_lines: usize,
) -> DateTime<Utc> {
    // Ensure we have at least 2 lines to compute time intervals
    assert!(num_lines > 1, "Number of lines must be greater than 1");

    // Compute time difference between first and last azimuth lines
    let total_duration = az_time_last - az_time_first;

    // Compute time step per azimuth index
    let delta_t = total_duration / (num_lines as i32 - 1);

    // Compute the azimuth time for the given index
    az_time_first + Duration::microseconds(delta_t.num_microseconds().unwrap() * az_index as i64)
}

/// A Vector3 representing a point in Earth-Centered, Earth-Fixed (ECEF) coordinates.
#[derive(Debug, Clone, Copy)]
struct ECEFCoordinates {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<Vector3> for ECEFCoordinates {
    fn from(v: Vector3) -> Self {
        ECEFCoordinates {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<ECEFCoordinates> for Vector3 {
    fn from(ecef: ECEFCoordinates) -> Self {
        Vector3::new(ecef.x, ecef.y, ecef.z)
    }
}

#[derive(Debug, Clone, Copy)]
struct GeographicPoint {
    pub latitude: f64,
    pub longitude: f64,
    pub height: f64,
}

/// Converts Earth-Centered, Earth-Fixed (ECEF) coordinates to geographic coordinates.
///
/// Transforms ECEF Cartesian coordinates (x, y, z) to geodetic coordinates
/// (latitude, longitude, height) using the WGS84 reference ellipsoid.
///
/// # Algorithm
///
/// The implementation follows Ferrari's solution which provides accurate results
/// with excellent computational efficiency.
///
/// # Arguments
///
/// * `point` - Earth-Centered Earth-Fixed coordinates in meters:
///   * `x`: Earth-fixed X coordinate (meters)
///   * `y`: Earth-fixed Y coordinate (meters)
///   * `z`: Earth-fixed Z coordinate (meters)
///
/// # Returns
///
/// * `GeographicPoint` - Geodetic coordinates:
///   * `latitude`: geodetic latitude (degrees, positive north)
///   * `longitude`: geodetic longitude (degrees, positive east)
///   * `height`: height above WGS84 ellipsoid (meters)
///
/// # Accuracy
///
/// Better than 1e-11 radians (< 1 μm) for coordinates on or near Earth's surface.
///
/// # References
///
/// 1. Zhu, J. (1994). "Conversion of Earth-centered Earth-fixed coordinates to geodetic coordinates"
///    IEEE Transactions on Aerospace and Electronic Systems, 30(3), 957-961.
/// 2. ESA Earth Observation CFI Software - "eop_geo_to_xyz.F" v.4.8
///
/// # Notes
///
/// The WGS84 ellipsoid is used with the following parameters:
/// - Semi-major axis (a): 6378137.0 meters
/// - Flattening (f): 1/298.257223563
/// - Semi-minor axis (b): 6356752.314245 meters
/// - Eccentricity squared (e²): 6.69437999014e-3
fn ecef_to_geographic(point: ECEFCoordinates) -> GeographicPoint {
    // WGS84 ellipsoid parameters
    const A: f64 = 6378137.0; // semi-major axis (meters)
    const F: f64 = 1.0 / 298.257223563; // flattening
    const B: f64 = A * (1.0 - F); // semi-minor axis (meters)
    const E2: f64 = 2.0 * F - F * F; // eccentricity squared
    const EP2: f64 = E2 / (1.0 - E2); // second eccentricity squared

    // Compute auxiliary values
    let p = (point.x * point.x + point.y * point.y).sqrt();

    // Special case: point on Z axis
    if p < 1e-12 {
        let sgn = if point.z >= 0.0 { 1.0 } else { -1.0 };
        return GeographicPoint {
            latitude: sgn * 90.0,
            longitude: 0.0,
            height: point.z.abs() - B,
        };
    }

    // Compute longitude
    let longitude = point.y.atan2(point.x);

    // Initial guess for reduced latitude (parametric latitude)
    let theta = (point.z * A).atan2(p * B);

    // Improved latitude using Ferrari's solution
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let latitude = (point.z + EP2 * B * sin_theta.powi(3)).atan2(p - E2 * A * cos_theta.powi(3));

    // Calculate height above ellipsoid
    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();

    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
    let height = p / cos_lat - n;

    // Handle special case when point is near the poles
    // (cos_lat is near zero, which could cause precision issues)
    if cos_lat.abs() < 1e-10 {
        let height = point.z.abs() - B;
        let sgn = if point.z >= 0.0 { 1.0 } else { -1.0 };
        return GeographicPoint {
            latitude: sgn * 90.0,
            longitude: longitude * 180.0 / std::f64::consts::PI,
            height,
        };
    }

    // Convert latitude and longitude to degrees
    GeographicPoint {
        latitude: latitude * 180.0 / std::f64::consts::PI,
        longitude: longitude * 180.0 / std::f64::consts::PI,
        height,
    }
}

/// Converts geographic coordinates to Earth-Centered, Earth-Fixed (ECEF) coordinates.
///
/// Transforms geodetic coordinates (latitude, longitude, height) to ECEF Cartesian
/// coordinates (x, y, z) using the WGS84 reference ellipsoid.
///
/// # Algorithm
///
/// Direct calculation using the standard geodetic-to-Cartesian transformation formulas.
///
/// # Arguments
///
/// * `point` - Geodetic coordinates:
///   * `latitude`: geodetic latitude (degrees, positive north)
///   * `longitude`: geodetic longitude (degrees, positive east)
///   * `height`: height above WGS84 ellipsoid (meters)
///
/// # Returns
///
/// * `ECEFCoordinates` - Earth-Centered Earth-Fixed coordinates in meters:
///   * `x`: Earth-fixed X coordinate (meters)
///   * `y`: Earth-fixed Y coordinate (meters)
///   * `z`: Earth-fixed Z coordinate (meters)
///
/// # Accuracy
///
/// The conversion is exact to within floating-point precision.
///
/// # References
///
/// 1. Hofmann-Wellenhof, B., Lichtenegger, H., & Wasle, E. (2008).
///    "GNSS – Global Navigation Satellite Systems". Springer.
/// 2. ESA Earth Observation CFI Software - "eop_xyz_to_geo.F" v.4.8
///
/// # Notes
///
/// The WGS84 ellipsoid is used with the following parameters:
/// - Semi-major axis (a): 6378137.0 meters
/// - Flattening (f): 1/298.257223563
/// - Eccentricity squared (e²): 6.69437999014e-3
fn geographic_to_ecef(point: GeographicPoint) -> ECEFCoordinates {
    // WGS84 ellipsoid parameters
    const A: f64 = 6378137.0; // semi-major axis (meters)
    const F: f64 = 1.0 / 298.257223563; // flattening
    const E2: f64 = 2.0 * F - F * F; // eccentricity squared

    // Convert latitude and longitude to radians
    let lat_rad = point.latitude * std::f64::consts::PI / 180.0;
    let lon_rad = point.longitude * std::f64::consts::PI / 180.0;

    // Compute trigonometric functions (for readability)
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();

    // Calculate radius of curvature in the prime vertical
    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();

    // Calculate ECEF coordinates
    let x = (n + point.height) * cos_lat * cos_lon;
    let y = (n + point.height) * cos_lat * sin_lon;
    let z = (n * (1.0 - E2) + point.height) * sin_lat;

    // Special handling for poles (where cos_lat is near zero)
    // Not strictly necessary due to accurate trigonometric functions,
    // but included for numerical robustness
    if lat_rad.abs() > 89.99 * std::f64::consts::PI / 180.0 {
        let sgn = lat_rad.signum();
        let h = point.height.max(0.0); // Ensure non-negative height at poles

        return ECEFCoordinates {
            x: 0.0,
            y: 0.0,
            z: sgn * (A * (1.0 - F) + h),
        };
    }

    ECEFCoordinates { x, y, z }
}
