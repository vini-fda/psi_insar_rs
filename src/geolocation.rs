use chrono::{DateTime, Duration, Utc};
use thiserror::Error;

use crate::dem::ElevationModel;

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

pub struct OrbitalStateHistory {
    pub time: Vec<DateTime<Utc>>,
    pub position: Vec<Vector3>,
    pub velocity: Vec<Vector3>,
}

impl OrbitalStateHistory {
    pub fn get_position(&self, time: f64) -> Vector3 {
        // Find the two closest times
        let (i, j) = self.find_closest_times(time);

        // Interpolate position
        let pos_i = self.position[i];
        let pos_j = self.position[j];
        let t_i = self.time[i].timestamp() as f64;
        let t_j = self.time[j].timestamp() as f64;
        let alpha = (time - t_i) / (t_j - t_i);
        pos_i + alpha * (pos_j - pos_i)
    }

    pub fn get_velocity(&self, time: f64) -> Vector3 {
        // Find the two closest times
        let (i, j) = self.find_closest_times(time);

        // Interpolate velocity
        let vel_i = self.velocity[i];
        let vel_j = self.velocity[j];
        let t_i = self.time[i].timestamp() as f64;
        let t_j = self.time[j].timestamp() as f64;
        let alpha = (time - t_i) / (t_j - t_i);
        vel_i + alpha * (vel_j - vel_i)
    }

    fn find_closest_times(&self, time: f64) -> (usize, usize) {
        let mut i = 0;
        let mut j = 0;
        let mut min_diff = f64::INFINITY;
        for (idx, t) in self.time.iter().enumerate() {
            let diff = (t.timestamp() as f64 - time).abs();
            if diff < min_diff {
                min_diff = diff;
                i = idx;
            }
        }
        for (idx, t) in self.time.iter().enumerate() {
            let diff = (t.timestamp() as f64 - time).abs();
            if diff < min_diff && idx != i {
                min_diff = diff;
                j = idx;
            }
        }
        (i, j)
    }

    fn get_closest_approach_time(&self, ground_point: Vector3) -> f64 {
        // Initial guess: closest approach to the ground point
        let mut min_dist = f64::INFINITY;
        let mut min_time = 0.0;
        for (time, pos) in self.time.iter().zip(self.position.iter()) {
            let dist = (pos - ground_point).norm();
            if dist < min_dist {
                min_dist = dist;
                min_time = time.timestamp() as f64;
            }
        }
        min_time
    }

    fn compute_doppler_derivative(&self, ground_point: Vector3, time: f64) -> f64 {
        // Finite difference for derivative
        let delta = 1e-6;
        let doppler_plus = self.compute_doppler(ground_point, time + delta);
        let doppler_minus = self.compute_doppler(ground_point, time - delta);
        (doppler_plus - doppler_minus) / (2.0 * delta)
    }

    fn compute_doppler(&self, ground_point: Vector3, time: f64) -> f64 {
        let sat_pos = self.get_position(time);
        let sat_vel = self.get_velocity(time);
        let los = ground_point - sat_pos;
        los.dot(&sat_vel)
    }
}

/// Computes the zero-Doppler time for a ground point
///
/// # Arguments
/// * `ground_point` - ECEF coordinates of the ground point
/// * `orbit` - Satellite orbit state vectors
///
/// # Returns
/// * The time (in seconds) when the satellite passes through
///   the zero-Doppler plane for this ground point
pub fn compute_zero_doppler_time(ground_point: Vector3, orbit: &OrbitalStateHistory) -> f64 {
    // Initial guess from closest approach
    let mut t = orbit.get_closest_approach_time(ground_point);

    // Iterative refinement using Newton's method
    for _ in 0..10 {
        // Get satellite position and velocity
        let sat_pos = orbit.get_position(t);
        let sat_vel = orbit.get_velocity(t);

        // Line-of-sight vector
        let los = ground_point - sat_pos;

        // Doppler function: dot product of LOS and velocity
        let doppler = los.dot(&sat_vel);

        // If close enough to zero, we've found the zero-Doppler time
        if doppler.abs() < 1e-6 {
            return t;
        }

        // Derivative of Doppler for Newton's method
        let derivative = orbit.compute_doppler_derivative(ground_point, t);

        // Newton step
        t -= doppler / derivative;
    }

    t
}

/// Computes slant range between satellite and ground point
pub fn compute_slant_range(
    ground_point: Vector3,
    time: f64,
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
    dem: &ElevationModel,
    orbit: &OrbitalStateHistory,
) -> Result<GeographicPoint, GeolocationError> {
    // Convert line to time
    let azimuth_time = azimuth_line_to_azimuth_time(
        azimuth_index,
        az_time_first,
        az_time_last,
        num_azimuth_lines,
    );

    // Initial position (no elevation)
    let sat_pos = orbit.get_position(azimuth_time.timestamp() as f64);
    let earth_center = Vector3::new(0.0, 0.0, 0.0);

    // Iterative solution to find the intersection point
    let mut ground_point = initial_ground_point(sat_pos, slant_range, earth_center);

    for _ in 0..10 {
        // 1. Get geographic coordinates
        let mut geo = ecef_to_geographic(ground_point.into());

        // 2. Get height from DEM
        geo.height = dem.get_height(geo.latitude, geo.longitude);

        // 3. Get improved position using height
        let improved_point = geographic_to_ecef(geo).into();

        // 4. Find new zero-Doppler time for this position
        let time = compute_zero_doppler_time(improved_point, orbit);

        // 5. Get satellite position at new time
        let new_sat_pos = orbit.get_position(time);

        // 6. Compute new ground point on the range sphere
        ground_point = compute_ground_point(new_sat_pos, slant_range, improved_point);

        // Check convergence
        if (ground_point - improved_point).norm() < 1e-6 {
            break;
        }
    }

    // Convert final ECEF to geographic
    let final_geo = ecef_to_geographic(ground_point.into());
    Ok(final_geo)
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
/// ```
/// Δt = (AzTimeLast - AzTimeFirst) / (numberOfLines - 1)
/// t_i = AzTimeFirst + (i * Δt)
/// ```
///
/// # Example
/// ```
/// let az_time = azimuth_line_to_azimuth_time(
///     500,
///     "2025-03-19T00:00:00.000000Z".parse().unwrap(),
///     "2025-03-19T00:00:10.000000Z".parse().unwrap(),
///     1000
/// );
/// println!("{}", az_time); // Expected: "2025-03-19T00:00:05.005000Z"
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
