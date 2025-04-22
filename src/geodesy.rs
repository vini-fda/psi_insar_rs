//! A module for converting between geodetic coordinates (latitude, longitude, height) and
//! Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates.
//!
//! This uses the WGS84 ellipsoid parameters and only the Rust standard library (`std`).

/// Semi-major axis of the WGS84 ellipsoid (meters).
const A: f64 = 6_378_137.0;
/// Flattening of the WGS84 ellipsoid.
const F: f64 = 1.0 / 298.257_223_563;
/// Square of eccentricity: e² = f(2 − f).
const E2: f64 = F * (2.0 - F);

/// Converts geodetic coordinates (latitude, longitude, ellipsoidal height) to
/// ECEF coordinates (x, y, z).
///
/// # Arguments
///
/// * `lat_deg` - Geodetic latitude in degrees.
/// * `lon_deg` - Geodetic longitude in degrees.
/// * `h` - Ellipsoidal height above the WGS84 reference ellipsoid, in meters.
///
/// # Returns
///
/// A tuple `(x, y, z)` representing the ECEF coordinates in meters.
///
/// # Example
///
/// ```rust
/// use psi_insar_rs::geodesy::geodetic_to_ecef;
/// let (x, y, z) = geodetic_to_ecef(52.5, 13.4, 140.2);
/// println!("ECEF = ({:.3}, {:.3}, {:.3})", x, y, z);
/// ```
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, h: f64) -> [f64; 3] {
    // Convert degrees to radians
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    // Radius of curvature in the prime vertical
    let n = A / (1.0 - E2 * lat.sin().powi(2)).sqrt();

    let cos_lat = lat.cos();
    let cos_lon = lon.cos();
    let sin_lat = lat.sin();
    let sin_lon = lon.sin();

    // Compute ECEF
    let x = (n + h) * cos_lat * cos_lon;
    let y = (n + h) * cos_lat * sin_lon;
    let z = (n * (1.0 - E2) + h) * sin_lat;

    [x, y, z]
}

/// Converts ECEF coordinates (x, y, z) back to geodetic coordinates
/// (latitude, longitude, ellipsoidal height) using an iterative method.
///
/// # Arguments
///
/// * `x` - ECEF X coordinate in meters.
/// * `y` - ECEF Y coordinate in meters.
/// * `z` - ECEF Z coordinate in meters.
///
/// # Returns
///
/// A tuple `(lat_deg, lon_deg, h)`:
/// * `lat_deg` - Geodetic latitude in degrees.
/// * `lon_deg` - Geodetic longitude in degrees.
/// * `h` - Ellipsoidal height above the WGS84 ellipsoid, in meters.
///
/// # Example
///
/// ```rust
/// use psi_insar_rs::geodesy::ecef_to_geodetic;
/// let (lat, lon, h) = ecef_to_geodetic(-912052.75, -5952183.0, 2107839.5);
/// println!("Lat = {:.6}°, Lon = {:.6}°, Height = {:.3} m", lat, lon, h);
/// ```
pub fn ecef_to_geodetic(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Longitude via atan2
    let lon = y.atan2(x);

    // Distance from Z-axis
    let p = (x.powi(2) + y.powi(2)).sqrt();

    // Initial latitude estimate
    let mut lat = z.atan2(p * (1.0 - E2));
    let mut h = 0.0_f64;
    let mut n;

    // Iterate to improve latitude and height
    for _ in 0..5 {
        let sin_lat = lat.sin();
        n = A / (1.0 - E2 * sin_lat.powi(2)).sqrt();
        h = p / lat.cos() - n;
        lat = z.atan2(p * (1.0 - E2 * (n / (n + h))));
    }

    // Convert to degrees
    let lat_deg = lat.to_degrees();
    let lon_deg = lon.to_degrees();

    (lat_deg, lon_deg, h)
}
