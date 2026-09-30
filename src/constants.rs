//! Constants used in the library.

/// Speed of light (and other electromagnetic waves) in m/s.
pub const C_LIGHT: f64 = 299_792_458.0;
/// Sentinel-1 radar carrier frequency in Hz (`radarFrequency` in the product annotation).
pub const SENTINEL_1_RADAR_FREQUENCY: f64 = 5.405_000_454_334_350e9;
/// Wavelength of the Sentinel-1 microwave radiation in meters (~0.05546576 m).
pub const SENTINEL_1_WAVELENGTH: f64 = C_LIGHT / SENTINEL_1_RADAR_FREQUENCY;
