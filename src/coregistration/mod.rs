use std::f32;

use nalgebra::Vector3;
use ndarray::Array2;

use crate::{
    dem::DEM,
    metadata::annotation_xml::SlcProductAnnotation,
    satellite_orbit::{OrbitalStateHistory, radar_coords_to_pixel_coords},
    sentinel::Sentinel1SlcIWSwath,
};

pub mod coarse_coregistration;
pub mod dem_assisted_coregistration;
pub mod deramping;
pub mod interpolation2d;
pub mod layover_detection;
pub mod spectrum;
pub mod warp_function;

/// Computes the warp function \rho between two SLC images, in the domain of the reference image.
pub fn compute_warp_function(
    reference: &Sentinel1SlcIWSwath,
    secondary: &Sentinel1SlcIWSwath,
    dem: &DEM,
) -> Array2<[u8; 4]> {
    let [slant_range_size, azimuth_size] = reference.data.raster_size();
    let ref_osh = reference.orbital_state_history();
    let sec_osh = secondary.orbital_state_history();
    let radar_coords = |ground_target_pos: Vector3<f64>,
                        osh: &OrbitalStateHistory,
                        annotation: &SlcProductAnnotation|
     -> [f32; 2] {
        let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
        radar_coords_to_pixel_coords(zero_doppler, annotation)
    };
    let mut rho = Array2::<[f32; 2]>::default((azimuth_size, slant_range_size));
    for (lat, lon) in dem.lat_lon_iter() {
        let pos = dem.get_ecef_at_lat_lon(lat, lon);
        let rc_ref = radar_coords(pos.into(), &ref_osh, &reference.metadata);
        let rc_sec = radar_coords(pos.into(), &sec_osh, &secondary.metadata);

        if (rc_ref[0] >= 0.0 && rc_ref[0] < azimuth_size as f32)
            && (rc_ref[1] >= 0.0 && rc_ref[1] < slant_range_size as f32)
        {
            let displacement = [rc_ref[0] - rc_sec[0], rc_ref[1] - rc_sec[1]];
            let i = rc_ref[0].floor() as usize;
            let j = rc_ref[1].floor() as usize;
            rho[[i, j]] = displacement;
        }
    }

    let norms = rho.map(|x| (x[0] * x[0] + x[1] * x[1]).sqrt());
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    for norm in norms {
        if norm > max {
            max = norm;
        } else if norm < min {
            min = norm;
        }
    }

    rho.map(|x| x.map(|e| (e - min) / (max - min)))
        .map(vector_to_rgba_u32)
}

#[inline] // Suggest inlining for performance in tight loops
pub fn vector_to_rgba_u32(vector: &[f32; 2]) -> [u8; 4] {
    // Extract components
    let x = vector[0];
    let y = vector[1];
    if x == 0.0 && y == 0.0 {
        return [0, 0, 0, 0];
    }

    // Map [-1.0, 1.0] range to [0.0, 1.0] range
    // let x_norm = (x + 1.0) * 0.5;
    // let y_norm = (y + 1.0) * 0.5;
    let x_norm = x * 0.5;
    let y_norm = y * 0.5;

    // Scale to [0.0, 255.0] range, clamp to ensure validity, and cast to u8
    let r = (x_norm * 255.0).round().clamp(0.0, 255.0) as u8;
    let g = (y_norm * 255.0).round().clamp(0.0, 255.0) as u8;
    let b: u8 = (-(x_norm + y_norm) / 2.0f32.sqrt() * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8; // Blue channel is fixed at 0
    let a: u8 = 255; // Alpha channel is fixed at 255 (fully opaque)

    // Combine into a u32 integer: 0xRRGGBBAA
    // R is shifted to the most significant byte (bits 24-31)
    // G is shifted to the next byte (bits 16-23)
    // B is shifted to the next byte (bits 8-15)
    // A is in the least significant byte (bits 0-7)
    // ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32)
    [r, g, b, a]
}
