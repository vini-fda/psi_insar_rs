use std::f32;

use nalgebra::{ComplexField, Vector3};
use ndarray::Array2;

use crate::{
    dem::DEM,
    metadata::annotation_xml::SlcProductAnnotation,
    satellite_orbit::{OrbitalStateHistory, radar_coords_slc_annotation_to_pixel_f32},
    sentinel::Sentinel1SlcBurst,
};

pub mod coarse_coregistration;
pub mod dem_assisted_coregistration;

/// Computes the warp function \rho between two SLC images, in the domain of the reference image.
pub fn compute_warp_function(
    reference: &Sentinel1SlcBurst,
    secondary: &Sentinel1SlcBurst,
    dem: &DEM,
) -> Array2<[u8; 4]> {
    let (cols, rows) = reference.data.raster_size();
    let ref_osh = reference.orbital_state_history();
    let sec_osh = secondary.orbital_state_history();
    let radar_coords = |ground_target_pos: Vector3<f64>,
                        osh: &OrbitalStateHistory,
                        annotation: &SlcProductAnnotation|
     -> [f32; 2] {
        let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
        let (row, col) = radar_coords_slc_annotation_to_pixel_f32(zero_doppler, annotation);
        [row, col]
    };
    let mut rho = Array2::<[f32; 2]>::default((rows, cols));
    for (_, _, lat, lon, _) in dem.indexed_lat_lon_height() {
        let pos = dem.get_ecef_at_lat_lon(lat, lon);
        let rc_ref = radar_coords(pos.into(), &ref_osh, &reference.metadata);
        let rc_sec = radar_coords(pos.into(), &sec_osh, &secondary.metadata);

        if (rc_ref[0] >= 0.0 && rc_ref[0] < rows as f32)
            && (rc_ref[1] >= 0.0 && rc_ref[1] < cols as f32)
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

#[cfg(test)]
mod tests {
    use ndarray::{Dimension, s};

    use crate::{dem::DEM, sentinel::Sentinel1SlcBurst};

    use super::compute_warp_function;

    #[test]
    fn test_warp_function() {
        let reference = Sentinel1SlcBurst::load_from_directory(
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
        let rho = compute_warp_function(&reference, &secondary, &dem);
        let (rows, cols) = rho.dim();
        let rho = rho.slice(s![0..rows, 0..cols / 2]).to_owned();

        let rr = rerun::RecordingStreamBuilder::new("test_warp_function")
            .connect_tcp()
            .expect("Could not connect to local Rerun instance.");
        let (rows, cols) = rho.dim();
        let (v, offset) = rho.into_raw_vec_and_offset();
        let img = rerun::Image::from_rgba32(v.as_flattened(), [cols as u32, rows as u32]);
        rr.log("warp_fn", &img).expect("Could not finish recording");
    }
}
