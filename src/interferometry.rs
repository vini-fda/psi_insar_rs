use crate::{
    constants::SENTINEL_1_WAVELENGTH, dem::DEM, perp_baseline::EnhancedDelaunayWarpFunction,
    satellite_orbit::zero_doppler_time, sentinel::Sentinel1SlcProduct,
};
use nalgebra::Vector3;
use ndarray::{Array2, Axis};
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

pub fn coregister_and_remove_flat_phase(
    reference: &Sentinel1SlcProduct,
    secondaries: &[Sentinel1SlcProduct],
    dem: &DEM,
) {
    for secondary in secondaries {
        log::info!("Computing warp function");
        let start_time = std::time::Instant::now();
        let warp_function = EnhancedDelaunayWarpFunction::new(reference, secondary, dem);
        let end_time = std::time::Instant::now();
        log::info!("Time taken: {:?}", end_time - start_time);

        let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();
        // let mut coregistered_secondary_img = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));

        // let kernel = KnabSincKernel::default();
        // let deramp = DerampSlcBurst::new();

        // log::info!("Deramping reference and secondary images");
        // let start_time = std::time::Instant::now();
        // let reference_img = deramp.apply_forward(&reference);
        // let secondary_img = deramp.apply_forward(&secondary);
        // let end_time = std::time::Instant::now();
        // log::info!("Time taken: {:?}", end_time - start_time);

        // log::info!("Resampling secondary image to reference image via warp function");
        // let start_time = std::time::Instant::now();
        const CHUNK_SIZE: usize = 256;
        // coregistered_secondary_img
        //     .axis_chunks_iter_mut(Axis(0), CHUNK_SIZE)
        //     .into_par_iter()
        //     .enumerate()
        //     .for_each_init(
        //         || warp_function.triangulation.natural_neighbor(),
        //         |nn, (chunk_idx, mut chunk)| {
        //             let az_offset = chunk_idx * CHUNK_SIZE;

        //             for (i, mut row) in chunk.outer_iter_mut().enumerate() {
        //                 let ref_az = az_offset + i;

        //                 for (ref_rg, value) in row.iter_mut().enumerate() {
        //                     let ref_coords = [ref_az as f64, ref_rg as f64];
        //                     let compute_mapped_coord = |dimension: usize| {
        //                         nn.interpolate(
        //                             |v| v.data().secondary_coords[dimension],
        //                             ref_coords.into(),
        //                         )
        //                     };
        //                     if let (Some(sec_az), Some(sec_rg)) =
        //                         (compute_mapped_coord(0), compute_mapped_coord(1))
        //                     {
        //                         let v = interpolate_2d(
        //                             secondary_img.view(),
        //                             sec_az as f32,
        //                             sec_rg as f32,
        //                             &kernel,
        //                         );
        //                         *value = v;
        //                     }
        //                 }
        //             }
        //         },
        //     );
        // let end_time = std::time::Instant::now();
        // log::info!("Time taken: {:?}", end_time - start_time);

        let osh_1 = reference.orbital_state_history();
        let annotation_1 = &reference.metadata;
        // let s = annotation_1
        //     .image_annotation
        //     .image_information
        //     .range_pixel_spacing;
        let osh_2 = secondary.orbital_state_history();
        let annotation_2 = &secondary.metadata;
        let mut phase_diff = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));
        // for i in 0..ref_azimuth_dim {
        //     for j in 0..ref_slant_range_dim {
        //         let s1 = reference_img[[i, j]];
        //         let s2 = coregistered_secondary_img[[i, j]];
        //         phase_diff[[i, j]] = (s1 * s2.conj()).arg();
        //     }
        // }

        log::info!("Removing topographic phase");
        let start_time = std::time::Instant::now();
        phase_diff
            .axis_chunks_iter_mut(Axis(0), CHUNK_SIZE)
            .into_par_iter()
            .enumerate()
            .for_each_init(
                || warp_function.triangulation.natural_neighbor(),
                |nn, (chunk_idx, mut chunk)| {
                    let az_offset = chunk_idx * CHUNK_SIZE;

                    for (i, mut row) in chunk.outer_iter_mut().enumerate() {
                        let ref_az = az_offset + i;

                        for (ref_rg, phase) in row.iter_mut().enumerate() {
                            let ref_coords = [ref_az as f64, ref_rg as f64];
                            let sec_az =
                                nn.interpolate(|v| v.data().secondary_coords[0], ref_coords.into());

                            // Now compute the ground target position
                            let ground_target_lat =
                                nn.interpolate(|v| v.data().lat, ref_coords.into());
                            let ground_target_lon =
                                nn.interpolate(|v| v.data().lon, ref_coords.into());

                            // if any are None, skip
                            if sec_az.is_none()
                                || ground_target_lat.is_none()
                                || ground_target_lon.is_none()
                            {
                                continue;
                            }
                            let sec_az = sec_az.unwrap();
                            let ground_target_lat = ground_target_lat.unwrap();
                            let ground_target_lon = ground_target_lon.unwrap();

                            let ground_target_pos = Vector3::from(
                                dem.get_ecef_at_lat_lon(ground_target_lat, ground_target_lon),
                            );

                            let zero_doppler_1 = zero_doppler_time(ref_az as f64, annotation_1);
                            let zero_doppler_2 = zero_doppler_time(sec_az, annotation_2);
                            let (s_1, _) = osh_1.interp_pos_vel(zero_doppler_1);
                            let (s_2, _) = osh_2.interp_pos_vel(zero_doppler_2);

                            let r1 = (s_1 - ground_target_pos).norm();
                            let r2 = (s_2 - ground_target_pos).norm();

                            let delta_phi =
                                4.0 * std::f64::consts::PI * (r2 - r1) / SENTINEL_1_WAVELENGTH;
                            *phase = delta_phi as f32;
                        }
                    }
                },
            );
        let end_time = std::time::Instant::now();
        log::info!("Time taken: {:?}", end_time - start_time);

        let secondary_name = &secondary.granule_id.raw_filename;
        let file = std::fs::File::create(format!("phase_diff_{secondary_name}.npy")).unwrap();
        phase_diff
            .write_npy(file)
            .expect("Failed to write npy file");
    }
}

/// Compute the bounding box of a stack of Sentinel1SlcBurst images,
/// with a small margin to account for the fact that the images are not
/// exactly aligned.
pub fn bounding_box_from_stack<'a, I: IntoIterator<Item = &'a Sentinel1SlcProduct>>(
    stack: I,
) -> [f64; 4] {
    const OFFSET_LAT: f64 = 0.05;
    const OFFSET_LON: f64 = 0.05;
    let mut min_lat_final = f64::MAX;
    let mut max_lat_final = f64::MIN;
    let mut min_lon_final = f64::MAX;
    let mut max_lon_final = f64::MIN;
    for burst in stack {
        let [min_lat, max_lat, min_lon, max_lon] = burst
            .metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .get_bounding_box_lat_lon();
        if min_lat < min_lat_final {
            min_lat_final = min_lat;
        }
        if max_lat > max_lat_final {
            max_lat_final = max_lat;
        }
        if min_lon < min_lon_final {
            min_lon_final = min_lon;
        }
        if max_lon > max_lon_final {
            max_lon_final = max_lon;
        }
    }
    [
        min_lat_final - OFFSET_LAT,
        max_lat_final + OFFSET_LAT,
        min_lon_final - OFFSET_LON,
        max_lon_final + OFFSET_LON,
    ]
}
