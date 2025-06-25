use crate::{
    constants::SENTINEL_1_WAVELENGTH,
    coregistration::{
        deramping::DerampSlcBurst,
        interpolation2d::{KnabSincKernel, interpolate_2d},
    },
    dem::DEM,
    geodesy::local_normal,
    perp_baseline::{EnhancedDelaunayWarpFunction, perp_baseline},
    satellite_orbit::{pixel_coords_to_radar_coords, zero_doppler_time},
    sentinel::Sentinel1SlcBurst,
};
use nalgebra::Vector3;
use ndarray::{Array2, Axis};
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

pub fn coregister_and_remove_flat_phase(
    reference: &Sentinel1SlcBurst,
    secondaries: &[Sentinel1SlcBurst],
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
                            let zero_doppler_2 = zero_doppler_time(sec_az as f64, annotation_2);
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
        let file = std::fs::File::create(format!("phase_diff_{}.npy", secondary_name)).unwrap();
        phase_diff
            .write_npy(file)
            .expect("Failed to write npy file");
    }
}

/// Compute the bounding box of a stack of Sentinel1SlcBurst images,
/// with a small margin to account for the fact that the images are not
/// exactly aligned.
pub fn bounding_box_from_stack<'a, I: IntoIterator<Item = &'a Sentinel1SlcBurst>>(
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

#[cfg(test)]
mod tests {
    use ndarray::s;
    use rerun::{ColorModel, Image};

    use crate::{dem::CopernicusDemType, satellite_orbit::OrbitalStateHistory};

    use super::*;

    #[test]
    fn test_stack_interferograms() {
        env_logger::init();
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondaries = [
            "download/S1_305967_IW3_20150916T122546_VV_8302-BURST",
            "download/S1_305967_IW3_20150928T122546_VV_5407-BURST",
            "download/S1_305967_IW3_20151010T122546_VV_7501-BURST",
            "download/S1_305967_IW3_20151103T122546_VV_AE93-BURST",
            "download/S1_305967_IW3_20151115T122546_VV_8956-BURST",
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        ]
        .iter()
        .map(|name| Sentinel1SlcBurst::load_first_from_directory(name).unwrap())
        .collect::<Vec<_>>();
        let all_bursts = std::iter::once(&reference).chain(&secondaries);
        let bounding_box = bounding_box_from_stack(all_bursts.clone());
        println!("Bounding box: {:?}", bounding_box);
        let dem = DEM::download_dem(bounding_box, CopernicusDemType::Cop30);
        coregister_and_remove_flat_phase(&reference, &secondaries, &dem);
        // let rr = rerun::RecordingStreamBuilder::new("test_stack_interferograms")
        //     .connect_grpc()
        //     .expect("Could not connect to local Rerun instance.");
        // for burst in all_bursts {
        //     let name = &burst.granule_id.raw_filename;
        //     let [min_lat, max_lat, min_lon, max_lon] = burst
        //         .metadata
        //         .geolocation_grid
        //         .geolocation_grid_point_list
        //         .get_bounding_box_lat_lon();
        //     let corners = [
        //         [min_lat, min_lon],
        //         [min_lat, max_lon],
        //         [max_lat, max_lon],
        //         [max_lat, min_lon],
        //         [min_lat, min_lon],
        //     ];
        //     rr.log(
        //         format!("{}", name),
        //         &rerun::GeoLineStrings::from_lat_lon([corners.windows(2).flatten()])
        //             .with_radii([rerun::Radius::new_ui_points(2.0)])
        //             .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        //     )
        //     .unwrap();
        // }
    }

    #[test]
    fn baseline_plot() {
        env_logger::init();
        let rr = rerun::RecordingStreamBuilder::new("test_baseline_plot")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondaries = [
            // "download/S1_305967_IW3_20150916T122546_VV_8302-BURST",
            // "download/S1_305967_IW3_20150928T122546_VV_5407-BURST",
            // "download/S1_305967_IW3_20151010T122546_VV_7501-BURST",
            // "download/S1_305967_IW3_20151103T122546_VV_AE93-BURST",
            // "download/S1_305967_IW3_20151115T122546_VV_8956-BURST",
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        ]
        .iter()
        .map(|name| Sentinel1SlcBurst::load_first_from_directory(name).unwrap())
        .collect::<Vec<_>>();
        let all_bursts = std::iter::once(&reference).chain(&secondaries);
        let bounding_box = bounding_box_from_stack(all_bursts.clone());
        println!("Bounding box: {:?}", bounding_box);
        let dem = DEM::download_dem(bounding_box, CopernicusDemType::Cop30);
        let start_time_ref = reference.metadata.ads_header.start_time;

        // Log reference sat position
        let orbit_list = &reference.metadata.general_annotation.orbit_list;
        let orbital_history = OrbitalStateHistory::from(orbit_list).interp_n(4);
        let points = orbital_history
            .position
            .iter()
            .map(|v| v.map(|x| x as f32).data.0[0]);

        rr.log_static(
            "ref_orbital_positions",
            &rerun::Points3D::new(points).with_radii([200.0]),
        )
        .unwrap();
        for secondary in secondaries {
            let start_time_sec = secondary.metadata.ads_header.start_time;
            let delta_time = start_time_sec - start_time_ref;
            log::info!("Processing secondary {}", secondary.granule_id.raw_filename);
            log::info!(
                "Days between secondary acquisition and reference: {}",
                delta_time.num_days()
            );
            // Log sec sat positions
            let orbit_list = &secondary.metadata.general_annotation.orbit_list;
            let orbital_history = OrbitalStateHistory::from(orbit_list).interp_n(4);
            let points = orbital_history
                .position
                .iter()
                .map(|v| v.map(|x| x as f32).data.0[0]);

            rr.log_static(
                "sec_orbital_positions",
                &rerun::Points3D::new(points).with_radii([200.0]),
            )
            .unwrap();
            let start_time = std::time::Instant::now();
            let warp_function = EnhancedDelaunayWarpFunction::new(&reference, &secondary, &dem);
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();

            const CHUNK_SIZE: usize = 2000;

            let osh_1 = reference.orbital_state_history();
            let annotation_1 = &reference.metadata;
            let osh_2 = secondary.orbital_state_history();
            let annotation_2 = &secondary.metadata;
            let mut phase_diff = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim / 2));

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
                                let sec_az = nn.interpolate(
                                    |v| v.data().secondary_coords[0],
                                    ref_coords.into(),
                                );

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
                                let zero_doppler_2 = zero_doppler_time(sec_az as f64, annotation_2);
                                let (s_1, v_1) = osh_1.interp_pos_vel(zero_doppler_1);
                                let (s_2, v_2) = osh_2.interp_pos_vel(zero_doppler_2);
                                let n = (s_1 - ground_target_pos).normalize();
                                let b_parallel = (s_1 - s_2).dot(&n);
                                let b_perp = s_1 - s_2 - b_parallel * n;

                                *phase = b_perp.norm() as f32;
                            }
                        }
                    },
                );
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            // let mut phase_diff = phase_diff
            //     .slice(s![.., 0..ref_slant_range_dim / 2])
            //     .to_owned();
            // let max = *phase_diff
            //     .iter()
            //     .max_by(|a, b| a.partial_cmp(b).unwrap())
            //     .unwrap();
            // phase_diff.mapv_inplace(|v| v / max);
            let tensor =
                rerun::Tensor::try_from(phase_diff.clone()).expect("Unable to create tensor.");
            rr.log("perp_baseline", &tensor)
                .expect("Unable to log tensor.");
            let img = Image::from_color_model_and_tensor(ColorModel::L, phase_diff)
                .expect("Could not load img.");
            rr.log("perp_baseline", &img).expect("Unable to log img");
        }
    }
}
