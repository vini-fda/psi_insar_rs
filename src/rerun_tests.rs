#[cfg(test)]
mod tests {
    use chrono::Utc;
    use nalgebra::{Complex, Vector3};
    use ndarray::{Array2, ArrayView2, Axis, s};
    use num_complex::ComplexFloat;
    use rayon::prelude::*;
    use rerun::{ColorModel, Image, RecordingStream};

    use crate::{
        constants::SENTINEL_1_WAVELENGTH,
        coregistration::{
            deramping::DerampSlcBurst,
            interpolation2d::{KnabSincKernel, interpolate_2d},
        },
        dem::{CopernicusDemType, DEM},
        dem_gdal::DEMGdal,
        geodesy::{geodetic_to_ecef, local_normal},
        interferometry::bounding_box_from_stack,
        metadata::annotation_xml::SlcProductAnnotation,
        perp_baseline::{EnhancedDelaunayWarpFunction, perp_baseline},
        satellite_orbit::{
            OrbitalStateHistory, pixel_coords_to_radar_coords, radar_coords_to_pixel_coords,
            zero_doppler_time,
        },
        sentinel::Sentinel1SlcBurst,
        visualization::cubehelix_colormap,
    };

    fn plot_sar_amplitude(rr: &RecordingStream, burst: &Sentinel1SlcBurst) {
        let name = &burst.granule_id.raw_filename;
        let log_name = format!("slc_amplitude_{}", name);
        let array = burst.data.array_data();
        let (_, cols) = array.dim();
        let mut amplitude = array.slice(s![.., 0..cols / 2]).map(|v| v.abs()).to_owned();
        let max_amplitude = *amplitude
            .iter()
            .max_by(|&a, &b| a.partial_cmp(b).unwrap())
            .unwrap();
        amplitude.map_inplace(|x| *x = (*x / max_amplitude).powf(0.33));
        let img = Image::from_color_model_and_tensor(rerun::ColorModel::L, amplitude)
            .expect("Could not load SLC data array into image");
        rr.log_static(log_name, &img)
            .expect("Could not log SLC Image");
    }

    fn rr_phase(phase: &Array2<f32>) -> rerun::Image {
        let (rows, cols) = phase.dim();
        let rgb_vector: Vec<u8> = phase
            .as_slice_memory_order()
            .unwrap()
            .iter()
            .flat_map(|&phase| {
                let remainder = phase.rem_euclid(std::f32::consts::TAU);
                let normalized_phase = remainder / (std::f32::consts::TAU);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr_image
    }

    #[test]
    fn plot_slc_images() {
        //Records logged during cargo test will not be captured by the test harness by default.
        // The Builder::is_test method can be used in unit tests to ensure logs will be captured
        let _ = env_logger::init();
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        )
        .unwrap();

        let rr = rerun::RecordingStreamBuilder::new("plot_slc_images")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");

        plot_sar_amplitude(&rr, &primary);
        plot_sar_amplitude(&rr, &secondary);
    }

    #[test]
    fn compare_zero_doppler() {
        let _ = env_logger::init();
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        )
        .unwrap();
        let osh1 = primary.orbital_state_history();
        let osh2 = secondary.orbital_state_history();
        let ground_target_pos =
            Vector3::new(-975242.9191498039, -5932367.375945898, 2129940.582817596);
        let zero_doppler1 = osh1.find_zero_doppler_state(ground_target_pos);
        let zero_doppler2 = osh2.find_zero_doppler_state(ground_target_pos);
        println!("zero doppler 1 = {:?}", zero_doppler1);
        println!("zero doppler 1 = {:?}", zero_doppler2);
        let rcoords1: [f32; 2] = radar_coords_to_pixel_coords(zero_doppler1, &primary.metadata);
        let rcoords2: [f32; 2] = radar_coords_to_pixel_coords(zero_doppler2, &secondary.metadata);
        println!("radar coords 1 = {:?}", rcoords1);
        println!("radar coords 2 = {:?}", rcoords2);

        // Dist to target at t0'
        let start_time = secondary.metadata.ads_header.start_time;
        let t0 = zero_doppler2.time;
        println!(
            "start_time - t0 = {} sec",
            (start_time - t0).as_seconds_f32()
        );
        let (pos, vel) = osh2.interp_pos_vel(t0);
        println!("pos = {pos}");
        println!("vel = {vel}");
        let normalized_displacement = (ground_target_pos - pos).normalize();
        let d = (pos - Vector3::from(ground_target_pos)).norm();
        println!("d = {d}");
        let doppler = vel
            .normalize()
            .dot(&normalized_displacement)
            .acos()
            .to_degrees();
        println!("theta = {doppler}");

        // Distance to start time zero-doppler plane
        let (pos, vel) = osh2.interp_pos_vel(start_time);
        let dist_pi = (ground_target_pos - pos).dot(&vel.normalize());
        println!(
            "Distance to start time zero-doppler plane: {} km",
            dist_pi * 1e-3
        );
    }

    #[test]
    fn test_backgeocoding() {
        let rr = rerun::RecordingStreamBuilder::new("test_backgeocoding")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let burst = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        )
        .unwrap();
        let annotation = &burst.metadata;
        let osh = burst.orbital_state_history();
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

        // DEM extent
        let dem_corners = dem.closed_corners_lat_lon();
        rr.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
        // GCPs on the map
        let gcps_lat_lon = gcps
            .iter()
            .map(|gcp| [gcp.latitude, gcp.longitude])
            .collect::<Vec<_>>();
        rr.log_static(
            "GCPs Lat Lon",
            &rerun::GeoPoints::from_lat_lon(&gcps_lat_lon),
        )
        .unwrap();
    }

    #[test]
    fn test_orbit_speed() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        )
        .unwrap();
        let rr = rerun::RecordingStreamBuilder::new("test_orbit_speed")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let bursts = [primary, secondary];

        for (burst_id, burst) in bursts.iter().enumerate() {
            let osh = burst.orbital_state_history();
            let start_time = burst.metadata.ads_header.start_time;
            let end_time = osh.time.last().unwrap();
            let n = 1000;
            let dt_nanos = (*end_time - start_time).num_nanoseconds().unwrap() / n;
            let mut t = start_time;
            let mut sat_pos = vec![];
            let mut sat_vel = vec![];

            for i in 0..n {
                let (pos, vel) = osh.interp_pos_vel(t);
                let pos_f32 = pos.map(|x| x as f32).data.0[0];
                let vel_f32 = vel.map(|x| x as f32).data.0[0];
                sat_pos.push(pos_f32);
                sat_vel.push(vel_f32);
                rr.set_time(
                    "time",
                    std::time::SystemTime::UNIX_EPOCH
                        + std::time::Duration::from_nanos((i * dt_nanos) as u64),
                );
                rr.log(
                    format!("sat_pos_{}", burst_id),
                    &rerun::Points3D::new([pos_f32]),
                )
                .expect("Unable to log sat pos");
                let arrow_vel = rerun::Arrows3D::from_vectors([vel_f32]).with_origins([pos_f32]);
                rr.log(format!("sat_vel_{}", burst_id), &arrow_vel).unwrap();
                rr.log(
                    format!("sat_vel_scalar_{}", burst_id),
                    &rerun::Scalars::new([vel.norm()]),
                )
                .expect("Unable to log scalar");
                t = t + chrono::TimeDelta::nanoseconds(dt_nanos);
            }

            let sat_trajectory = rerun::LineStrip3D::from_iter(sat_pos);
            rr.log_static(
                format!("Satellite trajectory {}", burst_id),
                &rerun::LineStrips3D::new([sat_trajectory]),
            )
            .expect("Unable to log sat trajectory");

            // Log original key points
            let sat_points = osh
                .position
                .iter()
                .map(|p| p.map(|x| x as f32).data.0[0])
                .collect::<Vec<_>>();
            rr.log_static(
                format!("Satellite position original points {}", burst_id),
                &rerun::Points3D::new(sat_points),
            )
            .expect("Unable to log sat points");

            // if burst_id == 1 {
            //     for (idx, v) in osh.velocity.iter().enumerate() {
            //         println!("v[{}] = {}", idx, v.norm());
            //     }
            // }
        }
    }

    #[test]
    fn differential_phase_plot() {
        env_logger::init();
        log::info!("Starting differential_phase_plot test.");
        let rr = rerun::RecordingStreamBuilder::new("differential_phase_plot")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
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
        let start_time_ref = reference.metadata.ads_header.start_time;

        for (id, secondary) in secondaries.iter().enumerate() {
            rr.set_time_sequence("secondary_id", id as i64);
            let start_time_sec = secondary.metadata.ads_header.start_time;
            let delta_time = start_time_sec - start_time_ref;
            log::info!("Processing secondary {}", secondary.granule_id.raw_filename);
            log::info!(
                "Days between secondary acquisition and reference: {}",
                delta_time.num_days()
            );

            log::info!("Computing warp function");
            let start_time = std::time::Instant::now();
            let warp_function = EnhancedDelaunayWarpFunction::new(&reference, &secondary, &dem);
            let end_time = std::time::Instant::now();
            log::info!(
                "Time taken to compute warp function: {:?}",
                end_time - start_time
            );

            let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();
            let mut coregistered_secondary_img =
                Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));

            let kernel = KnabSincKernel::default();
            let deramp = DerampSlcBurst::new();

            log::info!("Deramping reference and secondary images");
            let start_time = std::time::Instant::now();
            let reference_img = deramp.apply_forward(&reference);
            let secondary_img = deramp.apply_forward(&secondary);
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            const CHUNK_SIZE: usize = 256;
            log::info!("Resampling secondary image to reference image via warp function");
            let start_time = std::time::Instant::now();
            coregistered_secondary_img
                .axis_chunks_iter_mut(Axis(0), CHUNK_SIZE)
                .into_par_iter()
                .enumerate()
                .for_each_init(
                    || warp_function.triangulation.natural_neighbor(),
                    |nn, (chunk_idx, mut chunk)| {
                        let az_offset = chunk_idx * CHUNK_SIZE;

                        for (i, mut row) in chunk.outer_iter_mut().enumerate() {
                            let ref_az = az_offset + i;

                            for (ref_rg, value) in row.iter_mut().enumerate() {
                                let ref_coords = [ref_az as f64, ref_rg as f64];
                                let compute_mapped_coord = |dimension: usize| {
                                    nn.interpolate(
                                        |v| v.data().secondary_coords[dimension],
                                        ref_coords.into(),
                                    )
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
                        }
                    },
                );
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            let osh_1 = reference.orbital_state_history();
            let annotation_1 = &reference.metadata;
            let osh_2 = secondary.orbital_state_history();
            let annotation_2 = &secondary.metadata;
            let mut phase_diff =
                Array2::from_shape_fn((ref_azimuth_dim, ref_slant_range_dim / 2), |(i, j)| {
                    let s1 = reference_img[[i, j]];
                    let s2 = coregistered_secondary_img[[i, j]];
                    (s1 * s2.conj()).arg()
                });

            rr.log(format!("phase/{}", id), &rr_phase(&phase_diff))
                .expect("Could not log phase to Rerun");

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
                                let s_1 = osh_1.interp_pos(zero_doppler_1);
                                let s_2 = osh_2.interp_pos(zero_doppler_2);

                                let r1 = (s_1 - ground_target_pos).norm();
                                let r2 = (s_2 - ground_target_pos).norm();

                                let delta_phi =
                                    4.0 * std::f64::consts::PI * (r1 - r2) / SENTINEL_1_WAVELENGTH;

                                *phase -= delta_phi as f32;
                            }
                        }
                    },
                );
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            rr.log(format!("diff_phase/{}", id), &rr_phase(&phase_diff))
                .expect("Could not log phase to Rerun");
        }
    }

    #[test]
    fn topo_phase_plot() {
        env_logger::init();
        log::info!("Starting topo_phase_plot test.");
        let rr = rerun::RecordingStreamBuilder::new("topo_phase_plot")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
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
        let start_time_ref = reference.metadata.ads_header.start_time;

        for (id, secondary) in secondaries.iter().enumerate() {
            rr.set_time_sequence("secondary_id", id as i64);
            let start_time_sec = secondary.metadata.ads_header.start_time;
            let delta_time = start_time_sec - start_time_ref;
            log::info!("Processing secondary {}", secondary.granule_id.raw_filename);
            log::info!(
                "Days between secondary acquisition and reference: {}",
                delta_time.num_days()
            );

            log::info!("Computing warp function");
            let start_time = std::time::Instant::now();
            let warp_function = EnhancedDelaunayWarpFunction::new(&reference, &secondary, &dem);
            let end_time = std::time::Instant::now();
            log::info!(
                "Time taken to compute warp function: {:?}",
                end_time - start_time
            );

            let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();

            const CHUNK_SIZE: usize = 256;

            let osh_1 = reference.orbital_state_history();
            let annotation_1 = &reference.metadata;
            let osh_2 = secondary.orbital_state_history();
            let annotation_2 = &secondary.metadata;

            log::info!("Calculating topographic phase (exact)");
            let s = secondary
                .metadata
                .image_annotation
                .image_information
                .range_pixel_spacing;
            let mut topo_phase_exact = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim / 2));
            let start_time = std::time::Instant::now();
            topo_phase_exact
                .axis_chunks_iter_mut(Axis(0), CHUNK_SIZE)
                .into_par_iter()
                .enumerate()
                .for_each_init(
                    || warp_function.triangulation.natural_neighbor(),
                    |nn, (chunk_idx, mut chunk)| {
                        let az_offset = chunk_idx * CHUNK_SIZE;

                        for (i, mut row) in chunk.outer_iter_mut().enumerate() {
                            let ref_az = az_offset + i;
                            let mut accumulated_dphi = 0.0;
                            let mut current_height = None;

                            for (ref_rg, phase) in row.iter_mut().enumerate() {
                                let ref_coords = [ref_az as f64, ref_rg as f64];
                                // Helper function to compute mapped coordinate for a given dimension
                                let compute_mapped_coord = |dimension: usize| {
                                    nn.interpolate(
                                        |v| v.data().secondary_coords[dimension],
                                        ref_coords.into(),
                                    )
                                };
                                let sec_az = compute_mapped_coord(0);
                                let sec_rg = compute_mapped_coord(1);

                                // Now compute the ground target position
                                let ground_target_lat =
                                    nn.interpolate(|v| v.data().lat, ref_coords.into());
                                let ground_target_lon =
                                    nn.interpolate(|v| v.data().lon, ref_coords.into());

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

                                let ground_target_pos = Vector3::from(
                                    dem.get_ecef_at_lat_lon(ground_target_lat, ground_target_lon),
                                );

                                let radar_coords_1 = pixel_coords_to_radar_coords(
                                    ref_az as f64,
                                    ref_rg as f64,
                                    annotation_1,
                                );
                                let radar_coords_2 = pixel_coords_to_radar_coords(
                                    sec_az as f64,
                                    sec_rg as f64,
                                    annotation_2,
                                );
                                let (s_1, _) = osh_1.interp_pos_vel(radar_coords_1.time);
                                let (s_2, _) = osh_2.interp_pos_vel(radar_coords_2.time);

                                let bperp = perp_baseline(&s_1, &s_2, &ground_target_pos);
                                let l = (s_1 - ground_target_pos).normalize();
                                let r = (s_1 - ground_target_pos).norm();
                                let normal = Vector3::from(local_normal(
                                    ground_target_lat,
                                    ground_target_lon,
                                ));
                                let theta = l.dot(&normal).acos();

                                *phase -= accumulated_dphi as f32;

                                let height = dem
                                    .get_height_at_lat_lon(ground_target_lat, ground_target_lon)
                                    as f64;

                                if let Some(current_height) = current_height {
                                    let height_diff = height - current_height;
                                    let height_diff_dphi =
                                        (4.0 * std::f64::consts::PI * bperp * height_diff)
                                            / (r * SENTINEL_1_WAVELENGTH * theta.sin());
                                    accumulated_dphi -= height_diff_dphi;
                                }

                                current_height = Some(height);

                                let dphi = (4.0 * std::f64::consts::PI * bperp * s)
                                    / (r * SENTINEL_1_WAVELENGTH * theta.tan());
                                accumulated_dphi += dphi;
                            }
                        }
                    },
                );
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            rr.log("topo_phase/exact", &rr_phase(&topo_phase_exact))
                .expect("Could not log phase to Rerun");

            // Topographic phase: gradient approximation
            log::info!("Calculating topographic phase (approximation)");
            let mut topo_phase_approx = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim / 2));
            let start_time = std::time::Instant::now();
            topo_phase_approx
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
                                let s_1 = osh_1.interp_pos(zero_doppler_1);
                                let s_2 = osh_2.interp_pos(zero_doppler_2);

                                let r1 = (s_1 - ground_target_pos).norm();
                                let r2 = (s_2 - ground_target_pos).norm();

                                let delta_phi =
                                    4.0 * std::f64::consts::PI * (r1 - r2) / SENTINEL_1_WAVELENGTH;

                                *phase -= delta_phi as f32;
                            }
                        }
                    },
                );
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            rr.log("topo_phase/approx", &rr_phase(&topo_phase_approx))
                .expect("Could not log phase to Rerun");
        }
    }
}
