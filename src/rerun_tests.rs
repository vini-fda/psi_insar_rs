#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Unit, Vector3};
    use ndarray::{Array1, Array2, ArrayView2, Axis, s};
    use ndarray_npy::WriteNpyExt;
    use num_complex::{Complex, ComplexFloat};
    use rayon::prelude::*;
    use rerun::{Color, Image, RecordingStream};

    use crate::{
        constants::SENTINEL_1_WAVELENGTH,
        coregistration::{
            coarse_coregistration::{CoarseCoregistration, CoregistrationResult},
            deramping::DerampSlcBurst,
            interpolation2d::{KnabSincKernel, interpolate_2d},
        },
        dem::{CopernicusDemType, DEM},
        geodesy::{geodetic_to_ecef, local_normal},
        interferometry::bounding_box_from_stack,
        metadata::annotation_xml::SlcProductAnnotation,
        perp_baseline::{
            EnhancedDelaunayWarpFunction, FlatEarthComponentsInterpolator, flat_earth_dphi,
            perp_baseline,
        },
        satellite_orbit::{
            OrbitalStateHistory, pixel_coords_to_radar_coords, radar_coords_to_pixel_coords,
            zero_doppler_time,
        },
        sentinel::Sentinel1SlcBurst,
        stft::{Stft, WindowFunction},
        visualization::{cubehelix_colormap, turbo_colorized_values},
    };

    fn plot_sar_amplitude(rr: &RecordingStream, burst: &Sentinel1SlcBurst) {
        let name = &burst.granule_id.raw_filename;
        let log_name = format!("slc_amplitude_{name}");
        let array = burst.data.array_f32();
        let (_, cols) = array.dim();
        let mut amplitude = array
            .slice(s![.., 0..cols / 2])
            .map(|&v| v.abs())
            .to_owned();
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

    fn rr_gamma_corrected_amplitude(slc_data: &ArrayView2<Complex<f32>>) -> rerun::Image {
        const GAMMA: f32 = 0.3;
        let abs = slc_data.map(|x| x.abs());
        let max = *abs
            .iter()
            .max_by(|&a, &b| a.partial_cmp(b).unwrap())
            .unwrap();
        let corrected = abs.mapv_into(|x| (x / max).powf(GAMMA));
        rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, corrected)
            .expect("Unable to create Rerun image")
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

        rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        )
    }

    #[test]
    fn plot_slc_images() {
        //Records logged during cargo test will not be captured by the test harness by default.
        // The Builder::is_test method can be used in unit tests to ensure logs will be captured
        env_logger::init();
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
        env_logger::init();
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
        println!("zero doppler 1 = {zero_doppler1:?}");
        println!("zero doppler 1 = {zero_doppler2:?}");
        let rcoords1: [f32; 2] = radar_coords_to_pixel_coords(zero_doppler1, &primary.metadata);
        let rcoords2: [f32; 2] = radar_coords_to_pixel_coords(zero_doppler2, &secondary.metadata);
        println!("radar coords 1 = {rcoords1:?}");
        println!("radar coords 2 = {rcoords2:?}");

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
        let dem = DEM::open_file("dem.tif");
        // DEM extent
        let dem_corners = dem.closed_corners_lat_lon();
        rr.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();

        // Backgeocoding
        let bursts = [
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
            "download/S1_305967_IW3_20150916T122546_VV_8302-BURST",
            // "download/S1_305967_IW3_20150928T122546_VV_5407-BURST",
            // "download/S1_305967_IW3_20151010T122546_VV_7501-BURST",
            // "download/S1_305967_IW3_20151103T122546_VV_AE93-BURST",
            // "download/S1_305967_IW3_20151115T122546_VV_8956-BURST",
            // "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        ]
        .iter()
        .map(|name| Sentinel1SlcBurst::load_first_from_directory(name).unwrap())
        .collect::<Vec<_>>();
        for burst in bursts {
            let annotation = &burst.metadata;
            let burst_name = &burst.granule_id.raw_filename;
            let osh = burst.orbital_state_history();

            let mut points = vec![];
            for (lat, lon) in dem.lat_lon_iter() {
                let pos = dem.get_ecef_at_lat_lon(lat, lon);
                let ground_target_pos = Vector3::<f64>::from(pos);
                let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
                let [azimuth_idx, slant_range_idx] =
                    radar_coords_to_pixel_coords(zero_doppler, annotation);

                points.push([slant_range_idx, azimuth_idx]);
            }
            let points = rerun::Points2D::new(points)
                .with_colors(dem.vertex_colors())
                .with_radii([10.0]);
            rr.log_static(format!("backgeocoded_points/{burst_name}"), &points)
                .unwrap();

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
                    radar_coords_to_pixel_coords(zero_doppler, annotation);

                points.push([slant_range_idx, azimuth_idx]);
            }
            let points = rerun::Points2D::new(points).with_radii([30.0]);
            rr.log_static(format!("backgeocoded_gcps/{burst_name}"), &points)
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
    }

    #[test]
    fn test_orbit_speed() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151103T122546_VV_AE93-BURST",
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
                    format!("sat_pos_{burst_id}"),
                    &rerun::Points3D::new([pos_f32]),
                )
                .expect("Unable to log sat pos");
                let arrow_vel = rerun::Arrows3D::from_vectors([vel_f32]).with_origins([pos_f32]);
                rr.log(format!("sat_vel_{burst_id}"), &arrow_vel).unwrap();
                rr.log(
                    format!("sat_vel_scalar_{burst_id}"),
                    &rerun::Scalars::new([vel.norm()]),
                )
                .expect("Unable to log scalar");
                t += chrono::TimeDelta::nanoseconds(dt_nanos);
            }

            let sat_trajectory = rerun::LineStrip3D::from_iter(sat_pos);
            rr.log_static(
                format!("Satellite trajectory {burst_id}"),
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
                format!("Satellite position original points {burst_id}"),
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
        println!("Bounding box: {bounding_box:?}");
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
                                let zero_doppler_2 = zero_doppler_time(sec_az, annotation_2);
                                let s_1 = osh_1.interp_pos(zero_doppler_1);
                                let s_2 = osh_2.interp_pos(zero_doppler_2);
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
            let img = Image::from_color_model_and_tensor(rerun::ColorModel::L, phase_diff)
                .expect("Could not load img.");
            rr.log("perp_baseline", &img).expect("Unable to log img");
        }
    }
    fn normalize(data: &mut Array2<f32>) {
        let max_amplitude = data.iter().fold(0.0, |acc: f32, &x| acc.max(x));
        data.map_mut(|x| *x /= max_amplitude);
    }

    #[test]
    fn coarse_coregistration() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151103T122546_VV_AE93-BURST",
        )
        .unwrap();
        let reference_image = primary.data.array_f32();
        let secondary_image = secondary.data.array_f32();
        let rec = rerun::RecordingStreamBuilder::new("visual_test_coregistration")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let coregistration = CoarseCoregistration::new(255, 128).unwrap();
        let CoregistrationResult {
            offsets,
            correlation,
            ref_image_range,
            sec_image_range,
        } = coregistration.estimate_offset(&reference_image, &secondary_image);
        // Log correlation tensor
        let data = correlation.mapv_into_any(|c| c.norm());
        let tensor = rerun::Tensor::try_from(data)
            .unwrap()
            .with_dim_names(["rows", "cols"]);
        rec.log("correlation", &tensor)
            .expect("Could not finish recording");
        let mut sec_img: Array2<f32> = secondary_image.map(|c| c.norm());
        normalize(&mut sec_img);
        let img_sec =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, sec_img).unwrap();
        rec.log("secondary_image", &img_sec)
            .expect("Could not finish recording");
        // Log 2 images for comparison
        let mut ref_patch = reference_image
            .slice(s![ref_image_range[0].clone(), ref_image_range[1].clone()])
            .map(|c| c.norm().powf(0.3))
            .to_owned();
        normalize(&mut ref_patch);
        let img_ref =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, ref_patch).unwrap();
        rec.log("ref", &img_ref)
            .expect("Could not finish recording");
        let mut kernel = secondary_image
            .slice(s![sec_image_range[0].clone(), sec_image_range[1].clone()])
            .map(|c| c.norm().powf(0.3))
            .to_owned();
        normalize(&mut kernel);
        let img_sec =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, kernel).unwrap();
        rec.log("sec", &img_sec)
            .expect("Could not finish recording");
        rec.log(
            "logs",
            &rerun::TextLog::new(format!("offsets = {offsets:?}"))
                .with_level(rerun::TextLogLevel::INFO),
        )
        .unwrap();
        let lat_lon = [
            [19.49831428810679, -98.59301000370277],
            [19.50516497145322, -98.63155270190656],
            [19.51197728410019, -98.66992859316593],
            [19.51875199752162, -98.7081413973442],
            [19.52548985740382, -98.74619471009619],
            [19.53219158481003, -98.78409200847771],
            [19.53885787727854, -98.82183665623535],
            [19.5454894098591, -98.85943190879907],
            [19.55208683609183, -98.89688091799755],
            [19.55865078893238, -98.9341867365148],
            [19.56518188162694, -98.97135232210496],
            [19.57168070854037, -99.00838054158147],
            [19.57814784594043, -99.04527417459472],
            [19.58458385274104, -99.08203591721202],
            [19.59098927120689, -99.11866838531225],
            [19.5973646276222, -99.15517411780644],
            [19.60371043292545, -99.19155557969563],
            [19.61002718331234, -99.22781516497541],
            [19.61631536080904, -99.26395519939655],
            [19.62257559081166, -99.29997885004042],
            [19.62877652653297, -99.33570497416603],
            [19.33245909128312, -98.62592837407114],
            [19.33931662789667, -98.66442995280119],
            [19.34613584071955, -98.70276488660213],
            [19.35291750085852, -98.74093689141644],
            [19.35966235364623, -98.77894955911454],
            [19.36637111980492, -98.81680636309756],
            [19.37304449654394, -98.85451066358128],
            [19.3796831585957, -98.89206571258312],
            [19.38628775919396, -98.92947465863222],
            [19.39285893099849, -98.96674055122072],
            [19.39939728696963, -99.0038663450142],
            [19.40590342119594, -99.0408549038362],
            [19.41237790967802, -99.07770900444162],
            [19.41882131107135, -99.1144313400927],
            [19.4252341673906, -99.15102452394993],
            [19.43161700467787, -99.187491092289],
            [19.43797033363727, -99.22383350755531],
            [19.44429465023745, -99.2600541612652],
            [19.45059043628451, -99.29615537676354],
            [19.45685815996673, -99.33213941184621],
            [19.46306674963845, -99.36782713184502],
        ];
        rec.log(
            "geo_points",
            &rerun::GeoPoints::from_lat_lon(lat_lon.iter()),
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");

        let mut dem_corners = dem.corners_lat_lon().to_vec();
        let first = dem_corners.first().unwrap();
        dem_corners.push(*first);
        let dem_array = dem.array();
        let tensor = rerun::Tensor::try_from(dem_array)
            .unwrap()
            .with_dim_names(["rows", "cols"]);
        rec.log("DEM", &tensor).expect("Could not finish recording");
        let data_img = rerun::Image::from_color_model_and_tensor(
            rerun::ColorModel::L,
            reference_image
                .map(|c| c.norm().powf(0.3) / 22.0)
                .to_owned(),
        )
        .unwrap();
        rec.log("ref_img", &data_img)
            .expect("Could not finish recording");
        rec.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
    }

    #[test]
    fn coarse_coregistration_stack() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondaries = [
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
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
        let rec = rerun::RecordingStreamBuilder::new("coarse_coregistration_stack")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let reference_image = primary.data.array_f32();
        for (k, secondary) in secondaries.iter().enumerate() {
            rec.set_time_sequence("secondary_index", k as i64);
            let secondary_image = secondary.data.array_f32();

            let coregistration = CoarseCoregistration::new(255, 128).unwrap();
            let CoregistrationResult {
                offsets,
                correlation,
                ref_image_range,
                sec_image_range,
            } = coregistration.estimate_offset(&reference_image, &secondary_image);
            // Log correlation tensor
            let data = correlation.mapv_into_any(|c| c.norm());
            let tensor = rerun::Tensor::try_from(data)
                .unwrap()
                .with_dim_names(["azimuth", "slant_range"]);
            rec.log("correlation", &tensor)
                .expect("Could not finish recording");
            let sec_img = rr_gamma_corrected_amplitude(
                &secondary_image.slice(s![.., ..secondary_image.dim().1 / 2]),
            );
            rec.log("sec_img", &sec_img)
                .expect("Could not finish recording");
            // Log 2 images for comparison
            let mut ref_patch = reference_image
                .slice(s![ref_image_range[0].clone(), ref_image_range[1].clone()])
                .map(|c| c.norm().powf(0.3))
                .to_owned();
            normalize(&mut ref_patch);
            let img_ref =
                rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, ref_patch).unwrap();
            rec.log("ref", &img_ref)
                .expect("Could not finish recording");
            let mut kernel = secondary_image
                .slice(s![sec_image_range[0].clone(), sec_image_range[1].clone()])
                .map(|c| c.norm().powf(0.3))
                .to_owned();
            normalize(&mut kernel);
            let img_sec =
                rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, kernel).unwrap();
            rec.log("sec", &img_sec)
                .expect("Could not finish recording");
            rec.log(
                "logs",
                &rerun::TextLog::new(format!("offsets = {offsets:?}"))
                    .with_level(rerun::TextLogLevel::INFO),
            )
            .unwrap();

            let lat_lon = secondary
                .metadata
                .geolocation_grid
                .geolocation_grid_point_list
                .get_lat_lon();
            rec.log(
                "geo_points",
                &rerun::GeoPoints::from_lat_lon(lat_lon.iter()),
            )
            .unwrap();
            let ref_img = rr_gamma_corrected_amplitude(
                &reference_image.slice(s![.., ..reference_image.dim().1 / 2]),
            );
            rec.log("ref_img", &ref_img)
                .expect("Could not finish recording");
        }
        let dem = DEM::open_file("dem.tif");

        let mut dem_corners = dem.corners_lat_lon().to_vec();
        let first = dem_corners.first().unwrap();
        dem_corners.push(*first);
        let dem_array = dem.array();
        let tensor = rerun::Tensor::try_from(dem_array)
            .unwrap()
            .with_dim_names(["rows", "cols"]);
        rec.log_static("DEM", &tensor)
            .expect("Could not finish recording");
        rec.log_static(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
    }

    #[test]
    fn test_resample_secondary_to_reference() {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");

        let resampled_data = crate::coregistration::warp_function::resample_secondary_to_reference(
            &reference, &secondary, &dem,
        );
        // lets reduce the number of samples by 1/2 in the cols
        let (rows, cols) = resampled_data.dim();
        let resampled_data = resampled_data.slice(s![.., 0..cols / 2]).to_owned();

        // Log amplitude for resampled data
        let data_norm = resampled_data.map(|c| c.norm());
        let max_val = data_norm.iter().fold(0.0f32, |a, &b| a.max(b));
        let data_norm = data_norm.map(|c| (c / max_val).powf(0.3));
        let img =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, data_norm).unwrap();
        let rr = rerun::RecordingStreamBuilder::new("test_resample_secondary_to_reference")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        rr.log("resampled_amplitude", &img)
            .expect("Could not log resampled_amplitude to Rerun");

        // Log phase for resampled data
        let vector = resampled_data.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&x| {
                let phase = x.arg();
                let normalized_phase =
                    (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32 / 2, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log("resampled_phase", &rr_image)
            .expect("Could not log resampled_phase to Rerun");

        // Log the reference image amplitude
        let ref_array = reference.data.array_f32();
        let ref_array = ref_array.slice(s![.., 0..cols / 2]).to_owned();
        let ref_array_norm = ref_array.map(|c| c.norm());
        let max_val = ref_array_norm.iter().fold(0.0f32, |a, &b| a.max(b));
        let ref_array_norm = ref_array_norm.map(|c| (c / max_val).powf(0.3));
        let ref_img =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, ref_array_norm)
                .unwrap();
        rr.log("reference_amplitude", &ref_img)
            .expect("Could not log reference_amplitude to Rerun");

        // Log phase for reference data
        let vector = ref_array.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&x| {
                let phase = x.arg();
                let normalized_phase =
                    (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32 / 2, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log("reference_phase", &rr_image)
            .expect("Could not log reference_phase to Rerun");
    }
    #[test]
    fn testfn_dem() {
        let dem = DEM::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_warp_fn_dem")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance");

        // Collect vertices and heights
        let mut vertices = Vec::new();
        let mut heights = Vec::new();
        for (lat, lon, height) in dem.lat_lon_height_iter() {
            vertices.push([lon as f32, lat as f32, 0.0]);
            heights.push(height as f32);
        }

        // Create triangle indices for a grid
        let rows = dem.rows();
        let cols = dem.cols();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        for i in 0..(rows - 1) {
            for j in 0..(cols - 1) {
                let idx = (i * cols + j) as u32;
                let cols = cols as u32;
                // First triangle
                indices.push([idx, idx + 1, idx + cols]);
                // Second triangle
                indices.push([idx + 1, idx + cols + 1, idx + cols]);
            }
        }

        // Create vertex colors based on height
        let min_height = heights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_height = heights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let vertex_colors: Vec<u32> = heights
            .iter()
            .map(|&h| {
                let t = (h - min_height) / (max_height - min_height);
                let r = (t * 255.0) as u8;
                let g = ((1.0 - t) * 255.0) as u8;
                let b = 0;
                ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | 0xFF
            })
            .collect();

        // Create and log the mesh
        let mesh = rerun::Mesh3D::new(vertices)
            .with_triangle_indices(indices)
            .with_vertex_colors(vertex_colors);

        rr.log("dem_mesh", &mesh)
            .expect("Could not log mesh to Rerun");
    }

    #[test]
    fn testfn_dem_rgb() {
        let dem = DEM::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_warp_fn_dem_rgb")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance");

        // Collect vertices and heights
        let mut vertices = Vec::new();
        let mut lons = Vec::new();
        let mut lats = Vec::new();
        for (lat, lon) in dem.lat_lon_iter() {
            vertices.push([lon as f32, lat as f32, 0.0_f32]);
            lons.push(lon);
            lats.push(lat);
        }

        // Find min/max for normalization
        let min_lon = lons.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_lon = lons.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_lat = lats.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_lat = lats.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Create vertex colors based on normalized lon/lat
        let vertex_colors: Vec<u32> = vertices
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let lon_norm = (lons[i] - min_lon) / (max_lon - min_lon);
                let lat_norm = (lats[i] - min_lat) / (max_lat - min_lat);

                let r = (lon_norm * 255.0) as u8;
                let g = (lat_norm * 255.0) as u8;
                let b = 0;

                ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | 0xFF
            })
            .collect();

        // Create triangle indices for a grid
        let rows = dem.rows();
        let cols = dem.cols();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        for i in 0..(rows - 1) {
            for j in 0..(cols - 1) {
                let idx = (i * cols + j) as u32;
                let cols = cols as u32;
                // First triangle
                indices.push([idx, idx + 1, idx + cols]);
                // Second triangle
                indices.push([idx + 1, idx + cols + 1, idx + cols]);
            }
        }

        // Create and log the mesh
        let mesh = rerun::Mesh3D::new(vertices.clone())
            .with_triangle_indices(indices.clone())
            .with_vertex_colors(vertex_colors);

        rr.log("dem_mesh_rgb", &mesh)
            .expect("Could not log mesh to Rerun");

        // Create wireframe line strips for each triangle
        let mut line_strips = Vec::new();
        for triangle in indices {
            let strip = [
                vertices[triangle[0] as usize],
                vertices[triangle[1] as usize],
                vertices[triangle[2] as usize],
                vertices[triangle[0] as usize], // Close the triangle
            ];
            line_strips.push(strip);
        }

        // Log the wireframe
        rr.log(
            "wireframe",
            &rerun::LineStrips3D::new(line_strips).with_colors([0xFFFFFFFF]), // White color
        )
        .expect("Could not log wireframe to Rerun");
    }

    #[test]
    fn test_warp_function() {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let rho = crate::coregistration::compute_warp_function(&reference, &secondary, &dem);
        let (rows, cols) = rho.dim();
        let rho = rho.slice(s![0..rows, 0..cols / 2]).to_owned();

        let rr = rerun::RecordingStreamBuilder::new("test_warp_function")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let (rows, cols) = rho.dim();
        let (v, _) = rho.into_raw_vec_and_offset();
        let img = rerun::Image::from_rgba32(v.as_flattened(), [cols as u32, rows as u32]);
        rr.log("warp_fn", &img).expect("Could not finish recording");
    }

    #[test]
    fn testfn_dem_radar_coords() {
        let dem = DEM::open_file("dem.tif");
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let rr = rerun::RecordingStreamBuilder::new("test_warp_fn_radar_coords")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance");

        // Get orbital state history for radar coordinate conversion
        let ref_osh = reference.orbital_state_history();

        // Collect vertices in radar coordinates
        let mut vertices = Vec::new();
        let mut lons = Vec::new();
        let mut lats = Vec::new();
        for (lat, lon) in dem.lat_lon_iter() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let zero_doppler = ref_osh.find_zero_doppler_state(pos.into());
            let [azimuth, slant_range] =
                radar_coords_to_pixel_coords(zero_doppler, &reference.metadata);

            vertices.push([slant_range, azimuth, 0.0]); // Using col as x, row as y
            lons.push(lon);
            lats.push(lat);
        }

        // Find min/max for normalization
        let min_lon = lons.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_lon = lons.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_lat = lats.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_lat = lats.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Create vertex colors based on normalized lon/lat
        let vertex_colors: Vec<u32> = vertices
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let lon_norm = (lons[i] - min_lon) / (max_lon - min_lon);
                let lat_norm = (lats[i] - min_lat) / (max_lat - min_lat);

                let r = (lon_norm * 255.0) as u8;
                let g = (lat_norm * 255.0) as u8;
                let b = 0;

                ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | 0xFF
            })
            .collect();

        // Create triangle indices for a grid
        let rows = dem.rows();
        let cols = dem.cols();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        for i in 0..(rows - 1) {
            for j in 0..(cols - 1) {
                let idx = (i * cols + j) as u32;
                let cols = cols as u32;
                // First triangle
                indices.push([idx, idx + 1, idx + cols]);
                // Second triangle
                indices.push([idx + 1, idx + cols + 1, idx + cols]);
            }
        }

        // Create and log the mesh
        let mesh = rerun::Mesh3D::new(vertices.clone())
            .with_triangle_indices(indices.clone())
            .with_vertex_colors(vertex_colors);

        rr.log("dem_mesh_radar", &mesh)
            .expect("Could not log mesh to Rerun");

        // Create wireframe line strips for each triangle
        let mut line_strips = Vec::new();
        for triangle in indices {
            let strip = [
                vertices[triangle[0] as usize],
                vertices[triangle[1] as usize],
                vertices[triangle[2] as usize],
                vertices[triangle[0] as usize], // Close the triangle
            ];
            line_strips.push(strip);
        }

        // Log the wireframe
        rr.log(
            "wireframe",
            &rerun::LineStrips3D::new(line_strips).with_colors([0xFFFFFFFF]), // White color
        )
        .expect("Could not log wireframe to Rerun");
    }
    #[test]
    fn test_spectrum_visualization() -> Result<(), Box<dyn std::error::Error>> {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let image = reference.data.array_f32();
        let rec =
            rerun::RecordingStreamBuilder::new("fft_spectrum_visualization").connect_grpc()?;

        // Compute FFT along both dimensions
        let spectrum_0 = crate::coregistration::spectrum::compute_spectrum(&image, 0);
        let spectrum_1 = crate::coregistration::spectrum::compute_spectrum(&image, 1);

        // Compute average magnitude squared along each dimension
        let (rows, cols) = image.dim();

        // For dimension 0 (rows), average across columns
        // this shows the FFT in the Azimuth direction
        for row in 0..rows {
            let row_avg = spectrum_0.row(row).map(|&x| x.norm_sqr()).sum() / cols as f32;

            // Map row index to FFT frequency ordering
            let freq_idx = if row <= rows / 2 { row } else { row - rows };

            rec.set_time_sequence("row", freq_idx as i64);
            rec.log("fft_dim0", &rerun::Scalars::new([row_avg as f64]))?;
        }

        // calculate the DC center of the spectrum
        // find the center of mass of the spectrum distribution
        let mut total_mass = 0.0;
        let mut weighted_sum = 0.0;

        for row in 0..rows {
            let row_avg = spectrum_0.row(row).map(|&x| x.norm_sqr()).sum() / cols as f32;
            total_mass += row_avg;
            weighted_sum += row_avg * row as f32;
        }

        let dc_center_row = (weighted_sum / total_mass) as usize;
        let dc_center = spectrum_0.row(dc_center_row).map(|&x| x.norm_sqr()).sum() / cols as f32;

        rec.log(
            "dc_center",
            &rerun::TextLog::new(format!(
                "DC center row: {dc_center_row}, value: {dc_center}"
            )),
        )?;

        // For dimension 1 (columns), average across rows
        for col in 0..cols {
            let col_avg = spectrum_1.column(col).map(|&x| x.norm_sqr()).sum() / rows as f32;

            // Map column index to FFT frequency ordering
            let freq_idx = if col <= cols / 2 { col } else { col - cols };

            rec.set_time_sequence("col", freq_idx as i64);
            rec.log("fft_dim1", &rerun::Scalars::new([col_avg as f64]))?;
        }

        Ok(())
    }

    #[test]
    fn test_spectrogram() -> Result<(), Box<dyn std::error::Error>> {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let image = reference.data.array_f32();
        let rec = rerun::RecordingStreamBuilder::new("spectrogram_visualization").connect_grpc()?;

        // cut cols in half
        let image = image.slice(s![.., ..image.dim().1 / 2]).to_owned();
        let rows = image.dim().0;

        // Example usage of STFT
        // fn main() {
        //     // Initialize a new STFT object
        //     let n_fft = 1024;
        //     let hop_length = 256;
        //     let stft = Stft::new(n_fft, hop_length, WindowFunction::Hann::<f64>, true);
        //     // Create a 2D array of f64
        //     let data = vec![0.0; 2048];
        //     let input = ArrayView2::from_shape((2, 1024), &data).unwrap();
        //     let expected_output = input.clone();
        //     // Perform the forward STFT
        //     let stft_res = stft.forward(input).unwrap();
        //     // perform the inverse STFT
        //     let istft_res = stft.inverse(stft_res.view()).unwrap();
        //     assert_eq!(expected_output, istft_res);
        // }
        let hop_length = 32;
        let stft = Stft::new(rows, hop_length, WindowFunction::Hann::<f32>, true);
        // Let's apply to a single column
        let column = image.column(1000);
        let stft_res = stft.forward(column).unwrap();
        let (cols, rows) = stft_res.dim();
        // let istft_res = stft.inverse(stft_res.view()).unwrap();
        // assert_eq!(column, istft_res);

        let array = stft_res.map(|x| x.norm());

        // log as image
        let rr_image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, array)?;
        rec.log("spectrogram_visualization", &rr_image)?;
        // log also phase

        let vector = stft_res.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&x| {
                let phase = x.arg();
                let normalized_phase =
                    (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rec.log("phase_visualization", &rr_image)?;
        Ok(())
    }

    #[test]
    fn test_phase_visualization() -> Result<(), Box<dyn std::error::Error>> {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let array = reference.data.array_f32();
        // cut cols in half
        let array = array.slice(s![.., ..array.dim().1 / 2]).to_owned();
        let (rows, cols) = array.dim();
        let vector = array.as_slice_memory_order().unwrap().to_vec();
        // map the vactor to phase, then map that to RGB color
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&x| {
                let phase = x.arg();
                let normalized_phase =
                    (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rec = rerun::RecordingStreamBuilder::new("image_phase_visualization").connect_grpc()?;

        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rec.log("image_phase_visualization", &rr_image)?;
        Ok(())
    }

    #[test]
    fn test_resampled_phase_difference() {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");

        // Secondary resampled to reference (also deramped)
        let resampled_sec_data =
            crate::coregistration::warp_function::resample_secondary_to_reference(
                &reference, &secondary, &dem,
            );
        let ref_deramp = DerampSlcBurst::new().apply_forward(&reference);

        // cut cols by half in both images
        let (rows, cols) = resampled_sec_data.dim();
        let resampled_sec_data = resampled_sec_data.slice(s![.., 0..cols / 2]).to_owned();
        let ref_deramp = ref_deramp.slice(s![.., 0..cols / 2]).to_owned();

        let phase_sec = resampled_sec_data.map(|c| c.arg());
        let phase_ref = ref_deramp.map(|c| c.arg());
        // Compute the phase difference between the resampled secondary and the reference deramped data
        let phase_diff = phase_sec - phase_ref;

        // Log the phase difference
        let rr = rerun::RecordingStreamBuilder::new("test_resampled_phase_difference")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let vector = phase_diff.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase =
                    (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let phase_diff_img = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32 / 2, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log("phase_difference", &phase_diff_img)
            .expect("Could not log phase_difference to Rerun");
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
        println!("Bounding box: {bounding_box:?}");
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
            let warp_function = EnhancedDelaunayWarpFunction::new(&reference, secondary, &dem);
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
            let secondary_img = deramp.apply_forward(secondary);
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

            rr.log("phase", &rr_phase(&phase_diff))
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
                                let zero_doppler_2 = zero_doppler_time(sec_az, annotation_2);
                                let s_1 = osh_1.interp_pos(zero_doppler_1);
                                let s_2 = osh_2.interp_pos(zero_doppler_2);

                                let r1 = (s_1 - ground_target_pos).norm();
                                let r2 = (s_2 - ground_target_pos).norm();

                                let delta_phi =
                                    4.0 * std::f64::consts::PI * (r1 - r2) / SENTINEL_1_WAVELENGTH;

                                *phase += delta_phi as f32;
                            }
                        }
                    },
                );
            let end_time = std::time::Instant::now();
            log::info!("Time taken: {:?}", end_time - start_time);

            rr.log("diff_phase", &rr_phase(&phase_diff))
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
        println!("Bounding box: {bounding_box:?}");
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
            let warp_function = EnhancedDelaunayWarpFunction::new(&reference, secondary, &dem);
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

            // Topographic phase: gradient approximation
            log::info!("Calculating topographic phase (approximation)");
            let s = secondary
                .metadata
                .image_annotation
                .image_information
                .range_pixel_spacing;
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
                                let radar_coords_2 =
                                    pixel_coords_to_radar_coords(sec_az, sec_rg, annotation_2);
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

            rr.log("topo_phase/approx", &rr_phase(&topo_phase_approx))
                .expect("Could not log phase to Rerun");

            log::info!("Calculating topographic phase (exact)");
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
                                let zero_doppler_2 = zero_doppler_time(sec_az, annotation_2);
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

            rr.log("topo_phase/exact", &rr_phase(&topo_phase_exact))
                .expect("Could not log phase to Rerun");
        }
    }

    #[test]
    fn warp_fn_offsets_histogram() {
        env_logger::init();
        log::info!("Starting warp_fn_offsets_histogram test.");
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151103T122546_VV_AE93-BURST",
        )
        .unwrap();
        let bounding_box = bounding_box_from_stack([&reference, &secondary]);
        log::info!("Downloading DEM");
        let dem = DEM::download_dem(bounding_box, CopernicusDemType::Cop30);
        log::info!("DEM succesfully downloaded!");

        // Build (dx, dy) offset values for histogram
        log::info!("Building (dx, dy) offset values for histogram");
        let [slant_range_size, azimuth_size] = reference.data.raster_size();
        let ref_osh = reference.orbital_state_history();
        let sec_osh = secondary.orbital_state_history();
        let radar_coords = |ground_target_pos: Vector3<f64>,
                            osh: &OrbitalStateHistory,
                            annotation: &SlcProductAnnotation|
         -> [f64; 2] {
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);

            radar_coords_to_pixel_coords(zero_doppler, annotation)
        };

        let reference_metadata = &reference.metadata;
        let secondary_metadata = &secondary.metadata;
        let n = dem.len();
        let mut delta_azimuth_coords = Vec::with_capacity(n);
        let mut delta_slant_range_coords = Vec::with_capacity(n);
        let mut latitudes = Vec::with_capacity(n);
        let mut longitudes = Vec::with_capacity(n);
        for (lat, lon) in dem.lat_lon_iter() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let rc_ref = radar_coords(pos.into(), &ref_osh, reference_metadata);
            let rc_sec = radar_coords(pos.into(), &sec_osh, secondary_metadata);

            if (rc_ref[0] >= 0.0 && rc_ref[0] < azimuth_size as f64)
                && (rc_ref[1] >= 0.0 && rc_ref[1] < slant_range_size as f64)
            {
                delta_azimuth_coords.push(rc_sec[0] - rc_ref[0]);
                delta_slant_range_coords.push(rc_sec[1] - rc_ref[1]);
                latitudes.push(lat);
                longitudes.push(lon);
            }
        }
        log::info!("Built offset values!");
        // Build ndarrays
        log::info!("Loading into arrays and writing to files...");
        let delta_azimuth_coords = Array1::from_vec(delta_azimuth_coords);
        let delta_slant_range_coords = Array1::from_vec(delta_slant_range_coords);
        let latitudes = Array1::from_vec(latitudes);
        let longitudes = Array1::from_vec(longitudes);
        let file_az = std::fs::File::create("delta_azimuth_coords.npy").unwrap();
        let file_rg = std::fs::File::create("delta_slant_range_coords.npy").unwrap();
        let file_lat = std::fs::File::create("latitudes.npy").unwrap();
        let file_lon = std::fs::File::create("longitudes.npy").unwrap();
        delta_azimuth_coords.write_npy(file_az).unwrap();
        delta_slant_range_coords.write_npy(file_rg).unwrap();
        latitudes.write_npy(file_lat).unwrap();
        longitudes.write_npy(file_lon).unwrap();
        log::info!("Done!");
    }

    #[test]
    fn warp_fn_offsets_mesh() {
        env_logger::init();
        log::info!("Starting warp_fn_offsets_mesh test.");
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151010T122546_VV_7501-BURST",
        )
        .unwrap();
        let bounding_box = bounding_box_from_stack([&reference, &secondary]);
        log::info!("Downloading DEM");
        let dem = DEM::download_dem(bounding_box, CopernicusDemType::Cop90);
        log::info!("DEM succesfully downloaded!");

        // Build (dx, dy) offset values for histogram
        log::info!("Building (dx, dy) offset values for histogram");
        let [slant_range_size, azimuth_size] = reference.data.raster_size();

        let ref_osh = &reference.continuous_orbital_state_history();
        let sec_osh = &secondary.continuous_orbital_state_history();

        log::info!(
            "DEM original length: {}, original dimensions = {}, {}",
            dem.len(),
            dem.rows(),
            dem.cols()
        );
        const FACTOR_ROWS: usize = 2;
        const FACTOR_COLS: usize = 2;
        let rows = dem.rows() / FACTOR_ROWS;
        let cols = dem.cols() / FACTOR_COLS;
        let n = rows * cols;
        log::info!("DEM sampled length: {n}, sampled dimensions = {rows}, {cols}");
        let mut delta_azimuth_coords = Vec::with_capacity(n);
        let mut delta_slant_range_coords = Vec::with_capacity(n);
        let mut vertices = Vec::with_capacity(n);

        for i in 0..rows {
            for j in 0..cols {
                let [lat, lon] = dem.get_lat_lon_at_pixel(i * FACTOR_ROWS, j * FACTOR_COLS);
                let pos = dem.get_ecef_at_lat_lon(lat, lon);
                let rc_ref = ref_osh.find_zero_doppler_state_newton_raphson(pos.into());
                let rc_sec = sec_osh.find_zero_doppler_state_newton_raphson(pos.into());

                if (rc_ref[0] >= 0.0 && rc_ref[0] < azimuth_size as f64)
                    && (rc_ref[1] >= 0.0 && rc_ref[1] < slant_range_size as f64)
                {
                    delta_azimuth_coords.push((rc_sec[0] - rc_ref[0]) as f32);
                    delta_slant_range_coords.push((rc_sec[1] - rc_ref[1]) as f32);
                } else {
                    delta_azimuth_coords.push(f32::NAN);
                    delta_slant_range_coords.push(f32::NAN);
                }
                vertices.push([lon as f32, lat as f32, 0.0]);
            }
        }
        let mut triangle_indices = Vec::with_capacity((rows - 1) * (cols - 1) * 2);
        let mut max_index = 0;
        for i in 0..(rows - 1) {
            for j in 0..(cols - 1) {
                let a = (i * cols + j) as u32;
                let b = (i * cols + j + 1) as u32;
                let c = ((i + 1) * cols + j) as u32;
                let d = ((i + 1) * cols + j + 1) as u32; // rows * cols - 1
                if d > max_index {
                    max_index = d;
                }
                triangle_indices.push([a, b, d]);
                triangle_indices.push([a, d, c]);
            }
        }
        log::info!("max_index = {max_index}");
        let colors_delta_azimuth_coords = turbo_colorized_values(&delta_azimuth_coords);
        let colors_delta_slant_range_coords = turbo_colorized_values(&delta_slant_range_coords);

        log::info!("Built offset values!");
        // Log as a 3D Mesh
        let mesh_az_offsets = rerun::Mesh3D::new(vertices.iter())
            .with_vertex_colors(colors_delta_azimuth_coords)
            .with_triangle_indices(triangle_indices.iter());
        let mesh_rg_offsets = rerun::Mesh3D::new(vertices)
            .with_vertex_colors(colors_delta_slant_range_coords)
            .with_triangle_indices(triangle_indices);
        let rr = rerun::RecordingStreamBuilder::new("warp_fn_offsets_mesh")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        rr.log_static("mesh_az_offsets", &mesh_az_offsets).unwrap();
        rr.log_static("mesh_rg_offsets", &mesh_rg_offsets).unwrap();

        log::info!("Done!");
    }

    #[test]
    fn test_flat_earth_dphi() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_flat_earth_dphi")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let mut dem_corners = dem.corners_lat_lon().to_vec();
        let first = dem_corners.first().unwrap();
        dem_corners.push(*first);
        rr.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
        let [rg_size, az_size] = primary.data.raster_size();
        let data = Array2::<f64>::from_shape_fn((az_size, rg_size / 10), |(az_index, rg_index)| {
            flat_earth_dphi(&primary, &secondary, az_index as f64, rg_index as f64, &dem)
        });
        let vector = data.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase =
                    (phase + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
                cubehelix_colormap(normalized_phase as f32).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [rg_size as u32 / 10, az_size as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log_static("dphi", &rr_image).unwrap();
    }

    #[test]
    fn test_interpolated_flat_earth_dphi() {
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_interpolated_flat_earth_dphi")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let mut dem_corners = dem.corners_lat_lon().to_vec();
        let first = dem_corners.first().unwrap();
        dem_corners.push(*first);
        rr.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([dem_corners.windows(2).flatten()])
                .with_radii([rerun::Radius::new_ui_points(2.0)])
                .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();
        let interpolator =
            FlatEarthComponentsInterpolator::from_grid_params(&primary, &secondary, &dem, 140, 20);
        let wrap_phase = |phase: f64| {
            phase - 2.0 * std::f64::consts::PI * (phase / (2.0 * std::f64::consts::PI)).floor()
        };
        let data = interpolator
            .calculate_array()
            .map(|phase| wrap_phase(*phase));
        let data = data.slice(s![.., 0..data.dim().1 / 2]).to_owned();
        let (az_size, rg_size) = data.dim();
        let vector = data.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase = phase / (2.0 * std::f64::consts::PI);
                cubehelix_colormap(normalized_phase as f32).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [rg_size as u32, az_size as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log_static("dphi", &rr_image).unwrap();
    }

    #[test]
    fn test_interpolated_flat_earth_removal() {
        env_logger::init();
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();
        let [min_lat, max_lat, min_lon, max_lon] = &primary
            .metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .get_bounding_box_lat_lon();
        let offset_lat = 0.05;
        let offset_lon = 0.05;
        let bounds = [
            min_lat - offset_lat,
            max_lat + offset_lat,
            min_lon - offset_lon,
            max_lon + offset_lon,
        ];
        let dem = DEM::download_dem(bounds, CopernicusDemType::Cop30);
        let rr = rerun::RecordingStreamBuilder::new("test_interpolated_flat_earth_removal")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let wrap_phase = |phase: f32| {
            phase - 2.0 * std::f32::consts::PI * (phase / (2.0 * std::f32::consts::PI)).floor()
        };
        let phase =
            crate::perp_baseline::coregister_and_remove_flat_phase(&primary, &secondary, &dem)
                .map(|phase| wrap_phase(*phase));
        let phase = phase.slice(s![.., 0..phase.dim().1 / 2]).to_owned();
        let (az_size, rg_size) = phase.dim();
        let vector = phase.as_slice_memory_order().unwrap().to_vec();
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&phase| {
                let normalized_phase = phase / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [rg_size as u32, az_size as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rr.log_static("phase", &rr_image).unwrap();
    }

    #[test]
    fn plot_warp_fn() {
        env_logger::init();
        let primary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151022T122546_VV_5A48-BURST",
        )
        .unwrap();
        let secondary = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1_305967_IW3_20151127T122546_VV_14CF-BURST",
        )
        .unwrap();
        let [min_lat, max_lat, min_lon, max_lon] = &primary
            .metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .get_bounding_box_lat_lon();
        let offset_lat = 0.05;
        let offset_lon = 0.05;
        let bounds = [
            min_lat - offset_lat,
            max_lat + offset_lat,
            min_lon - offset_lon,
            max_lon + offset_lon,
        ];
        let dem = DEM::download_dem(bounds, CopernicusDemType::Cop30);
        let warp_fn = EnhancedDelaunayWarpFunction::new(&primary, &secondary, &dem);
        let [ref_slant_range_dim, ref_azimuth_dim] = primary.data.raster_size();
        let mut offsets = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim / 2));
        const CHUNK_SIZE: usize = 256;
        offsets
            .axis_chunks_iter_mut(Axis(0), CHUNK_SIZE)
            .into_par_iter()
            .enumerate()
            .for_each_init(
                || warp_fn.triangulation.natural_neighbor(),
                |nn, (chunk_idx, mut chunk)| {
                    let az_offset = chunk_idx * CHUNK_SIZE;

                    for (i, mut row) in chunk.outer_iter_mut().enumerate() {
                        let ref_az = az_offset + i;

                        for (ref_rg, val) in row.iter_mut().enumerate() {
                            let ref_coords = [ref_az as f64, ref_rg as f64];
                            // let u = match nn
                            //     .interpolate(|v| v.data().secondary_coords[0], ref_coords.into())
                            // {
                            //     Some(v) => v - ref_coords[0],
                            //     None => 0.0,
                            // };
                            let v = match nn
                                .interpolate(|v| v.data().secondary_coords[1], ref_coords.into())
                            {
                                Some(v) => v - ref_coords[1],
                                None => 0.0,
                            };
                            *val = v;
                        }
                    }
                },
            );
        // let max = *img
        //     .iter()
        //     .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        //     .unwrap();
        let rr = rerun::RecordingStreamBuilder::new("warp_fn_offsets")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let tensor = rerun::Tensor::try_from(offsets.clone()).expect("Unable to create tensor.");
        rr.log("offsets", &tensor).expect("Unable to log tensor.");
        let img = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, offsets).unwrap();
        rr.log("offsets_img", &img).unwrap();
    }

    #[test]
    fn dem_mesh_test() {
        let dem = DEM::open_file("dem.tif");
        let rec = rerun::RecordingStreamBuilder::new("dem_mesh_test")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let rows = dem.rows();
        let cols = dem.cols();

        let vertex_positions: Vec<[f32; 3]> = dem.vertex_positions();
        let vertex_normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; rows * cols];
        let vertex_colors: Vec<u32> = dem.vertex_colors();
        let triangle_indices = dem.triangle_indices();
        rec.log(
            "dem_mesh3d",
            &rerun::Mesh3D::new(vertex_positions)
                .with_vertex_normals(vertex_normals)
                .with_vertex_colors(vertex_colors)
                .with_triangle_indices(triangle_indices),
        )
        .unwrap();
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

    #[test]
    fn simple_test_satellite_orbit() {
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
            let time_nanos = time.timestamp_nanos_opt().unwrap();
            rec.set_time(
                "satellite_time",
                rerun::TimeCell::from_timestamp_nanos_since_epoch(time_nanos),
            );

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
        let asset = rerun::Asset3D::from_file_path("earth.glb").unwrap();

        rec.log_static(
            "universe/earth",
            &rerun::Transform3D::from_rotation_scale(
                rerun::RotationAxisAngle::new(
                    [1.0, -1.0, -1.0],
                    rerun::Angle::from_radians(2.0 * std::f32::consts::PI / 3.0),
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
}
