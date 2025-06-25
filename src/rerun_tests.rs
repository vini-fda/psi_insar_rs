#[cfg(test)]
mod tests {
    use chrono::Utc;
    use nalgebra::{Complex, Vector3};
    use ndarray::{Array2, s};
    use num_complex::ComplexFloat;
    use rerun::{Image, RecordingStream};

    use crate::{
        dem::DEM,
        dem_gdal::DEMGdal,
        geodesy::geodetic_to_ecef,
        metadata::annotation_xml::SlcProductAnnotation,
        satellite_orbit::{OrbitalStateHistory, radar_coords_to_pixel_coords},
        sentinel::Sentinel1SlcBurst,
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
}
