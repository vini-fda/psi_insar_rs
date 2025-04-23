use std::path::Path;

use crate::{
    interpolation::{
        unit_derivative_interval_cubic_hermite_spline_interpolation,
        unit_interval_cubic_hermite_spline_interpolation,
    },
    metadata::{
        annotation_xml::{OrbitList, SlcProductAnnotation},
        orbit_xml::{EarthExplorerFile, ListOfOsvs},
    },
};
use chrono::{DateTime, Duration, TimeDelta, Utc};
use nalgebra::Vector3;

#[derive(Clone, Copy, Debug)]
pub struct ZeroDopplerState {
    /// The zero-Doppler time.
    pub time: DateTime<Utc>,
    /// The distance from the satellite to the ground target.
    pub distance_to_target: f64,
}

/// A collection of orbital state vectors (position and velocity) over time
///
/// This structure stores a time series of orbital states and provides methods
/// for interpolating state values at arbitrary times and computing related
/// orbital parameters.
pub struct OrbitalStateHistory {
    pub time: Vec<DateTime<Utc>>,
    pub position: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
}

impl OrbitalStateHistory {
    /// From a Precise Orbit Ephemerides file, and a timeframe (start_time, end_time)
    pub fn from_poe_timeframe<P: AsRef<Path>>(
        path: P,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Self {
        let eef = EarthExplorerFile::open(path);
        let osvs = eef.data_block.list_of_osvs.osv;
        // osv.utc
        let first_index: usize = osvs.iter().rposition(|osv| osv.utc <= start_time).unwrap();
        let last_index: usize = osvs.iter().position(|osv| osv.utc >= end_time).unwrap();
        let n = (last_index + 1) - first_index;
        let mut time = Vec::with_capacity(n);
        let mut position = Vec::with_capacity(n);
        let mut velocity = Vec::with_capacity(n);

        for osv in &osvs[first_index..=last_index] {
            time.push(osv.utc);
            position.push([osv.x, osv.y, osv.z].into());
            velocity.push([osv.vx, osv.vy, osv.vz].into());
        }
        Self {
            time,
            position,
            velocity,
        }
    }
    #[inline(always)]
    pub fn interp_pos_vel(&self, t: DateTime<Utc>) -> (Vector3<f64>, Vector3<f64>) {
        let time: &[DateTime<Utc>] = self.time.as_slice();
        let pos: &[Vector3<f64>] = self.position.as_slice();
        let vel: &[Vector3<f64>] = self.velocity.as_slice();
        assert!(time.len() >= 2);
        // Try to find "t" in the slice "time"
        // 1. If you can find it, return the corresponding position, velocity pair
        match time.binary_search(&t) {
            Ok(i) => return (pos[i], vel[i]), // exact match
            Err(_) => { /* continue to interpolation */ }
        }
        // 2. Otherwise, search for the previous and next position & velocity (panics if out of bounds)
        let i = time
            .windows(2)
            .position(|w| w[0] <= t && t <= w[1])
            .expect("Interpolation time out of bounds");

        let t_prev = time[i];
        let t_next = time[i + 1];
        let p_prev = pos[i];
        let p_next = pos[i + 1];
        let v_prev = vel[i];
        let v_next = vel[i + 1];
        // 3. With "t_prev" and "t_next", perform interpolation
        let total_dt = (t_next - t_prev).num_nanoseconds().unwrap() as f64;
        let dt = (t - t_prev).num_nanoseconds().unwrap() as f64;
        let alpha = dt / total_dt;

        // cubic hermite interpolation
        let pos_interp =
            unit_interval_cubic_hermite_spline_interpolation(p_prev, v_prev, p_next, v_next, alpha);
        // Derivative of Hermite spline w.r.t. time
        // (the scale factor of 1e9 / total_dt comes from the chain rule
        // and also the fact that we have to use the metric/SI system,
        // thus we convert nanoseconds to seconds, that's why 1e9 appears in the numerator)
        let vel_interp =
            1e9 * unit_derivative_interval_cubic_hermite_spline_interpolation(
                p_prev, v_prev, p_next, v_next, alpha,
            ) / total_dt;
        (pos_interp, vel_interp)
    }

    /// Calculate the zero-Doppler time for a given ground target and satellite trajectory.
    /// The zero-Doppler time is the time `t` such that the satellite's velocity vector
    /// is perpendicular to the vector pointing from the satellite to the ground target:
    ///     v(t) · (ground_target_pos - s(t)) = 0
    /// This condition implies a dot product of zero between the velocity vector and
    /// the look vector, indicating orthogonality.
    ///
    /// With the time "t" calculated, we can also obtain the slant range distance to the ground target.
    pub fn find_zero_doppler_state(&self, ground_target_pos: Vector3<f64>) -> ZeroDopplerState {
        const NUM_BISECTION_ITER: usize = 32;
        const TOLERANCE: f64 = 1e-9;
        let time = self.time.as_slice();
        assert!(time.len() >= 2);

        let f = |t: DateTime<Utc>| {
            let (sat_pos, sat_vel) = self.interp_pos_vel(t);
            let normalized_displacement = (ground_target_pos - sat_pos).normalize();
            sat_vel.dot(&normalized_displacement)
        };

        let state = |t: DateTime<Utc>| {
            let (sat_pos, _) = self.interp_pos_vel(t);
            let distance_to_target = (ground_target_pos - sat_pos).norm();
            return ZeroDopplerState {
                time: t,
                distance_to_target,
            };
        };

        // Step 1: Search for a sign change across time intervals
        for i in 0..time.len() - 1 {
            let t0 = time[i];
            let t1 = time[i + 1];
            let f0 = f(t0);
            let f1 = f(t1);

            if f0 * f1 <= 0.0 {
                // Step 2: Narrow down using bisection over datetime
                let mut left = t0;
                let mut right = t1;
                for _ in 0..NUM_BISECTION_ITER {
                    let mid = left + (right - left) / 2;
                    let fm = f(mid);

                    if fm.abs() < TOLERANCE {
                        return state(mid);
                    } else if f0 * fm < 0.0 {
                        right = mid;
                    } else {
                        left = mid;
                    }
                }
                let mid = left + (right - left) / 2;
                return state(mid);
            }
        }

        panic!("Zero-Doppler point not found in trajectory window.");
    }

    /// Creates an upsampled version of Self, inserting k >= 1 samples inbetween every two samples in the original
    pub fn interp_n(&self, k: usize) -> Self {
        assert!(k >= 1);
        let n = self.time.len();
        let n_ups = k * (n - 1) + n;
        let mut time_ups = vec![DateTime::<Utc>::default(); n_ups];
        for i in 0..(n - 1) {
            time_ups[(k + 1) * i] = self.time[i];
            let delta_t = (self.time[i + 1] - self.time[i]).num_nanoseconds().unwrap() as f64
                / (k as f64 + 1.0);
            for j in 1..=k {
                let total_delta_t = delta_t * j as f64;
                let total_delta_t = TimeDelta::nanoseconds(total_delta_t.round() as i64);
                time_ups[(k + 1) * i + j] = self.time[i] + total_delta_t;
            }
        }
        time_ups[(k + 1) * (n - 1)] = self.time[n - 1];
        let mut position_ups = Vec::with_capacity(n_ups);
        let mut velocity_ups = Vec::with_capacity(n_ups);
        for t in &time_ups {
            let (pos, vel) = self.interp_pos_vel(*t);
            position_ups.push(pos);
            velocity_ups.push(vel);
        }
        Self {
            time: time_ups,
            position: position_ups,
            velocity: velocity_ups,
        }
    }
}

impl From<OrbitList> for OrbitalStateHistory {
    fn from(list: OrbitList) -> Self {
        Self::from(&list)
    }
}

impl From<&OrbitList> for OrbitalStateHistory {
    fn from(orbit_list: &OrbitList) -> Self {
        let mut time = Vec::with_capacity(orbit_list.count as usize);
        let mut position = Vec::with_capacity(orbit_list.count as usize);
        let mut velocity = Vec::with_capacity(orbit_list.count as usize);

        for orbit in orbit_list.orbit.iter() {
            time.push(orbit.time);
            position.push(orbit.position.into());
            velocity.push(orbit.velocity.into());
        }

        OrbitalStateHistory {
            time,
            position,
            velocity,
        }
    }
}

impl From<ListOfOsvs> for OrbitalStateHistory {
    fn from(list: ListOfOsvs) -> Self {
        Self::from(&list)
    }
}

impl From<&ListOfOsvs> for OrbitalStateHistory {
    fn from(osv_list: &ListOfOsvs) -> Self {
        let mut time = Vec::with_capacity(osv_list.count as usize);
        let mut position = Vec::with_capacity(osv_list.count as usize);
        let mut velocity = Vec::with_capacity(osv_list.count as usize);

        for osv in osv_list.osv.iter() {
            time.push(osv.utc);
            position.push([osv.x, osv.y, osv.z].into());
            velocity.push([osv.vx, osv.vy, osv.vz].into());
        }

        OrbitalStateHistory {
            time,
            position,
            velocity,
        }
    }
}

pub fn radar_coords_slc_annotation_to_pixel(
    zero_doppler: ZeroDopplerState,
    annotation: &SlcProductAnnotation,
) -> Option<(usize, usize)> {
    /// Speed of light in m/s
    const C_LIGHT: f64 = 299_792_458.0;
    let slant_range_time = annotation
        .image_annotation
        .image_information
        .slant_range_time;
    let near_edge_slant_range = 0.5 * C_LIGHT * slant_range_time;
    let t_start = annotation
        .image_annotation
        .image_information
        .product_first_line_utc_time;
    let stop_time = annotation.ads_header.stop_time;
    let prf = annotation
        .general_annotation
        .downlink_information_list
        .downlink_information
        .prf;
    // ADC Sampling Rate (?) TODO: LEARN
    // let fs = annotation
    //     .general_annotation
    //     .product_information
    //     .range_sampling_rate;
    let range_spacing = annotation
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let azimuth_time_interval = annotation
        .image_annotation
        .image_information
        .azimuth_time_interval;

    // Number of azimuth lines (rows)
    let num_rows = annotation
        .image_annotation
        .image_information
        .number_of_lines;
    // Number of range samples per azimuth line (columns)
    let num_cols = annotation
        .image_annotation
        .image_information
        .number_of_samples;

    // Time difference in seconds
    let delta_time = zero_doppler.time.signed_duration_since(t_start);
    let delta_time_secs = delta_time.num_microseconds().unwrap() as f64 * 1.0e-6;

    let total_delta_time = stop_time.signed_duration_since(t_start);
    let total_delta_time_secs = total_delta_time.num_microseconds().unwrap() as f64 * 1.0e-6;

    // Row calculation: (time - t_start) * prf (DOES NOT WORK)
    // TODO: Understand why this doesn't work
    // let row = delta_time_secs * prf;

    // Linear interpolation (this seems to work better)
    // let row = (delta_time_secs / total_delta_time_secs) * (num_rows as f64);

    // Calculation using azimuth_time_interval (similar to linear interp)
    // This is what the official SNAP microwave toolbox performs:
    // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-io/src/main/java/eu/esa/sar/io/sentinel1/Sentinel1Level1Directory.java
    // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/support/SARPosition.java
    let azimuth_index = delta_time_secs / azimuth_time_interval;

    // Column calculation:
    let slant_range_index =
        (zero_doppler.distance_to_target - near_edge_slant_range) / range_spacing;
    println!(
        "DEBUG | (azimuth, slant range) = {:?}, delta_r = {}, r_0 = {}",
        (azimuth_index, slant_range_index),
        range_spacing,
        near_edge_slant_range
    );
    // Check bounds and return if within image
    if azimuth_index >= 0.0 && slant_range_index >= 0.0 {
        let row_idx = azimuth_index.round() as usize;
        let col_idx = slant_range_index.round() as usize;

        if row_idx < num_rows && col_idx < num_cols {
            return Some((row_idx, col_idx));
        }
    }

    None
}

pub fn radar_coords_slc_annotation_to_pixel_f32(
    zero_doppler: ZeroDopplerState,
    annotation: &SlcProductAnnotation,
) -> (f32, f32) {
    /// Speed of light in m/s
    const C_LIGHT: f64 = 299_792_458.0;
    let slant_range_time = annotation
        .image_annotation
        .image_information
        .slant_range_time;
    let near_edge_slant_range = 0.5 * C_LIGHT * slant_range_time;
    let t_start = annotation
        .image_annotation
        .image_information
        .product_first_line_utc_time;

    let range_spacing = annotation
        .image_annotation
        .image_information
        .range_pixel_spacing;
    let azimuth_time_interval = annotation
        .image_annotation
        .image_information
        .azimuth_time_interval;

    // Time difference in seconds
    let delta_time = zero_doppler.time.signed_duration_since(t_start);
    let delta_time_secs = delta_time.num_microseconds().unwrap() as f64 * 1.0e-6;

    // Calculation using azimuth_time_interval (similar to linear interp)
    // This is what the official SNAP microwave toolbox performs:
    // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-io/src/main/java/eu/esa/sar/io/sentinel1/Sentinel1Level1Directory.java
    // - https://github.com/senbox-org/microwave-toolbox/blob/ff89cf020b8c426c101502f3187c2b2b389722c0/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/support/SARPosition.java
    let azimuth_index = delta_time_secs / azimuth_time_interval;

    // Column calculation:
    let slant_range_index =
        (zero_doppler.distance_to_target - near_edge_slant_range) / range_spacing;
    (azimuth_index as f32, slant_range_index as f32)
}

#[cfg(test)]
mod manual_tests_satellite_orbit {
    use std::f32::consts::PI;

    use nalgebra::{Matrix3, Unit, Vector3};
    use rerun::Color;

    use super::OrbitalStateHistory;
    use crate::{
        dem::DEM,
        geodesy::geodetic_to_ecef,
        metadata::{
            annotation_xml::{OrbitList, SlcProductAnnotation},
            orbit_xml::EarthExplorerFile,
        },
        satellite_orbit::{
            radar_coords_slc_annotation_to_pixel, radar_coords_slc_annotation_to_pixel_f32,
        },
    };

    /// Reads the first `<orbitList>` element found in the XML file at `path`.
    fn read_orbit_list_from_file(path: &str) -> OrbitList {
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);

        let xml_de = &mut quick_xml::de::Deserializer::from_reader(reader);
        // Parse the XML into our Product struct
        let result: Result<SlcProductAnnotation, _> = serde_path_to_error::deserialize(xml_de);
        let slc_product_annotation = match result {
            Ok(val) => val,
            Err(err) => {
                let path = err.path().to_string();
                panic!("Error parsing XML\nError path: {}\nError: {}", path, err);
            }
        };
        slc_product_annotation.general_annotation.orbit_list
    }

    #[test]
    fn read_orbit_list() {
        let orbit_list = read_orbit_list_from_file("src/metadata/test_data/annotation_example.xml");
        // println!("orbitList = {:?}", orbit_list);
    }

    #[test]
    fn find_zero_doppler_from_orbit_list() {
        let annotation =
            SlcProductAnnotation::open("src/metadata/test_data/annotation_example.xml");
        // let orbit_list = read_orbit_list_from_file("src/metadata/test_data/annotation_example.xml");
        let start_time = annotation.ads_header.start_time;
        let end_time = annotation.ads_header.stop_time;
        let osh = OrbitalStateHistory::from_poe_timeframe("orbit.EOF", start_time, end_time);
        let dem = DEM::open_file("dem.tif");
        let [lat, lon] = [19.49831428810679, -98.59301000370277];
        // let [lat, lon] = [19.46306674963845, -99.36782713184502];
        let pos = dem.get_ecef_at_lat_lon(lat, lon);
        // let ground_target_pos = Vector3::<f64>::new(-903403.94, -5955250.0, 2099050.0);
        let ground_target_pos = Vector3::<f64>::new(pos[0] as f64, pos[1] as f64, pos[2] as f64);

        let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
        println!("Zero-Doppler time = {:?}", zero_doppler);
        if let Some((row, col)) = radar_coords_slc_annotation_to_pixel(zero_doppler, &annotation) {
            println!("Found pixel at {row}, {col}");
        }
    }

    #[test]
    #[ignore]
    fn test_backgeocoding() {
        let rr = rerun::RecordingStreamBuilder::new("test_backgeocoding")
            .connect_tcp()
            .expect("Could not connect to local Rerun instance.");
        let annotation =
            SlcProductAnnotation::open("src/metadata/test_data/annotation_example.xml");
        let osh = OrbitalStateHistory::from(&annotation.general_annotation.orbit_list);
        let dem = DEM::open_file("dem.tif");

        let mut points = vec![];
        for (i, j, lat, lon, height) in dem.indexed_lat_lon_height() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let ground_target_pos =
                Vector3::<f64>::new(pos[0] as f64, pos[1] as f64, pos[2] as f64);
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
            let (row, col) = radar_coords_slc_annotation_to_pixel_f32(zero_doppler, &annotation);

            points.push([col as f32, row as f32]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors(dem.vertex_colors())
            .with_radii([10.0]);
        rr.log_static("backgeocoded_points", &points).unwrap();

        // Log ground control points
        let lat_lon_gcps = [
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

        let mut points = vec![];
        for [lat, lon] in lat_lon_gcps {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let ground_target_pos =
                Vector3::<f64>::new(pos[0] as f64, pos[1] as f64, pos[2] as f64);
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
            let (row, col) = radar_coords_slc_annotation_to_pixel_f32(zero_doppler, &annotation);

            points.push([col as f32, row as f32]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors([rerun::Color::from_rgb(255, 122, 100)])
            .with_radii([10.0]);
        rr.log_static("backgeocoded_gcps", &points).unwrap();

        // part 2
        let annotation = SlcProductAnnotation::open(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE/annotation/s1a-iw3-slc-vv-20151010t122546-20151010t122550-008090-00b578-001.xml",
        );
        let osh = OrbitalStateHistory::from(&annotation.general_annotation.orbit_list);
        let dem = DEM::open_file("dem.tif");

        let mut points = vec![];
        for (i, j, lat, lon, height) in dem.indexed_lat_lon_height() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let ground_target_pos =
                Vector3::<f64>::new(pos[0] as f64, pos[1] as f64, pos[2] as f64);
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
            let (row, col) = radar_coords_slc_annotation_to_pixel_f32(zero_doppler, &annotation);

            points.push([col as f32, row as f32]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors(dem.vertex_colors().iter().map(|&n| n & 0x0000FFFF))
            .with_radii([10.0]);
        rr.log_static("backgeocoded_points_2", &points).unwrap();

        let mut points = vec![];
        for [lat, lon] in lat_lon_gcps {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let ground_target_pos =
                Vector3::<f64>::new(pos[0] as f64, pos[1] as f64, pos[2] as f64);
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
            let (row, col) = radar_coords_slc_annotation_to_pixel_f32(zero_doppler, &annotation);

            points.push([col as f32, row as f32]);
        }
        let points = rerun::Points2D::new(points)
            .with_colors([rerun::Color::from_rgb(122, 255, 100)])
            .with_radii([10.0]);
        rr.log_static("backgeocoded_gcps_2", &points).unwrap();
    }

    #[test]
    #[ignore]
    fn simple() {
        let rec = rerun::RecordingStreamBuilder::new("simple_test_satellite_orbit")
            .recording_id("my_shared_recording")
            .connect_tcp()
            .expect("Could not connect to local Rerun instance.");
        let orbit_list = read_orbit_list_from_file("src/metadata/test_data/annotation_example.xml");
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
            let pos = orbital_history.position[k];
            let vel = orbital_history.velocity[k];
            let time = orbital_history.time[k];
            let look_rot3x3 = look_rotation_from_velocity_and_position(
                vel.map(|c| c as f32),
                pos.map(|c| c as f32),
            );
            rec.set_time_nanos("satellite_time", time.timestamp_nanos_opt().unwrap());

            let pos = rerun::Position3D::new(pos.x as f32, pos.y as f32, pos.z as f32);
            rec.log(
                "satellite_position",
                &rerun::Points3D::new([pos])
                    .with_colors([Color::WHITE])
                    .with_radii([1500.0]),
            )
            .unwrap();
            let arrow_vel =
                rerun::Arrows3D::from_vectors([(vel.x as f32, vel.y as f32, vel.z as f32)])
                    .with_origins([pos]);
            rec.log("satellite_velocity", &arrow_vel).unwrap();

            // Pinhole camera
            let ground_target = geodetic_to_ecef(19.49831428810679, -98.59301000370277, 0.0);
            let ground_target_vec3 = Vector3::<f32>::from(ground_target.map(|val| val as f32));
            let pos_vec3 = Vector3::<f32>::new(pos.x(), pos.y(), pos.z());
            let rot3x3 = look_at_ground_target(pos_vec3, ground_target_vec3);
            rec.log(
                "universe/camera",
                &rerun::Transform3D::from_translation_mat3x3(
                    [pos.x(), pos.y(), pos.z()],
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
        let asset = rerun::Asset3D::from_file("earth.glb").unwrap();

        rec.log_static(
            "universe/earth",
            &rerun::Transform3D::from_rotation_scale(
                rerun::RotationAxisAngle::new(
                    [1.0, -1.0, -1.0],
                    rerun::Angle::from_radians(2.0 * PI / 3.0),
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
        let (rows, cols) = dem.array_dim();
        let vertex_positions: Vec<[f32; 3]> = dem.vertex_positions();
        let vertex_normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; rows * cols];
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
        let radar_xyz: Vec<_> = [
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
        ]
        .iter()
        .map(|&[lat, lon]| dem.get_ecef_at_lat_lon(lat, lon).map(|val| val as f32))
        .collect();

        rec.log_static("geo_points", &rerun::Points3D::new(radar_xyz))
            .unwrap();
    }

    fn look_rotation_from_velocity_and_position(v: Vector3<f32>, p: Vector3<f32>) -> Matrix3<f32> {
        let forward = Unit::new_normalize(v);
        let radial = Unit::new_normalize(p);

        // Remove component of radial in the direction of forward (Gram-Schmidt)
        let up_raw = radial.into_inner();
        let up = Unit::new_normalize(up_raw - forward.into_inner() * up_raw.dot(&forward));

        // Right vector
        let right = Unit::new_normalize(up.cross(&forward));

        // Compose rotation matrix from right, up, forward as columns
        let rot_matrix =
            Matrix3::from_columns(&[right.into_inner(), up.into_inner(), forward.into_inner()]);

        rot_matrix
    }

    /// Constructs a rotation matrix that orients an object at `p_sat` to point toward `p_target`.
    pub fn look_at_ground_target(p_sat: Vector3<f32>, p_target: Vector3<f32>) -> Matrix3<f32> {
        let forward_vec = p_target - p_sat;
        let forward = Unit::new_normalize(forward_vec);

        let up_raw = Unit::new_normalize(p_sat).into_inner(); // Radial from Earth center
        let up = Unit::new_normalize(up_raw - forward.into_inner() * up_raw.dot(&forward));
        let right = Unit::new_normalize(up.cross(&forward));

        let rot_matrix =
            Matrix3::from_columns(&[right.into_inner(), up.into_inner(), forward.into_inner()]);

        rot_matrix
    }
}

// //! Log different transforms between three arrows.

// use std::f32::consts::TAU;

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let rec = rerun::RecordingStreamBuilder::new("rerun_example_transform3d").spawn()?;

//     let arrow = rerun::Arrows3D::from_vectors([(0.0, 1.0, 0.0)]).with_origins([(0.0, 0.0, 0.0)]);

//     rec.log("base", &arrow)?;

//     rec.log(
//         "base/translated",
//         &rerun::Transform3D::from_translation([1.0, 0.0, 0.0]),
//     )?;

//     rec.log("base/translated", &arrow)?;

//     rec.log(
//         "base/rotated_scaled",
//         &rerun::Transform3D::from_rotation_scale(
//             rerun::RotationAxisAngle::new([0.0, 0.0, 1.0], rerun::Angle::from_radians(TAU / 8.0)),
//             rerun::Scale3D::from(2.0),
//         ),
//     )?;

//     rec.log("base/rotated_scaled", &arrow)?;

//     Ok(())
// }
