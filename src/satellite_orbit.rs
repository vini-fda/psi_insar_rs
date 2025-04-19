use crate::metadata::annotation_xml::OrbitList;
use chrono::{DateTime, Duration, Utc};
use nalgebra::Vector3;

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

#[cfg(test)]
mod manual_tests_satellite_orbit {
    use std::f32::consts::{PI, TAU};

    use rerun::Color;

    use super::OrbitalStateHistory;
    use crate::{
        geodesy::geodetic_to_ecef,
        metadata::annotation_xml::{OrbitList, SlcProductAnnotation},
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
    #[ignore]
    fn simple() {
        let rec = rerun::RecordingStreamBuilder::new("simple_test_satellite_orbit")
            .connect_tcp()
            .expect("Could not connect to local Rerun instance.");
        let orbit_list = read_orbit_list_from_file("src/metadata/test_data/annotation_example.xml");
        let orbital_history = OrbitalStateHistory::from(orbit_list);
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

            // TODO: Pinhole camera
            // rec.log(
            //     "universe/camera",
            //     &rerun::Transform3D::from_translation_rotation(translation, rot),
            // )
            // .unwrap();

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

        // XYZ OF POINTS
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
        .map(|&[lat, lon]| geodetic_to_ecef(lat, lon, 0.0))
        .collect();

        rec.log_static("geo_points", &rerun::Points3D::new(radar_xyz))
            .unwrap();

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
