use crate::{
    constants::SENTINEL_1_WAVELENGTH,
    coregistration::{
        deramping::DerampSlcBurst,
        interpolation2d::{KnabSincKernel, interpolate_2d},
    },
    dem_gdal::DEMGdal,
    geodesy::local_normal,
    perp_baseline::{EnhancedDelaunayWarpFunction, perp_baseline},
    satellite_orbit::pixel_coords_to_radar_coords,
    sentinel::Sentinel1SlcBurst,
};
use nalgebra::Vector3;
use ndarray::Array2;
use ndarray_npy::WriteNpyExt;

pub fn coregister_and_remove_flat_phase(
    reference: &Sentinel1SlcBurst,
    secondary_imgs: &[Sentinel1SlcBurst],
    dem: &DEMGdal,
) {
    let deramp = DerampSlcBurst::new();
    let reference_img = deramp.apply_forward(&reference);

    // The indices in the domain of the reference image
    let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();
    let indices = (0..ref_slant_range_dim)
        .flat_map(|ref_rg| (0..ref_azimuth_dim).map(move |ref_az| [ref_az, ref_rg]));
    let indices_usize: Vec<[usize; 2]> = indices.clone().collect();

    let osh_1 = reference.orbital_state_history();
    let annotation_1 = &reference.metadata;
    let s = annotation_1
        .image_annotation
        .image_information
        .range_pixel_spacing;

    for secondary in secondary_imgs {
        let secondary_name = &secondary.granule_id.raw_filename;
        println!("Processing {}...", secondary_name);
        let warp_function = EnhancedDelaunayWarpFunction::new(reference, secondary, dem);
        let mut coregistered_secondary_img = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));
        let secondary_img = deramp.apply_forward(&secondary);

        warp_function
            .map(indices.clone().map(|[az, rg]| [az as f64, rg as f64]))
            .enumerate()
            .filter_map(|(i, coords)| coords.map(|c| (i, c)))
            .for_each(|(i, [sec_az, sec_rg])| {
                let [ref_az, ref_rg] = indices_usize[i];

                let value = interpolate_2d(
                    secondary_img.view(),
                    sec_az as f32,
                    sec_rg as f32,
                    &KnabSincKernel::default(),
                );
                if let Some(resampled_value) = coregistered_secondary_img.get_mut([ref_az, ref_rg])
                {
                    *resampled_value = value;
                }
            });

        let mut phase_diff = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));
        for i in 0..ref_azimuth_dim {
            for j in 0..ref_slant_range_dim {
                phase_diff[[i, j]] =
                    reference_img[[i, j]].arg() - coregistered_secondary_img[[i, j]].arg();
            }
        }

        let nn = warp_function.triangulation.natural_neighbor();
        let osh_2 = secondary.orbital_state_history();
        let annotation_2 = &secondary.metadata;

        for ref_az in 0..ref_azimuth_dim {
            let mut accumulated_dphi = 0.0;
            let mut prev_height = None;
            for ref_rg in 0..ref_slant_range_dim {
                let ref_coords = [ref_az as f64, ref_rg as f64];
                // Helper function to compute mapped coordinate for a given dimension
                let compute_mapped_coord = |dimension: usize| {
                    nn.interpolate(|v| v.data().secondary_coords[dimension], ref_coords.into())
                };
                let sec_az = compute_mapped_coord(0);
                let sec_rg = compute_mapped_coord(1);

                // Now compute the ground target position
                let ground_target_lat = nn.interpolate(|v| v.data().lat, ref_coords.into());
                let ground_target_lon = nn.interpolate(|v| v.data().lon, ref_coords.into());

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

                let ground_target_pos =
                    Vector3::from(dem.get_ecef_at_lat_lon(ground_target_lat, ground_target_lon));

                let radar_coords_1 =
                    pixel_coords_to_radar_coords(ref_az as f64, ref_rg as f64, annotation_1);
                let radar_coords_2 =
                    pixel_coords_to_radar_coords(sec_az as f64, sec_rg as f64, annotation_2);
                let (s_1, _) = osh_1.interp_pos_vel(radar_coords_1.time);
                let (s_2, _) = osh_2.interp_pos_vel(radar_coords_2.time);

                let bperp = perp_baseline(&s_1, &s_2, &ground_target_pos);
                let l = (s_1 - ground_target_pos).normalize();
                let normal =
                    Vector3::from(dem.get_normal_at_lat_lon(ground_target_lat, ground_target_lon));
                let theta = l.dot(&normal).acos();

                phase_diff[[ref_az, ref_rg]] -= accumulated_dphi as f32;

                let height = dem.get_height_at_lat_lon(ground_target_lat, ground_target_lon);
                let r = (s_1 - ground_target_pos).norm();

                if let Some(prev_height) = prev_height {
                    let height_diff = height - prev_height;
                    let height_diff_dphi = (4.0 * std::f64::consts::PI * bperp * height_diff)
                        / (r * SENTINEL_1_WAVELENGTH * theta.sin());
                    accumulated_dphi -= height_diff_dphi;
                }

                prev_height = Some(height);

                let dphi = (4.0 * std::f64::consts::PI * bperp * s)
                    / (r * SENTINEL_1_WAVELENGTH * theta.tan());
                accumulated_dphi += dphi;
            }
        }

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
    use crate::dem_gdal::CopernicusDemType;

    use super::*;

    #[test]
    fn test_stack_interferograms() {
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
        let dem = DEMGdal::download_dem(bounding_box, CopernicusDemType::Cop30);
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
}
