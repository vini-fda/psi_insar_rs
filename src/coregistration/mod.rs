use std::f32;

use nalgebra::{ComplexField, Vector3};
use ndarray::Array2;

use crate::{
    dem_gdal::DEMGdal,
    metadata::annotation_xml::SlcProductAnnotation,
    satellite_orbit::{OrbitalStateHistory, radar_coords_to_pixel_coords},
    sentinel::Sentinel1SlcBurst,
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
    reference: &Sentinel1SlcBurst,
    secondary: &Sentinel1SlcBurst,
    dem: &DEMGdal,
) -> Array2<[u8; 4]> {
    let [azimuth_size, slant_range_size] = reference.data.raster_size();
    let ref_osh = reference.orbital_state_history();
    let sec_osh = secondary.orbital_state_history();
    let radar_coords = |ground_target_pos: Vector3<f64>,
                        osh: &OrbitalStateHistory,
                        annotation: &SlcProductAnnotation|
     -> [f32; 2] {
        let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);
        radar_coords_to_pixel_coords(zero_doppler, annotation)
    };
    let mut rho = Array2::<[f32; 2]>::default((slant_range_size, azimuth_size));
    for (_, _, lat, lon, _) in dem.indexed_lat_lon_height() {
        let pos = dem.get_ecef_at_lat_lon(lat, lon);
        let rc_ref = radar_coords(pos.into(), &ref_osh, &reference.metadata);
        let rc_sec = radar_coords(pos.into(), &sec_osh, &secondary.metadata);

        if (rc_ref[0] >= 0.0 && rc_ref[0] < slant_range_size as f32)
            && (rc_ref[1] >= 0.0 && rc_ref[1] < azimuth_size as f32)
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

    use crate::{
        dem_gdal::DEMGdal, satellite_orbit::radar_coords_to_pixel_coords,
        sentinel::Sentinel1SlcBurst,
    };

    use super::compute_warp_function;

    #[test]
    fn testfn_dem() {
        let dem = DEMGdal::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_warp_fn_dem")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance");

        // Collect vertices and heights
        let mut vertices = Vec::new();
        let mut heights = Vec::new();
        for (_, _, lat, lon, height) in dem.indexed_lat_lon_height() {
            vertices.push([lon as f32, lat as f32, 0.0]);
            heights.push(height as f32);
        }

        // Create triangle indices for a grid
        let (rows, cols) = dem.array_dim();
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
        let dem = DEMGdal::open_file("dem.tif");
        let rr = rerun::RecordingStreamBuilder::new("test_warp_fn_dem_rgb")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance");

        // Collect vertices and heights
        let mut vertices = Vec::new();
        let mut lons = Vec::new();
        let mut lats = Vec::new();
        for (_, _, lat, lon, _) in dem.indexed_lat_lon_height() {
            vertices.push([lon as f32, lat as f32, 0.0 as f32]);
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
        let (rows, cols) = dem.array_dim();
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
        let dem = DEMGdal::open_file("dem.tif");
        let rho = compute_warp_function(&reference, &secondary, &dem);
        let (rows, cols) = rho.dim();
        let rho = rho.slice(s![0..rows, 0..cols / 2]).to_owned();

        let rr = rerun::RecordingStreamBuilder::new("test_warp_function")
            .connect_grpc()
            .expect("Could not connect to local Rerun instance.");
        let (rows, cols) = rho.dim();
        let (v, offset) = rho.into_raw_vec_and_offset();
        let img = rerun::Image::from_rgba32(v.as_flattened(), [cols as u32, rows as u32]);
        rr.log("warp_fn", &img).expect("Could not finish recording");
    }

    #[test]
    fn testfn_dem_radar_coords() {
        let dem = DEMGdal::open_file("dem.tif");
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
        for (_, _, lat, lon, _) in dem.indexed_lat_lon_height() {
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
        let (rows, cols) = dem.array_dim();
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
}
