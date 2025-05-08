//! Compute the warp function \rho between two SLC images, in the domain of the reference image.
//!
//! The warp function \rho is a function that maps the coordinates of the reference image to the coordinates of the secondary image.
//!
//! i.e. \rho: \Omega_ref -> \Omega_sec
//!
//! where \Omega_ref and \Omega_sec are the domains of the reference and secondary images, respectively.
//!
//! The warp function \rho is defined as:
//!
//! \rho(x, y) = (x, y) + \Delta x, \Delta y)
//!
//! where \Delta x and \Delta y are the range and azimuth offsets between the two images.

use nalgebra::Vector3;
use ndarray::Array2;
use num_complex::Complex;
use spade::{DelaunayTriangulation, HasPosition, NaturalNeighbor, Triangulation};
use std::iter::Map;

use crate::{
    dem::DEM,
    metadata::annotation_xml::SlcProductAnnotation,
    satellite_orbit::{OrbitalStateHistory, radar_coords_slc_annotation_to_pixel_f32},
    sentinel::Sentinel1SlcBurst,
};

use super::{
    deramping::DerampSlcBurst,
    interpolation2d::{BilinearKernel, KnabSincKernel, interpolate_2d},
};

/// A trait for a warp function.
pub trait WarpFunction {
    /// Maps reference image coordinates to their corresponding coordinates in the secondary image.
    ///
    /// Returns `None` if the input coordinates are outside the convex hull of the triangulation.
    fn map(&self, ref_coords: [f32; 2]) -> Option<[f32; 2]>;

    /// Maps multiple reference image coordinates to their corresponding coordinates in the secondary image.
    ///
    /// Returns an iterator over optional coordinates, where `None` indicates that the corresponding
    /// input point was outside the convex hull of the triangulation.
    fn map_many<'a, I>(
        &'a self,
        ref_coords: I,
    ) -> Map<I::IntoIter, impl FnMut(I::Item) -> Option<[f32; 2]> + 'a>
    where
        I: IntoIterator<Item = [f32; 2]>,
        I::IntoIter: 'a;
}

pub struct DelaunayWarpFunction {
    triangulation: DelaunayTriangulation<ExactMapping>,
}

/// A point which contains a single exact mapping of the reference coordinates to the secondary coordinates.
#[derive(Debug, Clone, Copy)]
pub struct ExactMapping {
    pub reference_coords: [f32; 2],
    pub secondary_coords: [f32; 2],
}

impl HasPosition for ExactMapping {
    type Scalar = f32;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        self.reference_coords.into()
    }
}

impl DelaunayWarpFunction {
    /// Computes the warp function \rho between two SLC images, in the domain of the reference image.
    pub fn new(reference: &Sentinel1SlcBurst, secondary: &Sentinel1SlcBurst, dem: &DEM) -> Self {
        let [azimuth_size, slant_range_size] = reference.data.raster_size();
        let ref_osh = reference.orbital_state_history();
        let sec_osh = secondary.orbital_state_history();
        let radar_coords = |ground_target_pos: Vector3<f64>,
                            osh: &OrbitalStateHistory,
                            annotation: &SlcProductAnnotation|
         -> [f32; 2] {
            let zero_doppler = osh.find_zero_doppler_state(ground_target_pos);

            radar_coords_slc_annotation_to_pixel_f32(zero_doppler, annotation)
        };
        let mut triangulation: DelaunayTriangulation<_> = DelaunayTriangulation::new();
        for (_, _, lat, lon, _) in dem.indexed_lat_lon_height() {
            let pos = dem.get_ecef_at_lat_lon(lat, lon);
            let rc_ref = radar_coords(pos.into(), &ref_osh, &reference.metadata);
            let rc_sec = radar_coords(pos.into(), &sec_osh, &secondary.metadata);

            if (rc_ref[0] >= 0.0 && rc_ref[0] < slant_range_size as f32)
                && (rc_ref[1] >= 0.0 && rc_ref[1] < azimuth_size as f32)
            {
                let mapping = ExactMapping {
                    reference_coords: rc_ref,
                    secondary_coords: rc_sec,
                };
                triangulation
                    .insert(mapping)
                    .expect("Failed to insert mapping");
            }
        }

        Self { triangulation }
    }
}

impl WarpFunction for DelaunayWarpFunction {
    /// Maps reference image coordinates to their corresponding coordinates in the secondary image.
    ///
    /// This method uses natural neighbor interpolation to compute the exact position in the secondary
    /// image that corresponds to a given point in the reference image. The mapping represents the
    /// direct transformation from reference to secondary image coordinates.
    ///
    /// # Arguments
    ///
    /// * `ref_coords` - A 2D array containing the [azimuth, range] coordinates in the reference image
    ///
    /// # Returns
    ///
    /// * `Some([azimuth, range])` - The coordinates in the secondary image that correspond to the
    ///   input reference coordinates
    /// * `None` - If the input coordinates are outside the convex hull of the triangulation
    fn map(&self, ref_coords: [f32; 2]) -> Option<[f32; 2]> {
        let nn = self.triangulation.natural_neighbor();

        // Helper function to compute mapped coordinate for a given dimension
        let compute_mapped_coord = |dimension: usize| {
            nn.interpolate(|v| v.data().secondary_coords[dimension], ref_coords.into())
        };

        // Compute mapped coordinates for both azimuth (0) and range (1) dimensions
        let mapped_azimuth = compute_mapped_coord(0)?;
        let mapped_range = compute_mapped_coord(1)?;

        Some([mapped_azimuth, mapped_range])
    }

    /// Maps multiple reference image coordinates to their corresponding coordinates in the secondary image.
    ///
    /// This method efficiently computes the mapped coordinates for multiple points by reusing the
    /// natural neighbor interpolation object. It's more efficient than calling `map()` multiple times
    /// as it avoids recreating the interpolation object for each point.
    ///
    /// # Arguments
    ///
    /// * `ref_coords` - An iterator of 2D arrays, each containing [azimuth, range] coordinates in the reference image
    ///
    /// # Returns
    ///
    /// An iterator over optional coordinates, where each element is:
    /// * `Some([azimuth, range])` - The coordinates in the secondary image that correspond to the
    ///   input reference coordinates
    /// * `None` - If the corresponding input coordinates are outside the convex hull of the triangulation
    fn map_many<'a, I>(
        &'a self,
        ref_coords: I,
    ) -> Map<I::IntoIter, impl FnMut(I::Item) -> Option<[f32; 2]> + 'a>
    where
        I: IntoIterator<Item = [f32; 2]>,
        I::IntoIter: 'a,
    {
        let nn = self.triangulation.natural_neighbor();
        ref_coords.into_iter().map(move |point| {
            let compute_mapped_coord = |dimension: usize| {
                nn.interpolate(|v| v.data().secondary_coords[dimension], point.into())
            };
            let mapped_azimuth = compute_mapped_coord(0)?;
            let mapped_range = compute_mapped_coord(1)?;
            Some([mapped_azimuth, mapped_range])
        })
    }
}

pub fn resample_secondary_to_reference(
    reference: &Sentinel1SlcBurst,
    secondary: &Sentinel1SlcBurst,
    dem: &DEM,
) -> Array2<Complex<f32>> {
    let warp_function = DelaunayWarpFunction::new(reference, secondary, dem);

    let [ref_slant_range_dim, ref_azimuth_dim] = reference.data.raster_size();
    let mut resampled_data = Array2::zeros((ref_azimuth_dim, ref_slant_range_dim));

    // The indices in the domain of the reference image
    let indices = (0..ref_slant_range_dim)
        .flat_map(|ref_rg| (0..ref_azimuth_dim).map(move |ref_az| [ref_az, ref_rg]));
    let indices_usize: Vec<[usize; 2]> = indices.clone().collect();

    let kernel = KnabSincKernel::default();
    let deramp = DerampSlcBurst::new();

    let secondary_img = deramp.apply_forward(&secondary);

    warp_function
        .map_many(indices.clone().map(|[az, rg]| [az as f32, rg as f32]))
        .enumerate()
        .filter_map(|(i, coords)| coords.map(|c| (i, c)))
        .for_each(|(i, [sec_az, sec_rg])| {
            let [ref_az, ref_rg] = indices_usize[i];

            let value = interpolate_2d(secondary_img.view(), sec_az, sec_rg, &kernel);
            if let Some(resampled_value) = resampled_data.get_mut([ref_az, ref_rg]) {
                *resampled_value = value;
            }
        });

    resampled_data
}

#[cfg(test)]
mod tests {
    use crate::visualization::cubehelix_colormap;
    use ndarray::s;

    use super::*;

    #[test]
    fn test_resample_secondary_to_reference() {
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

        let resampled_data = resample_secondary_to_reference(&reference, &secondary, &dem);
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
        let ref_array = reference.data.array_data();
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
}
