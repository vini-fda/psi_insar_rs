use crate::{
    constants::SENTINEL_1_WAVELENGTH,
    dem::DEM,
    perp_baseline::EnhancedDelaunayWarpFunction,
    sentinel::{Sentinel1SlcIWBurst, Sentinel1SlcIWSwath},
};
use nalgebra::Vector3;
use ndarray::{Array2, ArrayView2, Axis, Zip, s};
use ndarray_npy::WriteNpyExt;
use num_complex::Complex;
use num_traits::Zero;
use rayon::prelude::*;
use std::ops::AddAssign;

pub fn coregister_and_remove_flat_phase(
    reference: &Sentinel1SlcIWBurst,
    secondaries: &[Sentinel1SlcIWBurst],
    dem: &DEM,
) {
    for secondary in secondaries {
        log::info!("Computing warp function");
        let start_time = std::time::Instant::now();
        // The same orbits are used for the warp function and for the geometric phase
        let osh_1 = reference.continuous_orbital_state_history();
        let osh_2 = secondary.continuous_orbital_state_history();
        let warp_function =
            EnhancedDelaunayWarpFunction::with_orbits(reference, &osh_1, &osh_2, dem);
        let end_time = std::time::Instant::now();
        log::info!("Time taken: {:?}", end_time - start_time);

        let [ref_slant_range_dim, ref_azimuth_dim] = reference.burst_data.raster_size();
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

        // let s = reference.metadata
        //     .image_annotation
        //     .image_information
        //     .range_pixel_spacing;
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
                            let sec_rg =
                                nn.interpolate(|v| v.data().secondary_coords[1], ref_coords.into());

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

                            // Pixel coordinates include the R/c bistatic shift; undo it
                            let zero_doppler_1 =
                                osh_1.pixel_to_zero_doppler_time(ref_az as f64, ref_rg as f64);
                            let zero_doppler_2 = osh_2.pixel_to_zero_doppler_time(sec_az, sec_rg);
                            let s_1 = osh_1.interp_pos(zero_doppler_1);
                            let s_2 = osh_2.interp_pos(zero_doppler_2);

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

/// Compute the bounding box of a stack of Sentinel1SlcIWSwath images,
/// with a small margin to account for the fact that the images are not
/// exactly aligned.
pub fn bounding_box_from_stack<'a, I: IntoIterator<Item = &'a Sentinel1SlcIWSwath>>(
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

/// Compute the bounding box of a stack of Sentinel1SlcIWSwath images,
/// with a small margin to account for the fact that the images are not
/// exactly aligned.
pub fn bounding_box_from_burst_stack<'a, I: IntoIterator<Item = &'a Sentinel1SlcIWBurst>>(
    stack: I,
) -> [f64; 4] {
    const OFFSET_LAT: f64 = 0.05;
    const OFFSET_LON: f64 = 0.05;
    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;
    for burst in stack {
        let burst_index = burst.burst_index;
        let gcps_len = burst
            .metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .count as usize;
        let num_bursts = burst.metadata.swath_timing.burst_list.count as usize;
        // Rows along azimuth, columns along range
        let gcp_rows = num_bursts + 1;
        assert!(gcps_len.is_multiple_of(gcp_rows));
        let gcp_columns = gcps_len / gcp_rows;
        let start = burst_index * gcp_columns;
        let end = (burst_index + 2) * gcp_columns;

        for point in &burst
            .metadata
            .geolocation_grid
            .geolocation_grid_point_list
            .geolocation_grid_point[start..end]
        {
            min_lat = min_lat.min(point.latitude);
            max_lat = max_lat.max(point.latitude);
            min_lon = min_lon.min(point.longitude);
            max_lon = max_lon.max(point.longitude);
        }
    }
    assert_ne!(min_lat, f64::MAX);
    assert_ne!(max_lat, f64::MIN);
    assert_ne!(min_lon, f64::MAX);
    assert_ne!(max_lon, f64::MIN);
    [
        min_lat - OFFSET_LAT,
        max_lat + OFFSET_LAT,
        min_lon - OFFSET_LON,
        max_lon + OFFSET_LON,
    ]
}

/// Size of the window over which [`coherence`] is estimated, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherenceWindow {
    pub azimuth: usize,
    pub range: usize,
}

impl Default for CoherenceWindow {
    /// 3 azimuth × 10 range pixels, which is roughly square on the ground for Sentinel-1 IW
    /// SLCs (about 14 m azimuth by 3-4 m ground range pixels).
    fn default() -> Self {
        CoherenceWindow {
            azimuth: 3,
            range: 10,
        }
    }
}

/// The interferometric coherence of each pixel,
///
/// ```text
/// γ = |Σ s1 · s2* · e^(-iφ)| / √(Σ |s1|² · Σ |s2|²)
/// ```
///
/// with sums over the `window` centered on the pixel (truncated at the image edges), where
/// `s1` is the `reference` SLC and `s2` the `secondary` SLC coregistered to it. `interferogram`
/// is `s1 · s2* · e^(-iφ)`, i.e. the interferogram after removing any phase `φ` that is
/// expected to vary within the window: with the flat-earth and topographic phase removed, the
/// fringes they cause do not lower the coherence.
///
/// The result is in `[0, 1]`, and 0 where either SLC is zero over the whole window. With few
/// pixels per window the estimate is biased upwards: uncorrelated SLCs give about
/// `1 / √(window pixels)` rather than 0.
///
/// # Panics
///
/// If the three arrays do not have the same shape, or the window is empty.
pub fn coherence(
    interferogram: ArrayView2<'_, Complex<f32>>,
    reference: ArrayView2<'_, Complex<f32>>,
    secondary: ArrayView2<'_, Complex<f32>>,
    window: CoherenceWindow,
) -> Array2<f32> {
    assert_eq!(
        interferogram.dim(),
        reference.dim(),
        "Interferogram and reference shapes differ"
    );
    assert_eq!(
        interferogram.dim(),
        secondary.dim(),
        "Interferogram and secondary shapes differ"
    );
    assert!(
        window.azimuth > 0 && window.range > 0,
        "Empty coherence window: {window:?}"
    );
    let numerator = box_sum(interferogram, window);
    let reference_power = box_sum(reference.mapv(|v| v.norm_sqr()).view(), window);
    let secondary_power = box_sum(secondary.mapv(|v| v.norm_sqr()).view(), window);
    Zip::from(&numerator)
        .and(&reference_power)
        .and(&secondary_power)
        .par_map_collect(|numerator, &p1, &p2| {
            let denominator = (p1 as f64 * p2 as f64).sqrt();
            if denominator > 0.0 {
                (numerator.norm() as f64 / denominator).min(1.0) as f32
            } else {
                0.0
            }
        })
}

/// The sum of `data` over the `window` centered on each pixel, truncated at the edges. For an
/// even window size, the window extends one pixel further after the center than before it.
fn box_sum<A>(data: ArrayView2<'_, A>, window: CoherenceWindow) -> Array2<A>
where
    A: Copy + Zero + AddAssign + Send + Sync,
{
    let (rows, cols) = data.dim();
    let window_range = |center: usize, size: usize, len: usize| {
        center.saturating_sub((size - 1) / 2)..(center + size / 2 + 1).min(len)
    };

    // Sums along range, then along azimuth.
    let mut range_sums = Array2::zeros((rows, cols));
    range_sums
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(data.axis_iter(Axis(0)))
        .for_each(|(mut sums, row)| {
            for (j, sum) in sums.iter_mut().enumerate() {
                for &value in row.slice(s![window_range(j, window.range, cols)]) {
                    *sum += value;
                }
            }
        });
    let mut sums = Array2::zeros((rows, cols));
    sums.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut sums)| {
            for row in range_sums
                .slice(s![window_range(i, window.azimuth, rows), ..])
                .outer_iter()
            {
                sums.zip_mut_with(&row, |sum, &value| *sum += value);
            }
        });
    sums
}

/// Coherence below which NASA's Sentinel-1 interferogram recipe considers the phase unreliable:
/// "typically data with coherence values less than 0.3 are thrown out".
///
/// Source: <https://www.earthdata.nasa.gov/learn/data-recipes/create-interferogram-using-esas-sentinel-1-toolbox>
pub const COHERENCE_THRESHOLD: f32 = 0.3;

/// A coherence mask: `true` (valid) where `coherence` is at least `threshold` (e.g.
/// [`COHERENCE_THRESHOLD`]), `false` elsewhere.
///
/// Pixels whose whole coherence window has no data have zero coherence (see [`coherence`]), so
/// they are invalid for any positive threshold. But at the edges of the data, the window of a
/// pixel without data can reach valid neighbors, so combine it with [`valid_data_mask`] to
/// exclude all the pixels without data.
pub fn coherence_mask(coherence: ArrayView2<'_, f32>, threshold: f32) -> Array2<bool> {
    coherence.mapv(|gamma| gamma >= threshold)
}

/// A no-data mask: `true` (valid) where `data` is non-zero, `false` where it is zero, which is
/// how the processing steps of this crate mark pixels without data (e.g. outside the valid
/// samples of a burst, or outside the image when geocoding).
pub fn valid_data_mask<T: Zero + Clone>(data: ArrayView2<'_, T>) -> Array2<bool> {
    data.map(|value| !value.is_zero())
}

/// `data` with the pixels outside `mask` (where it is `false`) set to zero, i.e. to no data.
///
/// # Panics
///
/// If `data` and `mask` do not have the same shape.
pub fn apply_mask<T: Zero + Clone>(mut data: Array2<T>, mask: ArrayView2<'_, bool>) -> Array2<T> {
    assert_eq!(data.dim(), mask.dim(), "Data and mask shapes differ");
    data.zip_mut_with(&mask, |value, &valid| {
        if !valid {
            *value = T::zero();
        }
    });
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Deterministic pseudo-random complex values with unit-ish amplitude (xorshift).
    fn speckle(rows: usize, cols: usize, seed: u64) -> Array2<Complex<f32>> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        Array2::from_shape_simple_fn((rows, cols), || {
            Complex::from_polar(0.5 + next(), std::f32::consts::TAU * next())
        })
    }

    fn interferogram(
        s1: &Array2<Complex<f32>>,
        s2: &Array2<Complex<f32>>,
        phase: impl Fn(usize, usize) -> f32,
    ) -> Array2<Complex<f32>> {
        Array2::from_shape_fn(s1.dim(), |(i, j)| {
            s1[[i, j]] * s2[[i, j]].conj() * Complex::from_polar(1.0, -phase(i, j))
        })
    }

    #[test]
    fn box_sum_truncates_at_edges() {
        let data = Array2::<f32>::ones((4, 5));
        let window = CoherenceWindow {
            azimuth: 3,
            range: 4,
        };
        // Range window [j - 1, j + 2]: 3, 4, 4, 3, 2 pixels; azimuth window [i - 1, i + 1]:
        // 2, 3, 3, 2 pixels.
        let sums = box_sum(data.view(), window);
        assert_eq!(sums.row(0).to_vec(), [6.0, 8.0, 8.0, 6.0, 4.0]);
        assert_eq!(sums.row(1).to_vec(), [9.0, 12.0, 12.0, 9.0, 6.0]);
        assert_eq!(sums.row(3), sums.row(0));
    }

    #[test]
    fn identical_images_are_fully_coherent() {
        let s1 = speckle(20, 30, 1);
        // Same speckle with a constant phase offset and a different gain.
        let s2 = s1.mapv(|v| v * Complex::from_polar(2.0, 1.0));
        let gamma = coherence(
            interferogram(&s1, &s2, |_, _| 0.0).view(),
            s1.view(),
            s2.view(),
            CoherenceWindow::default(),
        );
        gamma
            .iter()
            .for_each(|&g| assert_relative_eq!(g, 1.0, epsilon = 1e-5));
    }

    #[test]
    fn removing_the_fringes_restores_coherence() {
        let s1 = speckle(20, 60, 2);
        // A fast range phase ramp, like flat-earth fringes: a full cycle every 8 samples.
        let ramp = |_: usize, j: usize| std::f32::consts::TAU * j as f32 / 8.0;
        let s2 = Array2::from_shape_fn(s1.dim(), |(i, j)| {
            s1[[i, j]] * Complex::from_polar(1.0, -ramp(i, j))
        });
        let window = CoherenceWindow::default();
        let with_fringes = coherence(
            interferogram(&s1, &s2, |_, _| 0.0).view(),
            s1.view(),
            s2.view(),
            window,
        );
        let flattened = coherence(
            interferogram(&s1, &s2, ramp).view(),
            s1.view(),
            s2.view(),
            window,
        );
        assert!(with_fringes.mean().unwrap() < 0.5);
        flattened
            .iter()
            .for_each(|&g| assert_relative_eq!(g, 1.0, epsilon = 1e-5));
    }

    #[test]
    fn independent_images_have_low_coherence() {
        let (s1, s2) = (speckle(60, 200, 3), speckle(60, 200, 4));
        let window = CoherenceWindow::default();
        let gamma = coherence(
            interferogram(&s1, &s2, |_, _| 0.0).view(),
            s1.view(),
            s2.view(),
            window,
        );
        // Bias of the estimator for 30-pixel windows: about 1 / sqrt(30) = 0.18.
        let mean = gamma.mean().unwrap();
        assert!(mean > 0.05 && mean < 0.3, "mean coherence {mean}");
    }

    #[test]
    fn zero_images_have_zero_coherence() {
        let zeros = Array2::<Complex<f32>>::zeros((5, 5));
        let gamma = coherence(
            zeros.view(),
            zeros.view(),
            zeros.view(),
            CoherenceWindow::default(),
        );
        assert!(gamma.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn masks() {
        let coherence = ndarray::array![[0.0f32, 0.29], [0.3, 0.9]];
        assert_eq!(
            coherence_mask(coherence.view(), COHERENCE_THRESHOLD),
            ndarray::array![[false, false], [true, true]]
        );

        let data = ndarray::array![[Complex::new(0.0f32, 0.0), Complex::new(0.0, 1.0)]];
        let mask = valid_data_mask(data.view());
        assert_eq!(mask, ndarray::array![[false, true]]);

        let masked = apply_mask(
            coherence,
            ndarray::array![[true, false], [false, true]].view(),
        );
        assert_eq!(masked, ndarray::array![[0.0, 0.0], [0.0, 0.9]]);
    }
}
