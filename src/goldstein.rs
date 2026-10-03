//! Goldstein Phase Filtering
//!
//! Adaptive filtering of interferometric phase noise, after Goldstein and Werner [\[1\]]: the
//! interferogram is split into overlapping patches, and the spectrum `Z(u, v)` of each patch is
//! multiplied by its own smoothed magnitude raised to a power `α`,
//!
//! ```text
//! H(u, v) = S{|Z(u, v)|}^α · Z(u, v)
//! ```
//!
//! where `S` is a moving-average smoothing. Fringes concentrate their energy in a few spectral
//! peaks, which are amplified relative to the noise spread over the spectrum, so the filter
//! strength adapts to the local fringe pattern: `α = 0` leaves the interferogram unchanged, and
//! larger `α` filters harder. The filtered patches are blended back with triangular weights.
//!
//! The defaults are 32 × 32 patches overlapping by 75% (a step of 8 pixels), a 3 × 3 smoothing
//! window and `α = 0.5`. SNAP's Goldstein Phase Filtering operator [\[2\]] uses the same filter
//! (also on the spectrum magnitude), with 64 × 64 blocks, a 3 × 3 window and `α = 1` by default.
//!
//! # Sources
//!
//! 1. R. M. Goldstein and C. L. Werner, "Radar interferogram filtering for geophysical
//!    applications", Geophysical Research Letters 25(21), 4035-4038, 1998.
//!    <https://doi.org/10.1029/1998GL900033>
//! 2. SNAP microwave toolbox, `GoldsteinFilterOp`:
//!    <https://github.com/senbox-org/microwave-toolbox/blob/master/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/filtering/GoldsteinFilterOp.java>
//!
//! [\[1\]]: https://doi.org/10.1029/1998GL900033
//! [\[2\]]: https://github.com/senbox-org/microwave-toolbox/blob/master/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/filtering/GoldsteinFilterOp.java

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
use num_complex::Complex;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

/// Parameters of the Goldstein phase filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoldsteinFilter {
    /// Filter exponent `α`: 0 leaves the interferogram unchanged, larger values filter harder.
    pub alpha: f32,
    /// Side of the square patches, in pixels.
    pub patch_size: usize,
    /// Distance between consecutive patches, in pixels (at most `patch_size`).
    pub step: usize,
    /// Side of the moving-average window that smooths the spectrum magnitude (odd).
    pub smoothing_window: usize,
}

impl Default for GoldsteinFilter {
    fn default() -> Self {
        GoldsteinFilter {
            alpha: 0.5,
            patch_size: 32,
            step: 8,
            smoothing_window: 3,
        }
    }
}

impl GoldsteinFilter {
    /// The filtered `interferogram` (`[azimuth, range]`).
    ///
    /// Zero pixels are treated as invalid: they do not contribute to the patches' spectra and
    /// stay zero. Only the phase of the result is meaningful, since the filter also scales the
    /// amplitude. Patches extending past the image edges are zero-padded, so every pixel is
    /// filtered.
    ///
    /// # Panics
    ///
    /// If `alpha` is negative, `patch_size` is zero, `step` is not in `1..=patch_size`, or
    /// `smoothing_window` is not odd.
    pub fn apply(&self, interferogram: ArrayView2<'_, Complex<f32>>) -> Array2<Complex<f32>> {
        let n = self.patch_size;
        assert!(
            self.alpha >= 0.0,
            "alpha must not be negative: {}",
            self.alpha
        );
        assert!(n > 0, "The patch size must be positive");
        assert!(
            (1..=n).contains(&self.step),
            "The step ({}) must be between 1 and the patch size ({n})",
            self.step
        );
        assert!(
            self.smoothing_window % 2 == 1,
            "The smoothing window must be odd: {}",
            self.smoothing_window
        );

        let (rows, cols) = interferogram.dim();
        let mut planner = FftPlanner::new();
        let ffts = Ffts {
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        };
        // Triangular weights, highest at the patch center and positive up to its edges.
        let half = n as f32 / 2.0;
        let taper: Vec<f32> = (0..n)
            .map(|k| 1.0 - (k as f32 - half + 0.5).abs() / half)
            .collect();

        let mut sum = Array2::<Complex<f32>>::zeros((rows, cols));
        let mut weights = Array2::<f32>::zeros((rows, cols));
        let column_origins = patch_origins(cols, n, self.step);
        // Patches of one row of patches are filtered in parallel, then accumulated in order.
        for y in patch_origins(rows, n, self.step) {
            let patches: Vec<_> = column_origins
                .par_iter()
                .filter_map(|&x| Some((x, self.filter_patch(interferogram, y, x, &ffts)?)))
                .collect();
            for (x, patch) in patches {
                for i in 0..n.min(rows - y) {
                    for j in 0..n.min(cols - x) {
                        if interferogram[[y + i, x + j]].norm_sqr() == 0.0 {
                            continue;
                        }
                        let weight = taper[i] * taper[j];
                        sum[[y + i, x + j]] += patch[i * n + j] * weight;
                        weights[[y + i, x + j]] += weight;
                    }
                }
            }
        }
        sum.zip_mut_with(&weights, |value, &weight| {
            if weight > 0.0 {
                *value /= weight;
            }
        });
        sum
    }

    /// The filtered patch with its top-left corner at (`y`, `x`), row-major, or `None` if the
    /// patch has no valid pixel.
    fn filter_patch(
        &self,
        interferogram: ArrayView2<'_, Complex<f32>>,
        y: usize,
        x: usize,
        ffts: &Ffts,
    ) -> Option<Vec<Complex<f32>>> {
        let n = self.patch_size;
        let (rows, cols) = interferogram.dim();
        let mut buffer = vec![Complex::new(0.0, 0.0); n * n];
        let mut any_valid = false;
        for i in 0..n.min(rows - y) {
            for j in 0..n.min(cols - x) {
                let value = interferogram[[y + i, x + j]];
                any_valid |= value.norm_sqr() != 0.0;
                buffer[i * n + j] = value;
            }
        }
        if !any_valid {
            return None;
        }

        // 2D FFT: FFT of the rows, transpose, FFT of the rows again. The spectrum is then
        // transposed, which does not matter for the (symmetric) smoothing and weighting.
        ffts.forward.process(&mut buffer);
        transpose(&mut buffer, n);
        ffts.forward.process(&mut buffer);

        let magnitude: Vec<f32> = buffer.iter().map(|z| z.norm()).collect();
        let half_window = (self.smoothing_window / 2) as isize;
        let window_len = (self.smoothing_window * self.smoothing_window) as f32;
        for u in 0..n {
            for v in 0..n {
                // The spectrum is periodic, so the window wraps around.
                let mut smoothed = 0.0;
                for du in -half_window..=half_window {
                    let uu = (u as isize + du).rem_euclid(n as isize) as usize;
                    for dv in -half_window..=half_window {
                        let vv = (v as isize + dv).rem_euclid(n as isize) as usize;
                        smoothed += magnitude[uu * n + vv];
                    }
                }
                buffer[u * n + v] *= (smoothed / window_len).powf(self.alpha);
            }
        }

        ffts.inverse.process(&mut buffer);
        transpose(&mut buffer, n);
        ffts.inverse.process(&mut buffer);
        // rustfft does not normalize the inverse transform.
        let scale = 1.0 / (n * n) as f32;
        buffer.iter_mut().for_each(|z| *z *= scale);
        Some(buffer)
    }
}

struct Ffts {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

/// Patch origins every `step` pixels along a dimension of length `len`, plus a last patch
/// ending at the edge, so that every pixel is covered. A dimension shorter than a patch gets a
/// single, zero-padded patch.
fn patch_origins(len: usize, patch_size: usize, step: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let last = len.saturating_sub(patch_size);
    let mut origins: Vec<usize> = (0..last).step_by(step).collect();
    origins.push(last);
    origins
}

/// Transposes the square `n × n` row-major matrix in `buffer` in place.
fn transpose(buffer: &mut [Complex<f32>], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            buffer.swap(i * n + j, j * n + i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{PI, TAU};

    /// The difference between two phases, wrapped to (-π, π].
    fn phase_error(a: Complex<f32>, b: Complex<f32>) -> f32 {
        (a * b.conj()).arg()
    }

    /// Fringes with a whole number of cycles per 32-pixel patch in each direction.
    fn fringes(rows: usize, cols: usize) -> Array2<Complex<f32>> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            Complex::from_polar(1.0, TAU * (2.0 * i as f32 + 3.0 * j as f32) / 32.0)
        })
    }

    /// Deterministic pseudo-random numbers in [0, 1) (xorshift).
    fn uniform(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    #[test]
    fn patch_origins_cover_every_pixel() {
        assert_eq!(
            patch_origins(100, 32, 8),
            [0, 8, 16, 24, 32, 40, 48, 56, 64, 68]
        );
        assert_eq!(patch_origins(64, 32, 8), [0, 8, 16, 24, 32]);
        assert_eq!(patch_origins(32, 32, 8), [0]);
        assert_eq!(patch_origins(10, 32, 8), [0]);
        assert!(patch_origins(0, 32, 8).is_empty());
    }

    #[test]
    fn zero_alpha_is_identity() {
        let mut next = uniform(1);
        // Not a multiple of the step, and smaller than a patch in one dimension.
        let data = Array2::from_shape_simple_fn((45, 20), || {
            Complex::from_polar(0.5 + next(), TAU * next())
        });
        let filter = GoldsteinFilter {
            alpha: 0.0,
            ..Default::default()
        };
        let filtered = filter.apply(data.view());
        for (a, b) in filtered.iter().zip(&data) {
            assert!((a - b).norm() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn noise_free_fringes_keep_their_phase() {
        let data = fringes(70, 90);
        let filtered = GoldsteinFilter::default().apply(data.view());
        for (a, b) in filtered.iter().zip(&data) {
            assert!(phase_error(*a, *b).abs() < 1e-3);
        }
    }

    #[test]
    fn reduces_phase_noise() {
        let clean = fringes(96, 96);
        let mut next = uniform(2);
        // Uniform phase noise in [-1.2, 1.2] rad (standard deviation 0.69 rad).
        let noisy = clean.mapv(|z| z * Complex::from_polar(1.0, 2.4 * (next() - 0.5)));
        let rms_error = |data: &Array2<Complex<f32>>| {
            let sum: f32 = data
                .iter()
                .zip(&clean)
                .map(|(a, b)| phase_error(*a, *b).powi(2))
                .sum();
            (sum / data.len() as f32).sqrt()
        };
        let filtered_rms_error = |alpha| {
            let filter = GoldsteinFilter {
                alpha,
                ..Default::default()
            };
            rms_error(&filter.apply(noisy.view()))
        };

        // Observed: 0.69 rad before filtering, 0.34 with α = 0.5 and 0.16 with α = 1.
        let before = rms_error(&noisy);
        let (half, one) = (filtered_rms_error(0.5), filtered_rms_error(1.0));
        assert!(
            half < 0.75 * before && one < 0.75 * half && one < 0.5 * before,
            "RMS phase error {before} -> {half} (α = 0.5) -> {one} (α = 1)"
        );
    }

    #[test]
    fn invalid_pixels_stay_zero() {
        let mut data = fringes(40, 40);
        for i in 0..40 {
            for j in 0..10 {
                data[[i, j]] = Complex::new(0.0, 0.0);
            }
        }
        let filtered = GoldsteinFilter::default().apply(data.view());
        for ((i, j), value) in filtered.indexed_iter() {
            if j < 10 {
                assert_eq!(*value, Complex::new(0.0, 0.0));
            } else {
                assert!(value.norm() > 0.0, "({i}, {j}) is zero");
            }
        }

        let zeros = Array2::<Complex<f32>>::zeros((40, 40));
        assert_eq!(GoldsteinFilter::default().apply(zeros.view()), zeros);
    }

    #[test]
    #[should_panic(expected = "step")]
    fn rejects_step_larger_than_patch() {
        let filter = GoldsteinFilter {
            step: 33,
            ..Default::default()
        };
        filter.apply(fringes(40, 40).view());
    }

    #[test]
    fn phase_of_constant_interferogram_is_preserved() {
        let data = Array2::from_elem((50, 50), Complex::from_polar(2.0, PI / 3.0));
        let filtered = GoldsteinFilter::default().apply(data.view());
        for value in &filtered {
            assert!((value.arg() - PI / 3.0).abs() < 1e-4);
        }
    }
}
