use ndarray::{Array2, s};
use num_complex::Complex64;

#[derive(Clone, Copy)]
pub struct CoarseCoregistration {
    /// Half-size of the square kernel patch (i.e., `K`).
    ///
    /// The full kernel will have size `2 * k + 1`.
    k: usize,
    kernel_size: usize,
    /// Size of the square search window in the reference image.
    search_size: usize,
}

impl CoarseCoregistration {
    /// Create a new instance of `CoregistrationParams`, enforcing valid values.
    ///
    /// # Arguments
    /// - `k`: Half-side of the kernel. Must be > 0.
    /// - `search_size`: Size of the search window. Must be ≥ 1.
    ///
    /// # Returns
    /// Returns `Some(params)` if values are valid, `None` otherwise.
    pub fn new(k: usize, search_size: usize) -> Option<Self> {
        if k == 0 || search_size == 0 {
            None
        } else {
            let kernel_size = 2 * k + 1;
            Some(Self {
                k,
                kernel_size,
                search_size,
            })
        }
    }

    /// Estimate integer-valued coarse offset `(Δx, Δy)` that maximizes the normalized
    /// cross-correlation (NCC) between a fixed kernel of the `secondary_image`
    /// and a sliding window in the `reference_image`.
    ///
    /// The objective is to solve:
    ///
    /// ```text
    /// argmax_{(i,j)} Re ⟨R_{i,j}, S⟩
    /// ```
    ///
    /// where:
    /// - `R_{i,j}` is a `KERNEL_SIZE × KERNEL_SIZE` patch of the reference image centered at offset `(i, j)`
    /// - `S` is a fixed patch from the secondary image
    /// - `⟨·,·⟩` denotes the Hermitian inner product on ℂⁿ
    ///
    /// # Mathematical Notes
    ///
    /// This is a form of **template matching** using **discrete cross-correlation**:
    ///
    /// - The algorithm assumes the phase and amplitude information is important (complex domain).
    /// - The method is robust for small displacements but limited to a ±(SEARCH_SIZE/2) range.
    /// - In the limit of high SNR and continuous signals, this approximates the location of the peak of the cross-ambiguity function.
    ///
    /// # Arguments
    ///
    /// - `reference_image`: A 2D array of complex pixels (e.g., SAR SLC).
    /// - `secondary_image`: A second 2D array to be registered to the reference.
    ///
    /// # Returns
    ///
    /// - `(Δx, Δy)`: Estimated row and column offset aligning the `secondary_image` to the `reference_image`.
    ///
    /// Effectively, one should expect `reference_image[[x, y]] ~= secondary_image[[x + Δx, y + Δy]]`
    /// or, more rigorously, the displacement `(Δx, Δy)` maximizes the spatial cross-correlation:
    ///
    /// ```math
    /// (\Delta x, \Delta y) = \arg\max_{(i, j) \in \mathcal{W}} \left| \sum_{(u,v) \in \mathcal{K}} \overline{R[u + i, v + j]} \cdot S[u, v] \right|^2
    /// ```
    ///
    /// where:
    /// - `R` is the reference image,
    /// - `S` is the patch extracted from the secondary image,
    /// - `𝒦` is the kernel domain (a square of side `2k + 1`),
    /// - `𝒲` is the search window domain in the reference image.
    ///
    /// # Panics
    ///
    /// Panics if the image dimensions are smaller than the required kernel or search window sizes.
    pub fn estimate_offset(
        &self,
        reference_image: &Array2<Complex64>,
        secondary_image: &Array2<Complex64>,
    ) -> (i32, i32) {
        let CoarseCoregistration {
            k,
            kernel_size,
            search_size,
        } = *self;

        let (rows1, cols1) = reference_image.dim();
        let (rows2, cols2) = secondary_image.dim();
        // Preliminary checks
        assert!(rows1 >= search_size + 2 * k, "Reference image too small.");
        assert!(cols1 >= search_size + 2 * k, "Reference image too small.");
        assert!(rows2 >= kernel_size, "Secondary image too small.");
        assert!(cols2 >= kernel_size, "Secondary image too small.");

        // Patch offsets (in secondary image):
        let offset_rows2 = (rows2 - kernel_size) / 2;
        let offset_cols2 = (cols2 - kernel_size) / 2;
        let mut correlation = Array2::<Complex64>::zeros((search_size, search_size));
        // Search window offsets (in reference image):
        let offset_rows1 = (rows1 - search_size) / 2;
        let offset_cols1 = (cols1 - search_size) / 2;
        self.compute_correlation_naive(
            &mut correlation,
            reference_image,
            secondary_image,
            offset_rows1,
            offset_cols1,
            offset_rows2,
            offset_cols2,
        );
        let mut max_x = 0;
        let mut max_y = 0;
        let mut current_max = 0.0;
        for x in 0..search_size {
            for y in 0..search_size {
                let val = correlation[[x, y]].norm_sqr();
                if val > current_max {
                    current_max = val;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        let delta_x = (offset_rows2 as i32) - (max_x + offset_rows1 - k) as i32;
        let delta_y = (offset_cols2 as i32) - (max_y + offset_cols1 - k) as i32;
        (delta_x, delta_y)
    }

    /// Computes the cross-correlation surface between the reference and secondary images using a naïve nested loop.
    ///
    /// # Arguments
    ///
    /// * `out` - A mutable 2D array (must be of shape `[search_size, search_size]`) where the correlation result will be stored.
    /// * `reference_image` - The primary complex-valued 2D image.
    /// * `secondary_image` - The secondary image to compare against, with the kernel extracted from its center.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of the output or input arrays are inconsistent with the configuration in `self`.
    ///
    /// # Performance
    ///
    /// This implementation is not optimized for speed. It uses four nested loops and performs manual indexing into the images
    /// for every patch comparison. Prefer using `Zip` or blocking if performance is critical.
    #[inline(always)]
    pub fn compute_correlation_naive(
        &self,
        out: &mut Array2<Complex64>,
        reference_image: &Array2<Complex64>,
        secondary_image: &Array2<Complex64>,
        offset_rows1: usize,
        offset_cols1: usize,
        offset_rows2: usize,
        offset_cols2: usize,
    ) {
        let CoarseCoregistration {
            k,
            kernel_size,
            search_size,
        } = *self;
        for x in 0..search_size {
            for y in 0..search_size {
                let mut acc = Complex64::new(0.0, 0.0);
                for dx in 0..kernel_size {
                    for dy in 0..kernel_size {
                        let ref_val =
                            reference_image[[x + dx + offset_rows1 - k, y + dy + offset_cols1 - k]];
                        let sec_val = secondary_image[[dx + offset_rows2, dy + offset_cols2]];
                        acc += ref_val.conj() * sec_val;
                    }
                }
                out[[x, y]] = acc;
            }
        }
    }

    /// Computes the coarse cross-correlation between a reference and secondary image using
    /// fast memory-aware access patterns.
    ///
    /// # Assumptions
    /// - The images are large enough to extract the required patches based on the configured `k`.
    /// - The output array must be pre-allocated with shape `(search_size, search_size)`.
    /// - This implementation assumes that the user is **not memory-bound**, and the kernel and
    ///   search windows are **small relative to the full image size**.
    ///
    /// # Performance
    /// This method is optimized for CPU cache locality and vectorization. It extracts the kernel
    /// and search patches as contiguous array slices, significantly improving access speed by
    /// avoiding scattered indexing. The compiler can more effectively apply loop unrolling and
    /// SIMD operations in the inner loop.
    ///
    /// # Arguments
    /// - `out`: Mutable 2D array to hold the correlation result. Must match `(search_size, search_size)`.
    /// - `reference_image`: The full reference image containing the target patch in the center.
    /// - `secondary_image`: The full secondary image containing the kernel patch in the center.
    ///
    /// # Panics
    /// Panics if `out.dim()` does not match `(search_size, search_size)`.
    #[inline(always)]
    pub fn compute_correlation_optimized(
        &self,
        out: &mut Array2<Complex64>,
        reference_image: &Array2<Complex64>,
        secondary_image: &Array2<Complex64>,
        offset_rows1: usize,
        offset_cols1: usize,
        offset_rows2: usize,
        offset_cols2: usize,
    ) {
        let CoarseCoregistration {
            k,
            kernel_size,
            search_size,
        } = *self;

        // Compute the center kernel slice from secondary_image
        let kernel = secondary_image
            .slice(s![
                offset_rows2..offset_rows2 + kernel_size,
                offset_cols2..offset_cols2 + kernel_size
            ])
            .to_owned();

        // Precompute the full region needed from reference_image
        let ref_patch = reference_image
            .slice(s![
                offset_rows1 - k..offset_rows1 - k + search_size + kernel_size,
                offset_cols1 - k..offset_cols1 - k + search_size + kernel_size
            ])
            .to_owned();

        for x in 0..search_size {
            for y in 0..search_size {
                let mut acc = Complex64::new(0.0, 0.0);

                // Element-wise conj multiplication and accumulation
                for dx in 0..kernel_size {
                    for dy in 0..kernel_size {
                        let ref_val = ref_patch[[x + dx, y + dy]].conj();
                        let sec_val = kernel[[dx, dy]];
                        acc += ref_val * sec_val;
                    }
                }

                out[[x, y]] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    /// Returns a zero-filled image with a single complex impulse at the center.
    fn make_impulse_image(dim: usize) -> Array2<Complex64> {
        let mut image = Array2::<Complex64>::zeros((dim, dim));
        let center = dim / 2;
        image[[center, center]] = Complex64::new(1.0, 0.0);
        image
    }

    #[test]
    fn test_zero_offset_impulse_alignment() {
        let image_dim = 127;
        let reference = make_impulse_image(image_dim);
        let secondary = reference.clone();

        let correlator = CoarseCoregistration::new(5, 5).expect("valid parameters");
        let (dx, dy) = correlator.estimate_offset(&reference, &secondary);

        // Identical images should yield zero offset
        assert_eq!(
            (dx, dy),
            (0, 0),
            "Expected zero offset when reference and secondary are identical"
        );
    }

    #[test]
    fn test_known_integer_offset() {
        let image_dim = 127;
        let offset_rows = 2;
        let offset_cols = 1;

        let reference = make_impulse_image(image_dim);
        let mut secondary = Array2::<Complex64>::zeros((image_dim, image_dim));

        let center = image_dim / 2;
        let shifted_row = center + offset_rows;
        let shifted_col = center + offset_cols;
        assert!(
            shifted_row < image_dim && shifted_col < image_dim,
            "Offset exceeds image bounds"
        );

        secondary[[shifted_row, shifted_col]] = Complex64::new(1.0, 0.0);

        let correlator = CoarseCoregistration::new(5, 5).expect("valid parameters");
        let (dx, dy) = correlator.estimate_offset(&reference, &secondary);

        // Since the reference is unshifted and the secondary is forward shifted,
        // the displacement of the secondary relative to reference is +Δ
        assert_eq!(
            (dx, dy),
            (offset_rows as i32, offset_cols as i32),
            "Expected offset ({offset_rows}, {offset_cols}) but got ({dx}, {dy})"
        );
    }
}
