use ndarray::{Array2, s};
use num_complex::Complex32;
use std::ops::Range;

pub struct CoregistrationResult {
    pub offsets: (i32, i32),
    pub correlation: Array2<Complex32>,
    // The range of indices in the Reference Image, identified as the matching target to the Kernel
    pub ref_image_range: [Range<usize>; 2],
    // The range of indices in the Secondary Image, used as the Kernel
    pub sec_image_range: [Range<usize>; 2],
}

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
    /// - `k`: Half-side of the kernel. Must be ≥ 1.
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
    /// cross-correlation (NCC) between a fixed kernel of the `reference_image`
    /// and a sliding window in the `secondary_image`.
    ///
    /// The objective is to solve:
    ///
    /// ```text
    /// argmax_{(i,j)} Re ⟨R, S_{i,j}⟩
    /// ```
    ///
    /// where:
    /// - `R` is a fixed patch from the reference image
    /// - `S_{i,j}` is a `KERNEL_SIZE × KERNEL_SIZE` patch of the secondary image centered at offset `(i, j)`
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
    /// (\Delta x, \Delta y) = \arg\max_{(i, j) \in \mathcal{W}} \left| \sum_{(u,v) \in \mathcal{K}} \overline{S[u + i, v + j]} \cdot R[u, v] \right|^2
    /// ```
    ///
    /// where:
    /// - `R` is the patch extracted from the reference image,
    /// - `S` is the secondary image,
    /// - `𝒦` is the kernel domain (a square of side `2k + 1`),
    /// - `𝒲` is the search window domain in the secondary image.
    ///
    /// # Panics
    ///
    /// Panics if the image dimensions are smaller than the required kernel or search window sizes.
    pub fn estimate_offset(
        &self,
        reference_image: &Array2<Complex32>,
        secondary_image: &Array2<Complex32>,
    ) -> CoregistrationResult {
        let CoarseCoregistration {
            k,
            kernel_size,
            search_size,
            ..
        } = *self;

        let (rows1, cols1) = reference_image.dim();
        let (rows2, cols2) = secondary_image.dim();
        // Preliminary checks
        assert!(rows1 >= search_size + 2 * k, "Secondary image too small.");
        assert!(cols1 >= search_size + 2 * k, "Secondary image too small.");
        assert!(rows2 >= kernel_size, "Reference image too small.");
        assert!(cols2 >= kernel_size, "Reference image too small.");

        // Search window offsets (in secondary image):
        let offset_rows2 = (rows2 - search_size) / 2;
        let offset_cols2 = (cols2 - search_size) / 2;
        let mut correlation = Array2::<Complex32>::zeros((search_size, search_size));
        // Patch offsets (in reference image):
        let offset_rows1 = (rows1 - kernel_size) / 2;
        let offset_cols1 = (cols1 - kernel_size) / 2;
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

        // Now we can obtain the offsets of the patch on the secondary image
        // which maximizes the correlation of the secondary image with the reference image patch
        let max_patch_offset_rows = max_x + offset_rows2 - k;
        let max_patch_offset_cols = max_y + offset_cols2 - k;

        let delta_x = (max_patch_offset_rows - offset_rows1) as i32;
        let delta_y = (max_patch_offset_cols - offset_cols1) as i32;
        let offsets = (delta_x, delta_y);
        let ref_image_range = [
            offset_rows1..offset_rows1 + kernel_size,
            offset_cols1..offset_cols1 + kernel_size,
        ];
        let sec_image_range = [
            max_patch_offset_rows..max_patch_offset_rows + kernel_size,
            max_patch_offset_cols..max_patch_offset_cols + kernel_size,
        ];
        CoregistrationResult {
            offsets,
            correlation,
            ref_image_range,
            sec_image_range,
        }
    }

    /// Computes the cross-correlation surface between the reference and secondary images using a naïve nested loop.
    ///
    /// # Arguments
    ///
    /// * `out` - A mutable 2D array (must be of shape `[search_size, search_size]`) where the correlation result will be stored.
    /// * `reference_image` - The primary complex-valued 2D image, with the kernel extracted from its center.
    /// * `secondary_image` - The secondary image to compare against.
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
        out: &mut Array2<Complex32>,
        reference_image: &Array2<Complex32>,
        secondary_image: &Array2<Complex32>,
        offset_rows1: usize,
        offset_cols1: usize,
        offset_rows2: usize,
        offset_cols2: usize,
    ) {
        let CoarseCoregistration {
            k,
            kernel_size,
            search_size,
            ..
        } = *self;
        for x in 0..search_size {
            for y in 0..search_size {
                let mut acc = Complex32::new(0.0, 0.0);
                for dx in 0..kernel_size {
                    for dy in 0..kernel_size {
                        let ref_val = reference_image[[dx + offset_rows1, dy + offset_cols1]];
                        let sec_val =
                            secondary_image[[x + dx + offset_rows2 - k, y + dy + offset_cols2 - k]];
                        acc += sec_val.conj() * ref_val;
                    }
                }
                out[[x, y]] = acc;
            }
        }
    }
}
