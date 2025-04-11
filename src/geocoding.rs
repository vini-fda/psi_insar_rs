// r = |P - S(a)|
// v(a) *  (P - S(a)) = 0

use ndarray::{Array2, ArrayView2, s};
use num_complex::Complex64;

const PATCH_WIDTH: usize = 64;
const PATCH_HEIGHT: usize = 64;

/// Calculates the coarse coregistration coefficients,
/// based on the cross-correlation of the reference image
/// and a fixed patch of the secondary image (centered at the image's center).
pub fn coarse_coregistration_coefficients(
    reference_image: &Array2<Complex64>,
    secondary_image: &Array2<Complex64>,
) -> (usize, usize) {
    // Preliminary checks
    // Removed, because the images not necessary have the same dimensions.
    //
    // Example:
    //  left: (23739, 1507)
    //  right: (23695, 1506)
    // debug_assert_eq!(reference_image.dim(), secondary_image.dim());
    let (width, height) = secondary_image.dim();
    debug_assert!(width >= PATCH_WIDTH);
    debug_assert!(height >= PATCH_HEIGHT);

    // Patch offsets: width and height
    let ow = (width - PATCH_WIDTH) / 2;
    let oh = (height - PATCH_HEIGHT) / 2;
    let patch = secondary_image.slice(s![ow..(PATCH_WIDTH + ow), oh..(oh + PATCH_HEIGHT)]);
    let mut correlation =
        Array2::<Complex64>::zeros((width - PATCH_WIDTH + 1, height - PATCH_HEIGHT + 1));
    let (correlation_width, correlation_height) = correlation.dim();
    for dx in 0..correlation_width {
        for dy in 0..correlation_height {
            for x in 0..PATCH_WIDTH {
                for y in 0..PATCH_HEIGHT {
                    correlation[[dx, dy]] +=
                        reference_image[[x + dx, y + dy]].conj() * patch[[x, y]];
                }
            }
        }
    }
    let mut max_dx = 0;
    let mut max_dy = 0;
    let mut current_max = 0.0;
    for dx in 0..correlation_width {
        for dy in 0..correlation_height {
            let val = correlation[[dx, dy]].norm_sqr();
            if val > current_max {
                current_max = val;
                max_dx = dx;
                max_dy = dy;
            }
        }
    }
    (max_dx, max_dy)
}
