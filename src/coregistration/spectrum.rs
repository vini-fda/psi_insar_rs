use ndarray::{Array1, Array2, Axis};
use num_complex::Complex;
use rustfft::{FftPlanner, num_traits::Zero};

/// Helper function that computes FFT along dimension 1 (which is always contiguous)
fn compute_spectrum_dim1(image: &Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let mut result = image.to_owned();
    let (rows, cols) = result.dim();
    let fft_size = cols;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut buffer = vec![Complex::zero(); fft_size];

    // Apply FFT along dimension 1
    for idx in 0..rows {
        let slice = result.index_axis(Axis(0), idx);
        buffer.copy_from_slice(slice.as_slice_memory_order().unwrap());
        fft.process(&mut buffer);
        result
            .index_axis_mut(Axis(0), idx)
            .assign(&Array1::from_vec(buffer.clone()));
    }

    result
}

/// Performs an in-place FFT shift along the specified dimension
/// This reorders the data so that the zero frequency is in the middle
pub fn fftshift_inplace(array: &mut Array2<Complex<f32>>, dim: usize) {
    let (rows, cols) = array.dim();
    let size = if dim == 0 { rows } else { cols };
    let half = size / 2;

    // Create a temporary buffer to store the shifted data
    let mut temp = vec![Complex::zero(); size];

    if dim == 0 {
        // Shift along rows
        for col in 0..cols {
            // Copy the second half to the first half of temp
            for i in 0..half {
                temp[i] = array[[i + half, col]];
            }
            // Copy the first half to the second half of temp
            for i in 0..half {
                temp[i + half] = array[[i, col]];
            }
            // Copy back to the array
            for i in 0..size {
                array[[i, col]] = temp[i];
            }
        }
    } else {
        // Shift along columns
        for row in 0..rows {
            // Copy the second half to the first half of temp
            for i in 0..half {
                temp[i] = array[[row, i + half]];
            }
            // Copy the first half to the second half of temp
            for i in 0..half {
                temp[i + half] = array[[row, i]];
            }
            // Copy back to the array
            for i in 0..size {
                array[[row, i]] = temp[i];
            }
        }
    }
}

/// Computes the FFT in one dimension of the image.
pub fn compute_spectrum(image: &Array2<Complex<f32>>, dim: usize) -> Array2<Complex<f32>> {
    match dim {
        0 => {
            // For dimension 0, we need to actually reorder the data
            let (rows, cols) = image.dim();
            let mut transposed = Array2::zeros((cols, rows));
            for i in 0..rows {
                for j in 0..cols {
                    transposed[[j, i]] = image[[i, j]];
                }
            }
            let result = compute_spectrum_dim1(&transposed);

            // Transpose back
            let mut final_result = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    final_result[[i, j]] = result[[j, i]];
                }
            }
            final_result
        }
        1 => compute_spectrum_dim1(image),
        _ => panic!("Invalid dimension: must be 0 or 1"),
    }
}

#[cfg(test)]
mod tests {
    use crate::sentinel::Sentinel1SlcBurst;

    use super::*;

    #[test]
    #[ignore = "Needs to open external files"]
    fn test_spectrum() {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let image = reference.data.array_f32();

        // Test FFT along both dimensions
        let spectrum_0 = compute_spectrum(&image, 0);
        let spectrum_1 = compute_spectrum(&image, 1);

        // Basic validation
        assert_eq!(spectrum_0.dim(), image.dim());
        assert_eq!(spectrum_1.dim(), image.dim());
    }
}
