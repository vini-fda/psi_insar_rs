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
    use crate::stft::{Stft, WindowFunction};
    use crate::visualization::cubehelix_colormap;
    use ndarray::s;

    use super::*;

    #[test]
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

    #[test]
    fn test_spectrum_visualization() -> Result<(), Box<dyn std::error::Error>> {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let image = reference.data.array_f32();
        let rec =
            rerun::RecordingStreamBuilder::new("fft_spectrum_visualization").connect_grpc()?;

        // Compute FFT along both dimensions
        let spectrum_0 = compute_spectrum(&image, 0);
        let spectrum_1 = compute_spectrum(&image, 1);

        // Compute average magnitude squared along each dimension
        let (rows, cols) = image.dim();

        // For dimension 0 (rows), average across columns
        // this shows the FFT in the Azimuth direction
        for row in 0..rows {
            let row_avg = spectrum_0.row(row).map(|&x| x.norm_sqr()).sum() / cols as f32;

            // Map row index to FFT frequency ordering
            let freq_idx = if row <= rows / 2 { row } else { row - rows };

            rec.set_time_sequence("row", freq_idx as i64);
            rec.log("fft_dim0", &rerun::Scalars::new([row_avg as f64]))?;
        }

        // calculate the DC center of the spectrum
        // find the center of mass of the spectrum distribution
        let mut total_mass = 0.0;
        let mut weighted_sum = 0.0;

        for row in 0..rows {
            let row_avg = spectrum_0.row(row).map(|&x| x.norm_sqr()).sum() / cols as f32;
            total_mass += row_avg;
            weighted_sum += row_avg * row as f32;
        }

        let dc_center_row = (weighted_sum / total_mass) as usize;
        let dc_center = spectrum_0.row(dc_center_row).map(|&x| x.norm_sqr()).sum() / cols as f32;

        rec.log(
            "dc_center",
            &rerun::TextLog::new(format!(
                "DC center row: {dc_center_row}, value: {dc_center}"
            )),
        )?;

        // For dimension 1 (columns), average across rows
        for col in 0..cols {
            let col_avg = spectrum_1.column(col).map(|&x| x.norm_sqr()).sum() / rows as f32;

            // Map column index to FFT frequency ordering
            let freq_idx = if col <= cols / 2 { col } else { col - cols };

            rec.set_time_sequence("col", freq_idx as i64);
            rec.log("fft_dim1", &rerun::Scalars::new([col_avg as f64]))?;
        }

        Ok(())
    }

    #[test]
    fn test_spectrogram() -> Result<(), Box<dyn std::error::Error>> {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let image = reference.data.array_f32();
        let rec = rerun::RecordingStreamBuilder::new("spectrogram_visualization").connect_grpc()?;

        // cut cols in half
        let image = image.slice(s![.., ..image.dim().1 / 2]).to_owned();
        let rows = image.dim().0;

        // Example usage of STFT
        // fn main() {
        //     // Initialize a new STFT object
        //     let n_fft = 1024;
        //     let hop_length = 256;
        //     let stft = Stft::new(n_fft, hop_length, WindowFunction::Hann::<f64>, true);
        //     // Create a 2D array of f64
        //     let data = vec![0.0; 2048];
        //     let input = ArrayView2::from_shape((2, 1024), &data).unwrap();
        //     let expected_output = input.clone();
        //     // Perform the forward STFT
        //     let stft_res = stft.forward(input).unwrap();
        //     // perform the inverse STFT
        //     let istft_res = stft.inverse(stft_res.view()).unwrap();
        //     assert_eq!(expected_output, istft_res);
        // }
        let hop_length = 32;
        let stft = Stft::new(rows, hop_length, WindowFunction::Hann::<f32>, true);
        // Let's apply to a single column
        let column = image.column(1000);
        let stft_res = stft.forward(column).unwrap();
        let (cols, rows) = stft_res.dim();
        // let istft_res = stft.inverse(stft_res.view()).unwrap();
        // assert_eq!(column, istft_res);

        let array = stft_res.map(|x| x.norm());

        // log as image
        let rr_image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, array)?;
        rec.log("spectrogram_visualization", &rr_image)?;
        // log also phase

        let vector = stft_res.as_slice_memory_order().unwrap().to_vec();
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
            [cols as u32, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rec.log("phase_visualization", &rr_image)?;
        Ok(())
    }

    #[test]
    fn test_phase_visualization() -> Result<(), Box<dyn std::error::Error>> {
        let reference = Sentinel1SlcBurst::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();

        let array = reference.data.array_f32();
        // cut cols in half
        let array = array.slice(s![.., ..array.dim().1 / 2]).to_owned();
        let (rows, cols) = array.dim();
        let vector = array.as_slice_memory_order().unwrap().to_vec();
        // map the vactor to phase, then map that to RGB color
        let rgb_vector: Vec<u8> = vector
            .iter()
            .flat_map(|&x| {
                let phase = x.arg();
                let normalized_phase =
                    (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
            })
            .collect();
        let rec = rerun::RecordingStreamBuilder::new("image_phase_visualization").connect_grpc()?;

        let rr_image = rerun::Image::from_color_model_and_bytes(
            rgb_vector,
            [cols as u32, rows as u32],
            rerun::ColorModel::RGB,
            rerun::ChannelDatatype::U8,
        );
        rec.log("image_phase_visualization", &rr_image)?;
        Ok(())
    }
}
