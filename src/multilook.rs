//! Multilooking
//!
//! Averages blocks of `azimuth × range` pixels into one, which reduces noise (each output pixel
//! averages several independent "looks") and gives pixels of about the same size on the ground
//! in both directions. Sentinel-1 IW SLC pixels are about 2.3 m in slant range (4 m or more in
//! ground range) by 14 m in azimuth, so several range looks are needed per azimuth look.
//!
//! Interferograms must be multilooked in the complex domain: averaging `s1 · s2*` weights each
//! pixel's phase by its amplitude and keeps the phase continuous, whereas averaging wrapped
//! phases near ±π gives meaningless values.

use std::ops::{AddAssign, Div};

use ndarray::{Array2, ArrayView2, Axis, s};
use num_traits::Zero;
use rayon::prelude::*;

use crate::metadata::annotation_xml::SlcProductAnnotation;

/// Number of looks (pixels averaged) in each direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Looks {
    pub azimuth: usize,
    pub range: usize,
}

impl Looks {
    /// # Panics
    ///
    /// If either number of looks is zero.
    pub fn new(azimuth: usize, range: usize) -> Self {
        assert!(
            azimuth > 0 && range > 0,
            "The number of looks must be positive, got {azimuth} × {range}"
        );
        Looks { azimuth, range }
    }

    /// `range` looks, with the number of azimuth looks that makes the output pixels as close
    /// to square on the ground as possible (at least 1), like SNAP's "GR Square Pixel" option.
    ///
    /// The ground range spacing is the slant range spacing divided by the sine of the
    /// incidence angle at mid swath. For example, 6 range looks give 2 azimuth looks (pixels of
    /// about 25 × 28 m) in IW1 and IW2, as in NASA's Sentinel-1 interferogram recipe, but only
    /// 1 azimuth look in IW3, whose larger incidence angle gives finer ground range pixels.
    pub fn square_ground_pixel(range: usize, metadata: &SlcProductAnnotation) -> Self {
        let info = &metadata.image_annotation.image_information;
        let incidence_angle: f64 = info
            .incidence_angle_mid_swath
            .trim()
            .parse()
            .unwrap_or_else(|err| {
                panic!(
                    "Invalid incidenceAngleMidSwath {:?}: {err}",
                    info.incidence_angle_mid_swath
                )
            });
        let ground_range_spacing = info.range_pixel_spacing / incidence_angle.to_radians().sin();
        let azimuth =
            (range as f64 * ground_range_spacing / info.azimuth_pixel_spacing).round() as usize;
        Looks::new(azimuth.max(1), range)
    }

    /// The (fractional) `[azimuth, range]` coordinates in the multilooked image of the
    /// full-resolution coordinates `[azimuth, range]`. Output pixel `i` averages input pixels
    /// `i * looks..(i + 1) * looks`, so its center is at input coordinate
    /// `i * looks + (looks - 1) / 2`.
    pub fn multilooked_coords(&self, [azimuth, range]: [f64; 2]) -> [f64; 2] {
        let to_multilooked = |coord: f64, looks: usize| {
            let looks = looks as f64;
            (coord - (looks - 1.0) / 2.0) / looks
        };
        [
            to_multilooked(azimuth, self.azimuth),
            to_multilooked(range, self.range),
        ]
    }
}

/// The mean of each block of `looks.azimuth × looks.range` pixels of `data` (`[azimuth,
/// range]`). Incomplete blocks at the bottom and right edges are dropped, so the output has
/// `rows / looks.azimuth` rows and `cols / looks.range` columns.
///
/// Zero (invalid) pixels are averaged like any other, which lowers the amplitude of blocks at
/// the edge of the valid area but not their phase.
pub fn multilook<A>(data: ArrayView2<'_, A>, looks: Looks) -> Array2<A>
where
    A: Copy + Zero + AddAssign + Div<f32, Output = A> + Send + Sync,
{
    let (rows, cols) = data.dim();
    let (out_rows, out_cols) = (rows / looks.azimuth, cols / looks.range);
    let count = (looks.azimuth * looks.range) as f32;
    let mut output = Array2::zeros((out_rows, out_cols));
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut out_row)| {
            let block_rows = data.slice(s![i * looks.azimuth..(i + 1) * looks.azimuth, ..]);
            for row in block_rows.outer_iter() {
                for (j, out) in out_row.iter_mut().enumerate() {
                    for &value in row.slice(s![j * looks.range..(j + 1) * looks.range]) {
                        *out += value;
                    }
                }
            }
            out_row.mapv_inplace(|sum| sum / count);
        });
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex;

    #[test]
    fn averages_blocks_and_drops_incomplete_ones() {
        // 5 × 7 with 2 × 3 looks: 2 × 2 output, the last row and column are dropped.
        let data = Array2::from_shape_fn((5, 7), |(i, j)| (10 * i + j) as f32);
        let output = multilook(data.view(), Looks::new(2, 3));
        // Block (0, 0): rows 0-1, cols 0-2 -> mean of 0,1,2,10,11,12 = 6.
        let expected = ndarray::array![[6.0, 9.0], [26.0, 29.0]];
        assert_eq!(output, expected);
    }

    #[test]
    fn averages_complex_values_not_phases() {
        // Phases of ±(π - 0.1) average to π in the complex domain, but to 0 as real numbers.
        let phase = std::f32::consts::PI - 0.1;
        let data = ndarray::array![[
            Complex::from_polar(1.0, phase),
            Complex::from_polar(1.0, -phase)
        ]];
        let output = multilook(data.view(), Looks::new(1, 2));
        assert_relative_eq!(
            output[[0, 0]].arg().abs(),
            std::f32::consts::PI,
            epsilon = 1e-6
        );
        assert_relative_eq!(output[[0, 0]].norm(), phase.cos().abs(), epsilon = 1e-6);
    }

    #[test]
    fn multilooked_coords_of_block_centers() {
        let looks = Looks::new(2, 3);
        // Block (0, 0) is centered between input pixels 0-1 and on input pixel 1.
        assert_eq!(looks.multilooked_coords([0.5, 1.0]), [0.0, 0.0]);
        assert_eq!(looks.multilooked_coords([2.5, 4.0]), [1.0, 1.0]);
        assert_eq!(looks.multilooked_coords([0.0, 0.0]), [-0.25, -1.0 / 3.0]);
    }

    #[test]
    fn single_look_is_identity() {
        let data = Array2::from_shape_fn((3, 4), |(i, j)| (i * j) as f32);
        assert_eq!(multilook(data.view(), Looks::new(1, 1)), data);
    }

    #[test]
    fn square_ground_pixel_looks() {
        let metadata: SlcProductAnnotation =
            quick_xml::de::from_str(include_str!("metadata/test_data/annotation_example.xml"))
                .unwrap();
        // IW3: 2.33 m / sin(44.1°) = 3.35 m ground range spacing, 13.97 m azimuth spacing.
        assert_eq!(Looks::square_ground_pixel(6, &metadata), Looks::new(1, 6));
        assert_eq!(Looks::square_ground_pixel(8, &metadata), Looks::new(2, 8));
        assert_eq!(Looks::square_ground_pixel(1, &metadata), Looks::new(1, 1));
    }

    #[test]
    #[should_panic(expected = "must be positive")]
    fn zero_looks_panic() {
        Looks::new(0, 4);
    }
}
