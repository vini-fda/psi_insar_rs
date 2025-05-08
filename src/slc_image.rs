//! A module for handling Sentinel-1 SLC image data from GeoTIFF files, using GDAL.

use std::path::Path;

use gdal::{
    Dataset,
    raster::{Buffer, GdalType},
};
use ndarray::Array2;
use num_complex::Complex;

pub struct SlcImage {
    pub data: Dataset,
}

impl SlcImage {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let data = Dataset::open(path).unwrap();
        Self { data }
    }

    /// Returns the size of the raster in the slant range and azimuth dimensions, respectively.
    pub fn raster_size(&self) -> [usize; 2] {
        let (slant_range, azimuth) = self.data.raster_size();
        [slant_range, azimuth]
    }

    pub fn raw_buffer(&self) -> Buffer<ComplexI16> {
        let band = self.data.rasterband(1).expect("Could not read band");
        let buf: Buffer<ComplexI16> = band.read_band_as().unwrap();
        buf
    }

    /// Returns the value of the pixel at the given azimuth and slant range indices.
    pub fn value(&self, azimuth_idx: usize, slant_range_idx: usize) -> Option<Complex<f32>> {
        let band = self.data.rasterband(1).expect("Could not read band");
        let buffer = match band.read_as::<ComplexI16>(
            (slant_range_idx as isize, azimuth_idx as isize),
            (1, 1),
            (1, 1),
            None,
        ) {
            Ok(buffer) => buffer,
            Err(_) => return None,
        };

        let ComplexI16(value) = buffer.data()[0];
        Some(Complex::new(value.re as f32, value.im as f32))
    }

    /// Returns the data as an array of complex numbers.
    pub fn array_data(&self) -> Array2<Complex<f32>> {
        let [slant_range, azimuth] = self.raster_size();
        let band = self.data.rasterband(1).expect("Could not read band");
        let complex_data: Vec<Complex<f32>> = band
            .read_as::<ComplexI16>((0, 0), (slant_range, azimuth), (slant_range, azimuth), None)
            .expect("Could not read data")
            .data()
            .iter()
            .map(|&value| {
                let ComplexI16(value) = value;
                let re = value.re;
                let im = value.im;
                Complex::new(re as f32, im as f32)
            })
            .collect();
        Array2::from_shape_vec((azimuth, slant_range), complex_data)
            .expect("Could not create array from complex data")
    }
}

#[derive(Copy, Clone)]
pub struct ComplexI16(Complex<i16>);

impl From<ComplexI16> for Complex<i16> {
    fn from(value: ComplexI16) -> Self {
        value.0
    }
}

impl From<Complex<i16>> for ComplexI16 {
    fn from(value: Complex<i16>) -> Self {
        ComplexI16(value)
    }
}

impl GdalType for ComplexI16 {
    fn gdal_ordinal() -> gdal_sys::GDALDataType::Type {
        gdal_sys::GDALDataType::GDT_CInt16
    }
}
