//! A module for handling Sentinel-1 SLC image data from GeoTIFF files.

use std::path::Path;

use gdal::raster::GdalType;
use geotiff::{GeoTiff, raster_data::RasterData};
use ndarray::Array2;
use num_complex::Complex;

pub struct SlcImage {
    pub array: Array2<Complex<i16>>,
}

impl SlcImage {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let geotiff_data = GeoTiff::read(file).unwrap();
        let (azimuth_rows, slant_range_cols) =
            (geotiff_data.raster_height, geotiff_data.raster_width);
        let array = match geotiff_data.raster_data {
            RasterData::CInt16(ref items) => {
                
                Array2::from_shape_vec((azimuth_rows, slant_range_cols), items.clone())
                    .expect("Unable to build array from values.")
            }
            _ => panic!("Unable to read image"),
        };
        Self { array }
    }

    /// Returns the size of the raster in the slant range and azimuth dimensions, respectively.
    pub fn raster_size(&self) -> [usize; 2] {
        let (azimuth, slant_range) = self.array.dim();
        [slant_range, azimuth]
    }

    /// Returns the value of the pixel at the given azimuth and slant range indices.
    pub fn value(&self, azimuth_idx: usize, slant_range_idx: usize) -> Complex<f32> {
        let value = self.array[(azimuth_idx, slant_range_idx)];
        Complex::new(value.re as f32, value.im as f32)
    }

    /// Returns the Complex<f32> array
    pub fn array_f32(&self) -> Array2<Complex<f32>> {
        self.array
            .map(|v| Complex::<f32>::new(v.re as f32, v.im as f32))
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
