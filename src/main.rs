use geotiff::{
    GeoTiff,
    raster_data::{RasterData, RasterValue},
};
use nalgebra::Complex;
use ndarray::Array2;
use psi_insar_rs::geocoding::coarse_coregistration_coefficients;
use quick_xml::reader::Reader;
use serde::{Deserialize, Serialize};
use xml_schema_generator::{Options, into_struct};

fn main() {
    // create struct from XML
    // let xml_content = include_str!("metadata/test_data/annotation_example.xml");
    // let mut reader = Reader::from_str(xml_content);

    // if let Ok(root) = into_struct(&mut reader) {
    //     let struct_as_string = root.to_serde_struct(&Options::quick_xml_de());
    //     println!("{}", struct_as_string); // this prints the struct Library and Book as listed above
    // }

    // parse XML into generated struct
    // let library: Library = quick_xml::de::from_str(xml_content).unwrap();
    // assert_eq!(3, library.book.len());

    // Read tiff
    // Load GeoTIFF data
    let measurement_path_1 = "./download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE/measurement/s1a-iw3-slc-vv-20151022t122546-20151022t122549-008265-00ba51-001.tif";
    let reference_image = extract_data(measurement_path_1);
    let measurement_path_2 = "./download_new/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE/measurement/s1a-iw3-slc-vv-20151010t122546-20151010t122550-008090-00b578-001.tiff";
    let secondary_image = extract_data(measurement_path_2);
    let offsets = coarse_coregistration_coefficients(&reference_image, &secondary_image);
    println!("offsets = {:?}", offsets);
}

fn extract_data(measurement_path: &str) -> Array2<Complex<f64>> {
    let geotiff_file =
        std::fs::File::open(&measurement_path).expect("Failed to open measurement TIFF file");
    let data = GeoTiff::read(geotiff_file).expect("Failed to parse TIFF file");
    println!("{:?}", data.raster_width);
    if let RasterData::CInt16(vec) = data.raster_data {
        let mapped_vec: Vec<num_complex::Complex<f64>> = vec
            .into_iter()
            .map(|x| Complex::<f64> {
                re: x.re as f64,
                im: x.im as f64,
            })
            .collect();
        return Array2::from_shape_vec((data.raster_width, data.raster_height), mapped_vec)
            .unwrap();
    }
    panic!();
}
