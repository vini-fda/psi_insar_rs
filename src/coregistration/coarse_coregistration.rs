use geotiff::{GeoTiff, raster_data::RasterData};
use ndarray::{Array2, s};
use num_complex::{Complex, Complex32};
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
        assert!(rows1 >= search_size + 2 * k, "Reference image too small.");
        assert!(cols1 >= search_size + 2 * k, "Reference image too small.");
        assert!(rows2 >= kernel_size, "Secondary image too small.");
        assert!(cols2 >= kernel_size, "Secondary image too small.");

        // Patch offsets (in secondary image):
        let offset_rows2 = (rows2 - kernel_size) / 2;
        let offset_cols2 = (cols2 - kernel_size) / 2;
        let mut correlation = Array2::<Complex32>::zeros((search_size, search_size));
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
        let offsets = (delta_x, delta_y);
        let ref_image_range = [
            offset_rows1 - k + max_x..offset_rows1 - k + max_x + kernel_size,
            offset_cols1 - k + max_y..offset_cols1 - k + max_y + kernel_size,
        ];
        let sec_image_range = [
            offset_rows2..offset_rows2 + kernel_size,
            offset_cols2..offset_cols2 + kernel_size,
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
                let mut acc = Complex32::new(0.0, 0.0);

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

pub fn extract_data(measurement_path: &str) -> Array2<Complex<f32>> {
    let geotiff_file =
        std::fs::File::open(&measurement_path).expect("Failed to open measurement TIFF file");
    let data = GeoTiff::read(geotiff_file).expect("Failed to parse TIFF file");

    if let RasterData::CInt16(vec) = data.raster_data {
        let mapped_vec: Vec<num_complex::Complex<f32>> = vec
            .into_iter()
            .map(|x| Complex::<f32>::new(x.re as f32, x.im as f32))
            .collect();
        return Array2::from_shape_vec((data.raster_height, data.raster_width), mapped_vec)
            .unwrap();
    }
    panic!();
}

#[cfg(test)]
mod manual_tests {
    use ndarray::{Array2, s};

    use crate::dem::DEM;

    use super::{CoarseCoregistration, CoregistrationResult, extract_data};

    fn normalize(data: &mut Array2<f32>) {
        let max_amplitude = data.iter().fold(0.0, |acc: f32, &x| acc.max(x));
        data.map_mut(|x| *x = *x / max_amplitude);
    }

    /// A visual test using the Rerun framework.
    ///
    /// This test is intended to be ignored by CI/CD, as it only works
    /// if you have a local Rerun instance running.
    #[test]
    #[ignore]
    fn visual_test_rerun() {
        let measurement_path_1 = "./download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE/measurement/s1a-iw3-slc-vv-20151022t122546-20151022t122549-008265-00ba51-001.tif";
        let reference_image = extract_data(measurement_path_1);
        let measurement_path_2 = "./download_new/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE/measurement/s1a-iw3-slc-vv-20151010t122546-20151010t122550-008090-00b578-001.tiff";
        let secondary_image = extract_data(measurement_path_2);
        let rec = rerun::RecordingStreamBuilder::new("visual_test_coregistration")
            .connect_tcp()
            .expect("Could not connect to local Rerun instance.");
        let coregistration = CoarseCoregistration::new(255, 64).unwrap();
        let CoregistrationResult {
            offsets,
            correlation,
            ref_image_range,
            sec_image_range,
        } = coregistration.estimate_offset(&reference_image, &secondary_image);
        // Log correlation tensor
        let data = correlation.mapv_into_any(|c| c.norm());
        let tensor = rerun::Tensor::try_from(data)
            .unwrap()
            .with_dim_names(["rows", "cols"]);
        rec.log("correlation", &tensor)
            .expect("Could not finish recording");

        // Log 2 images for comparison
        let mut ref_patch = reference_image
            .slice(s![ref_image_range[0].clone(), ref_image_range[1].clone()])
            .map(|c| c.norm().powf(0.3))
            .to_owned();
        normalize(&mut ref_patch);
        let img_ref =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, ref_patch).unwrap();
        rec.log("ref", &img_ref)
            .expect("Could not finish recording");
        let mut kernel = secondary_image
            .slice(s![sec_image_range[0].clone(), sec_image_range[1].clone()])
            .map(|c| c.norm().powf(0.3))
            .to_owned();
        normalize(&mut kernel);
        let img_sec =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, kernel).unwrap();
        rec.log("sec", &img_sec)
            .expect("Could not finish recording");
        rec.log(
            "logs",
            &rerun::TextLog::new(format!("offsets = {offsets:?}"))
                .with_level(rerun::TextLogLevel::INFO),
        )
        .unwrap();
        let lat_lon = [
            [19.49831428810679, -98.59301000370277],
            [19.50516497145322, -98.63155270190656],
            [19.51197728410019, -98.66992859316593],
            [19.51875199752162, -98.7081413973442],
            [19.52548985740382, -98.74619471009619],
            [19.53219158481003, -98.78409200847771],
            [19.53885787727854, -98.82183665623535],
            [19.5454894098591, -98.85943190879907],
            [19.55208683609183, -98.89688091799755],
            [19.55865078893238, -98.9341867365148],
            [19.56518188162694, -98.97135232210496],
            [19.57168070854037, -99.00838054158147],
            [19.57814784594043, -99.04527417459472],
            [19.58458385274104, -99.08203591721202],
            [19.59098927120689, -99.11866838531225],
            [19.5973646276222, -99.15517411780644],
            [19.60371043292545, -99.19155557969563],
            [19.61002718331234, -99.22781516497541],
            [19.61631536080904, -99.26395519939655],
            [19.62257559081166, -99.29997885004042],
            [19.62877652653297, -99.33570497416603],
            [19.33245909128312, -98.62592837407114],
            [19.33931662789667, -98.66442995280119],
            [19.34613584071955, -98.70276488660213],
            [19.35291750085852, -98.74093689141644],
            [19.35966235364623, -98.77894955911454],
            [19.36637111980492, -98.81680636309756],
            [19.37304449654394, -98.85451066358128],
            [19.3796831585957, -98.89206571258312],
            [19.38628775919396, -98.92947465863222],
            [19.39285893099849, -98.96674055122072],
            [19.39939728696963, -99.0038663450142],
            [19.40590342119594, -99.0408549038362],
            [19.41237790967802, -99.07770900444162],
            [19.41882131107135, -99.1144313400927],
            [19.4252341673906, -99.15102452394993],
            [19.43161700467787, -99.187491092289],
            [19.43797033363727, -99.22383350755531],
            [19.44429465023745, -99.2600541612652],
            [19.45059043628451, -99.29615537676354],
            [19.45685815996673, -99.33213941184621],
            [19.46306674963845, -99.36782713184502],
        ];
        rec.log(
            "geo_points",
            &rerun::GeoPoints::from_lat_lon(lat_lon.iter()),
        )
        .unwrap();
        let dem = DEM::open_file("dem.tif");

        let dem_extent = dem.data.model_extent();
        let lines = dem_extent.to_lines();
        let dem_array = dem.read_raster_data();
        // let tensor = rerun::Tensor::try_from(dem)
        //     .unwrap()
        //     .with_dim_names(["rows", "cols"]);
        let tensor =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, dem_array).unwrap();
        rec.log("DEM", &tensor).expect("Could not finish recording");
        let data_img = rerun::Image::from_color_model_and_tensor(
            rerun::ColorModel::L,
            reference_image
                .map(|c| c.norm().powf(0.3) / 22.0)
                .to_owned(),
        )
        .unwrap();
        rec.log("ref_img", &data_img)
            .expect("Could not finish recording");
        rec.log(
            "DEM Extent",
            &rerun::GeoLineStrings::from_lat_lon([lines
                .map(|line| [[line.start.y, line.start.x], [line.end.y, line.end.x]])
                .as_flattened()])
            .with_radii([rerun::Radius::new_ui_points(2.0)])
            .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
        )
        .unwrap();

        // let row = 300;
        // let col = 261;
        // rec.log(
        //     "logs",
        //     &rerun::TextLog::new(format!(
        //         "lon, lat at (300, 261) = {:?}\n\
        //         height = {}\n\
        //         xyz = {:?}",
        //         dem.get_lon_lat_at_index(row, col),
        //         dem.get_value_at_index(row, col),
        //         dem.get_ecef_at_pixel(row, col)
        //     ))
        //     .with_level(rerun::TextLogLevel::INFO),
        // )
        // .unwrap();
    }
}
