use crate::{
    metadata::annotation_xml::{Polynomial, Velocity},
    sentinel::Sentinel1SlcBurst,
};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use std::f64::consts::PI;
use std::mem::size_of;

#[derive(Debug, Clone, Copy)]
pub enum DerampingMode {
    /// Only perform deramping (centers spectra at Doppler centroid frequency)
    Standard,
    /// Perform both deramping and demodulation (centers spectra at 0Hz)
    FullDemodulation,
}

impl Default for DerampingMode {
    fn default() -> Self {
        Self::FullDemodulation
    }
}

pub struct DerampSlcBurst {
    mode: DerampingMode,
}

/// - Forward, or *Deramping*, is the removal of the linear frequency modulation introduced by antenna steering
/// - Backward, or *Reramping*, is the inverse of deramping, i.e. it adds back the linear frequency modulation introduced by antenna steering
///
/// By default, we want to deramp the SLC data, then reramp it back to the original data.
pub enum Direction {
    Forward,
    Backward,
}

impl Default for Direction {
    fn default() -> Self {
        Self::Forward
    }
}

struct RelevantParameters {
    /// Azimuth steering rate
    k_psi: f64,
    /// Doppler centroid frequency polynomial
    f_eta_c: Polynomial,
    /// Azimuth FM rate polynomial
    k_a: Polynomial,
    /// Radar frequency
    f_c: f64,
    /// Satellite velocity vector
    v_s: Velocity,
    /// Number of lines per burst
    nl_burst: usize,
    /// Azimuth time interval
    delta_t_s: f64,
    /// Number of samples in swath
    ns_swath: usize,
    /// Range sampling interval
    delta_tau_s: f64,
    /// Slant range time
    tau_0: f64,
}

impl RelevantParameters {
    /// Extracts the relevant parameters for the deramping from the Sentinel1SlcBurst metadata
    pub fn new(slc: &Sentinel1SlcBurst) -> Self {
        // k_psi: Azimuth steering rate
        let k_psi = slc
            .metadata
            .general_annotation
            .product_information
            .azimuth_steering_rate;

        // f_eta_c: Doppler centroid frequency polynomial
        let f_eta_c = &slc.metadata.doppler_centroid.dc_estimate_list.dc_estimate[0]
            .data_dc_polynomial
            .coefficients;

        // k_a: Azimuth FM rate polynomial
        let k_a = &slc
            .metadata
            .general_annotation
            .azimuth_fm_rate_list
            .azimuth_fm_rate[0]
            .azimuth_fm_rate_polynomial
            .coefficients;

        // f_c: Radar frequency
        let f_c = slc
            .metadata
            .general_annotation
            .product_information
            .radar_frequency;

        // V_S: Satellite velocity vector [x,y,z]
        let v_s = slc.metadata.general_annotation.orbit_list.orbit[0].velocity;

        // Nl_burst: Number of lines per burst
        let nl_burst = slc.metadata.swath_timing.lines_per_burst;

        // Δt_s: Azimuth time interval
        let delta_t_s = slc
            .metadata
            .image_annotation
            .image_information
            .azimuth_time_interval;

        // NS_swath: Number of samples in swath
        let ns_swath = slc
            .metadata
            .image_annotation
            .image_information
            .number_of_samples;

        // Δτ_s: Range sampling interval (inverse of range sampling rate)
        let delta_tau_s = 1.0
            / slc
                .metadata
                .general_annotation
                .product_information
                .range_sampling_rate;

        // τ(0): Slant range time
        let tau_0 = slc
            .metadata
            .image_annotation
            .image_information
            .slant_range_time;

        Self {
            k_psi,
            f_eta_c: f_eta_c.clone(),
            k_a: k_a.clone(),
            f_c,
            v_s,
            nl_burst,
            delta_t_s,
            ns_swath,
            delta_tau_s,
            tau_0,
        }
    }
}

impl DerampSlcBurst {
    pub fn new() -> Self {
        Self {
            mode: DerampingMode::default(),
        }
    }

    pub fn set_mode(mut self, mode: DerampingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn apply_forward(&self, slc: &Sentinel1SlcBurst) -> Array2<Complex<f32>> {
        self.apply(slc, Direction::Forward)
    }

    pub fn apply_backward(&self, slc: &Sentinel1SlcBurst) -> Array2<Complex<f32>> {
        self.apply(slc, Direction::Backward)
    }

    fn apply(&self, slc: &Sentinel1SlcBurst, direction: Direction) -> Array2<Complex<f32>> {
        let mode = self.mode;

        let RelevantParameters {
            k_psi,
            f_eta_c,
            k_a,
            f_c,
            v_s,
            nl_burst,
            delta_t_s,
            ns_swath,
            delta_tau_s,
            tau_0,
        } = RelevantParameters::new(slc);
        // Calculate k_s (Doppler rate introduced by antenna steering)
        // k_s = (2 * v_s * f_c * k_psi) / c
        let c = 299792458.0; // speed of light in m/s
        let v_s_magnitude = (v_s.x * v_s.x + v_s.y * v_s.y + v_s.z * v_s.z).sqrt();
        let k_s = (2.0 * v_s_magnitude * f_c * k_psi) / c;

        // Helper function to calculate k_a at a given range time tau
        let k_a_at_tau = |tau: f64| -> f64 {
            let tau_diff = tau - tau_0;
            k_a.coefficients[0]
                + k_a.coefficients[1] * tau_diff
                + k_a.coefficients[2] * tau_diff * tau_diff
        };

        // Helper function to calculate f_eta_c at a given range time tau
        let f_eta_c_at_tau = |tau: f64| -> f64 {
            let tau_diff = tau - tau_0;
            f_eta_c.coefficients[0]
                + f_eta_c.coefficients[1] * tau_diff
                + f_eta_c.coefficients[2] * tau_diff * tau_diff
        };

        // Helper function to calculate k_t at a given range time tau
        // k_t = (k_a * k_s) / (k_a - k_s)
        let k_t_at_tau = |tau: f64| -> f64 {
            let k_a_val = k_a_at_tau(tau);
            (k_a_val * k_s) / (k_a_val - k_s)
        };

        // Helper function to calculate eta_ref at a given range time tau
        // eta_ref = -f_eta_c / k_a
        let eta_ref_at_tau = |tau: f64| -> f64 {
            let f_eta_c_val = f_eta_c_at_tau(tau);
            let k_a_val = k_a_at_tau(tau);
            -f_eta_c_val / k_a_val
        };

        // The deramping phase function phi(eta, tau)
        // For deramping only: phi = -π * k_t(τ) * (η - η_ref(τ))²
        // For deramping + demodulation: phi = -π * k_t(τ) * (η - η_ref(τ))² - 2π * f_eta_c(τ) * (η - η_ref(τ))
        let phi = |eta: f64, tau: f64| -> f64 {
            let k_t = k_t_at_tau(tau);
            let eta_ref = eta_ref_at_tau(tau);
            let eta_diff = eta - eta_ref;
            let phase = -PI * k_t * eta_diff * eta_diff;

            let phase = match mode {
                DerampingMode::Standard => phase,
                DerampingMode::FullDemodulation => {
                    let f_eta_c_val = f_eta_c_at_tau(tau);
                    phase - 2.0 * PI * f_eta_c_val * eta_diff
                }
            };

            match direction {
                Direction::Forward => phase,
                Direction::Backward => -phase,
            }
        };

        // Calculate eta vector (azimuth times centered in middle of burst)
        // eta = [-Nl_burst/2 * Δt_s, Nl_burst/2 * Δt_s]
        let eta: Array1<f64> = Array1::from_iter(
            (-(nl_burst as i32 / 2)..(nl_burst as i32 / 2)).map(|i| i as f64 * delta_t_s),
        );

        // Calculate tau vector (range times for each sample)
        // tau(i) = tau(0) + i * Δτ_s
        let tau: Array1<f64> =
            Array1::from_iter((0..ns_swath).map(|i| tau_0 + i as f64 * delta_tau_s));

        let buffer = slc.data.read_buffer();
        let mut deramped = Array2::<Complex<f32>>::zeros((nl_burst, ns_swath));
        for (i, &eta_val) in eta.iter().enumerate() {
            for (j, &tau_val) in tau.iter().enumerate() {
                // Read the complex value
                let x = Complex::<i16>::from(buffer[(i, j)]);
                let x = Complex::new(x.re as f64, x.im as f64);

                // Calculate and apply phase
                let phase = phi(eta_val, tau_val);
                let phase_cos = phase.cos();
                let phase_sin = phase.sin();
                let x = x * Complex::new(phase_cos, phase_sin);

                // Convert back to Complex<i16> and write
                let x = Complex::<f32>::new(x.re as f32, x.im as f32);
                deramped[[i, j]] = x;
            }
        }

        deramped
    }
}

#[cfg(test)]
mod tests {
    use ndarray::s;

    use crate::visualization::cubehelix_colormap;

    use super::*;

    // #[test]
    // fn test_deramp() {
    //     let mut slc = Sentinel1SlcBurst::load_from_directory(
    //         "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
    //         "S1A_IW_SLC__1SVV_20151022T122546_20151022T122549_008265_00BA51_422D",
    //     )
    //     .unwrap();

    //     // visualize the amplitude and phase of the original data
    //     let original_data = slc.data.array_data();
    //     // cut cols in half
    //     let original_data = original_data
    //         .slice(s![.., ..original_data.dim().1 / 2])
    //         .to_owned();
    //     let (cols, rows) = original_data.dim();
    //     let normalized = original_data.map(|&x| x.norm());
    //     let max_norm = normalized
    //         .iter()
    //         .copied()
    //         .max_by(|a, b| a.partial_cmp(b).unwrap())
    //         .unwrap();
    //     let normalized = normalized.map(|&x| (x / max_norm).powf(0.3));
    //     let rr_image =
    //         rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, normalized).unwrap();
    //     let rec = rerun::RecordingStreamBuilder::new("image_phase_visualization")
    //         .connect_grpc()
    //         .unwrap();

    //     rec.log("amplitude_visualization_original", &rr_image)
    //         .unwrap();

    //     // log also phase
    //     let vector = original_data.as_slice_memory_order().unwrap().to_vec();
    //     let rgb_vector: Vec<u8> = vector
    //         .iter()
    //         .flat_map(|&x| {
    //             let phase = x.arg();
    //             let normalized_phase =
    //                 (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    //             cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
    //         })
    //         .collect();
    //     let rr_image = rerun::Image::from_color_model_and_bytes(
    //         rgb_vector,
    //         [rows as u32, cols as u32],
    //         rerun::ColorModel::RGB,
    //         rerun::ChannelDatatype::U8,
    //     );
    //     rec.log("phase_visualization_original", &rr_image).unwrap();

    //     // visualize the amplitude and phase of the deramped data
    //     let deramp = DerampSlcBurst::new().set_mode(DerampingMode::Standard);

    //     let deramped = deramp.apply_forward(&mut slc);
    //     // cut cols in half
    //     let deramped = deramped.slice(s![.., ..deramped.dim().1 / 2]).to_owned();
    //     let (cols, rows) = deramped.dim();
    //     let normalized = deramped.map(|&x| x.norm());
    //     let max_norm = normalized
    //         .iter()
    //         .copied()
    //         .max_by(|a, b| a.partial_cmp(b).unwrap())
    //         .unwrap();
    //     let normalized = normalized.map(|&x| (x / max_norm).powf(0.3));
    //     let rr_image =
    //         rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, normalized).unwrap();
    //     // let rec = rerun::RecordingStreamBuilder::new("image_phase_visualization")
    //     //     .connect_grpc()
    //     //     .unwrap();

    //     rec.log("amplitude_visualization", &rr_image).unwrap();

    //     // log also phase

    //     let vector = deramped.as_slice_memory_order().unwrap().to_vec();
    //     let rgb_vector: Vec<u8> = vector
    //         .iter()
    //         .flat_map(|&x| {
    //             let phase = x.arg();
    //             let normalized_phase =
    //                 (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    //             cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
    //         })
    //         .collect();
    //     let rr_image = rerun::Image::from_color_model_and_bytes(
    //         rgb_vector,
    //         [rows as u32, cols as u32],
    //         rerun::ColorModel::RGB,
    //         rerun::ChannelDatatype::U8,
    //     );
    //     rec.log("phase_visualization", &rr_image).unwrap();
    // }
}
