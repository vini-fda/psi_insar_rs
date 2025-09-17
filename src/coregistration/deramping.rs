use crate::{
    metadata::annotation_xml::{Polynomial, Velocity},
    sentinel::Sentinel1SlcIWSwath,
};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use std::f64::consts::PI;

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
    /// Azimuth steering rate (radians/s)
    k_psi: f64,
    /// Doppler centroid frequency polynomial
    f_eta_c: Polynomial,
    /// t0 for f_eta_c polynomial
    f_eta_c_t0: f64,
    /// Azimuth FM rate polynomial
    k_a: Polynomial,
    /// t0 for k_a polynomial
    k_a_t0: f64,
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
    /// Extracts the relevant parameters for the deramping from the Sentinel1SlcIWSwath metadata
    // TODO: check if there's a better solution than burst_index
    pub fn new(slc: &Sentinel1SlcIWSwath, burst_index: usize) -> Self {
        let burst_count = slc.metadata.swath_timing.burst_list.bursts.len();
        assert!(burst_index < burst_count);
        // k_psi: Azimuth steering rate (radians/s)
        let k_psi = slc
            .metadata
            .general_annotation
            .product_information
            .azimuth_steering_rate
            .to_radians();

        // Nl_burst: Number of lines per burst
        let nl_burst_usize = slc.metadata.swath_timing.lines_per_burst;
        let nl_burst_f64 = nl_burst_usize as f64;

        // Δt_s: Azimuth time interval
        let delta_t_s_val = slc
            .metadata
            .image_annotation
            .image_information
            .azimuth_time_interval;

        // Current burst metadata for reference times
        let current_burst_metadata = &slc.metadata.swath_timing.burst_list.bursts[burst_index];
        let burst_start_anx_time = current_burst_metadata.azimuth_anx_time;
        let ref_burst_utc_time = current_burst_metadata.azimuth_time; // DateTime<Utc>
        let ref_burst_anx_time = current_burst_metadata.azimuth_anx_time; // f64

        let eta_mid_anx_time = burst_start_anx_time + (nl_burst_f64 / 2.0) * delta_t_s_val;

        // f_eta_c: Doppler centroid frequency polynomial
        // Select the polynomial whose azimuth time is closest to eta_mid_anx_time.
        let dc_estimate_list = &slc.metadata.doppler_centroid.dc_estimate_list.dc_estimate;

        if dc_estimate_list.is_empty() {
            panic!("Doppler centroid estimate list is empty. Cannot select f_eta_c polynomial.");
        }

        let selected_dc_estimate = dc_estimate_list
            .iter()
            .min_by(|a, b| {
                // Convert DcEstimate's azimuth_time (DateTime<Utc>) to an f64 ANX-equivalent time
                let duration_a = a.azimuth_time.signed_duration_since(ref_burst_utc_time);
                let item_a_anx_equivalent = ref_burst_anx_time +
                    duration_a.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                let duration_b = b.azimuth_time.signed_duration_since(ref_burst_utc_time);
                let item_b_anx_equivalent = ref_burst_anx_time +
                    duration_b.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;

                let diff_a = (item_a_anx_equivalent - eta_mid_anx_time).abs();
                let diff_b = (item_b_anx_equivalent - eta_mid_anx_time).abs();
                diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("dc_estimate_list was checked not to be empty but min_by found no minimum. This indicates a data issue or NaN times.");

        let f_eta_c_poly_container = selected_dc_estimate.data_dc_polynomial.clone();
        let f_eta_c_t0_val = selected_dc_estimate.t0;

        // k_a: Azimuth FM rate polynomial
        // Select the polynomial whose azimuth time is closest to eta_mid_anx_time.
        let azimuth_fm_rate_list = &slc
            .metadata
            .general_annotation
            .azimuth_fm_rate_list
            .azimuth_fm_rate;

        if azimuth_fm_rate_list.is_empty() {
            panic!("Azimuth FM rate list is empty. Cannot select k_a polynomial.");
        }

        let selected_azimuth_fm_rate_item = azimuth_fm_rate_list
            .iter()
            .min_by(|a, b| {
                // Convert AzimuthFmRate's azimuth_time (DateTime<Utc>) to an f64 ANX-equivalent time
                let duration_a = a.azimuth_time.signed_duration_since(ref_burst_utc_time);
                let item_a_anx_equivalent = ref_burst_anx_time +
                    duration_a.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                let duration_b = b.azimuth_time.signed_duration_since(ref_burst_utc_time);
                let item_b_anx_equivalent = ref_burst_anx_time +
                    duration_b.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;

                let diff_a = (item_a_anx_equivalent - eta_mid_anx_time).abs();
                let diff_b = (item_b_anx_equivalent - eta_mid_anx_time).abs();
                diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("azimuth_fm_rate_list was checked not to be empty but min_by found no minimum. This indicates a data issue or NaN times.");

        let k_a_poly = selected_azimuth_fm_rate_item
            .azimuth_fm_rate_polynomial
            .coefficients
            .clone();
        let k_a_t0_val = selected_azimuth_fm_rate_item.t0;

        // f_c: Radar frequency
        let f_c = slc
            .metadata
            .general_annotation
            .product_information
            .radar_frequency;

        // V_S: Satellite velocity vector [x,y,z]
        let v_s = slc.metadata.general_annotation.orbit_list.orbit[0].velocity;

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
            f_eta_c: f_eta_c_poly_container.polynomial,
            f_eta_c_t0: f_eta_c_t0_val,
            k_a: k_a_poly,
            k_a_t0: k_a_t0_val,
            f_c,
            v_s,
            nl_burst: nl_burst_usize,
            delta_t_s: delta_t_s_val,
            ns_swath,
            delta_tau_s,
            tau_0,
        }
    }
}

impl Default for DerampSlcBurst {
    fn default() -> Self {
        Self::new()
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

    pub fn apply_forward(
        &self,
        slc: &Sentinel1SlcIWSwath,
        burst_index: usize,
    ) -> Array2<Complex<f32>> {
        // TODO: remove burst_index
        self.apply(slc, Direction::Forward, burst_index)
    }

    pub fn apply_backward(
        &self,
        slc: &Sentinel1SlcIWSwath,
        burst_index: usize,
    ) -> Array2<Complex<f32>> {
        // TODO: remove burst_index
        self.apply(slc, Direction::Backward, burst_index)
    }

    // TODO: remove burst_index
    fn apply(
        &self,
        slc: &Sentinel1SlcIWSwath,
        direction: Direction,
        burst_index: usize,
    ) -> Array2<Complex<f32>> {
        let mode = self.mode;

        let RelevantParameters {
            k_psi,
            f_eta_c,
            f_eta_c_t0,
            k_a,
            k_a_t0,
            f_c,
            v_s,
            nl_burst,
            delta_t_s,
            ns_swath,
            delta_tau_s,
            tau_0,
        } = RelevantParameters::new(slc, burst_index); // TODO: remove burst_index
        // Calculate k_s (Doppler rate introduced by antenna steering)
        // k_s = (2 * v_s * f_c * k_psi) / c
        let c = 299792458.0; // speed of light in m/s
        let v_s_magnitude = (v_s.x * v_s.x + v_s.y * v_s.y + v_s.z * v_s.z).sqrt();
        let k_s = (2.0 * v_s_magnitude * f_c * k_psi) / c;

        // Helper function to calculate k_a at a given range time tau
        let k_a_at_tau = |tau: f64| -> f64 {
            let tau_diff = tau - k_a_t0;
            k_a.evaluate(tau_diff)
        };

        // Helper function to calculate f_eta_c at a given range time tau
        let f_eta_c_at_tau = |tau: f64| -> f64 {
            let tau_diff = tau - f_eta_c_t0;
            f_eta_c.evaluate(tau_diff)
        };

        // Helper function to calculate eta_c at a given range time tau
        // eta_c(tau) = -f_eta_c(tau) / k_a(tau)
        let calculate_eta_c = |tau_val_for_eta_c: f64| -> f64 {
            let f_eta_c_val = f_eta_c_at_tau(tau_val_for_eta_c);
            let k_a_val = k_a_at_tau(tau_val_for_eta_c);
            if k_a_val.abs() < 1e-9 {
                // Avoid division by zero or near-zero
                // This case needs careful consideration based on SAR physics.
                // Returning 0.0 implies eta_c = 0 if k_a is effectively zero.
                // The markdown states k_a is always negative, so it shouldn't be zero.
                // If it can be zero due to data issues, a panic or error might be more appropriate.
                // For now, retaining a default to avoid panic during processing of potentially valid edge cases.
                0.0
            } else {
                -f_eta_c_val / k_a_val
            }
        };

        // Calculate eta_c at mid-swath range time, to be used in eta_ref calculation
        // tau_mid_swath = tau(0) + (NS_swath / 2) * Δτ_s. NS_swath/2 is integer division for sample index.
        let mid_swath_sample_index = ns_swath / 2; // Integer division gives the floor for odd ns_swath
        let tau_mid_swath = tau_0 + mid_swath_sample_index as f64 * delta_tau_s;
        let eta_c_at_mid_swath = calculate_eta_c(tau_mid_swath);

        // Helper function to calculate k_t at a given range time tau
        // k_t = (k_a * k_s) / (k_a - k_s)
        let k_t_at_tau = |tau: f64| -> f64 {
            let k_a_val = k_a_at_tau(tau);
            // Add protection for k_a_val - k_s being zero if necessary,
            // though the document doesn't specify handling for k_a = k_s.
            if (k_a_val - k_s).abs() < 1e-9 {
                // Handle singularity: e.g., return a very large number or a representative value.
                // Or, if k_s is also very small, k_t might be considered 0.
                // This case implies alpha (Equ.3) is near zero.
                // k_t = k_s / alpha. If alpha is 0, k_t is infinite.
                // For now, let's return a large representative value or a flag.
                // This often indicates an issue or an extreme edge case in parameters.
                // Returning k_a_val as a fallback, though not physically robust without more context.
                // A proper handling might involve looking at limits or specific ESA guidance for this case.
                // For TOPSAR, k_a should generally be different from k_s.
                if k_s.abs() < 1e-9 {
                    return 0.0;
                } // if k_s is zero, k_t is zero unless k_a is also zero.
                return 1e12; // Placeholder for a very large k_t
            }
            (k_a_val * k_s) / (k_a_val - k_s)
        };

        // The deramping phase function phi(eta, tau)
        // For deramping only: phi = -π * k_t(τ) * (η - η_ref(τ))²
        // where η_ref(τ) = η_c(τ) - η_c_at_mid_swath
        // For deramping + demodulation: phi = -π * k_t(τ) * (η - η_ref(τ))² - 2π * f_eta_c(τ) * (η - η_ref(τ))
        let phi = |eta: f64, tau: f64| -> f64 {
            let k_t_val = k_t_at_tau(tau);

            let current_eta_c = calculate_eta_c(tau);
            let eta_ref_val = current_eta_c - eta_c_at_mid_swath;

            let eta_diff = eta - eta_ref_val;
            let phase_deramp_only = -PI * k_t_val * eta_diff * eta_diff;

            let final_phase = match mode {
                DerampingMode::Standard => phase_deramp_only,
                DerampingMode::FullDemodulation => {
                    let f_eta_c_val_at_tau = f_eta_c_at_tau(tau);
                    phase_deramp_only - 2.0 * PI * f_eta_c_val_at_tau * eta_diff
                }
            };

            match direction {
                Direction::Forward => final_phase,
                Direction::Backward => -final_phase,
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

        let buffer = &slc.data.array;
        let mut deramped = Array2::<Complex<f32>>::zeros((nl_burst, ns_swath));
        for (i, &eta_val) in eta.iter().enumerate() {
            for (j, &tau_val) in tau.iter().enumerate() {
                // Read the complex value
                let x = buffer[(i, j)];
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
