use crate::{
    constants::C_LIGHT,
    coregistration::bilinear_polynomial::BilinearPolynomial,
    metadata::annotation_xml::{Polynomial, SlcProductAnnotation, Velocity},
    sentinel::{Sentinel1SlcIWBurst, Sentinel1SlcIWSwath},
};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, s};
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
    params: RelevantParameters,
    direction: Direction,
}

/// - Forward, or *Deramping*, is the removal of the linear frequency modulation introduced by antenna steering
/// - Backward, or *Reramping*, is the inverse of deramping, i.e. it adds back the linear frequency modulation introduced by antenna steering
///
/// By default, we want to deramp the SLC data, then reramp it back to the original data.
#[derive(Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl Default for Direction {
    fn default() -> Self {
        Self::Forward
    }
}

#[derive(Clone, Debug)]
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
    /// Azimuth FM Rate
    azimuth_fm_rate_polynomial: BilinearPolynomial,
    /// DC Estimate
    dc_estimate_polynomial: BilinearPolynomial,
}

impl RelevantParameters {
    /// Extracts the relevant parameters for the deramping from the Sentinel1SlcIWSwath metadata
    // TODO: check if there's a better solution than burst_index
    pub fn new(slc_metadata: &SlcProductAnnotation, burst_index: usize) -> Self {
        let burst_count = slc_metadata.swath_timing.burst_list.bursts.len();
        assert!(burst_index < burst_count);
        // k_psi: Azimuth steering rate (radians/s)
        let k_psi = slc_metadata
            .general_annotation
            .product_information
            .azimuth_steering_rate
            .to_radians();

        // Nl_burst: Number of lines per burst
        let nl_burst_usize = slc_metadata.swath_timing.lines_per_burst;
        let nl_burst_f64 = nl_burst_usize as f64;

        // Δt_s: Azimuth time interval
        let delta_t_s_val = slc_metadata
            .image_annotation
            .image_information
            .azimuth_time_interval;

        // Current burst metadata for reference times
        let current_burst_metadata = &slc_metadata.swath_timing.burst_list.bursts[burst_index];
        // ANX time is useful only for  matching with Orbital information
        // TODO: pick velocity at middle of the azimuth time in the burst
        //let burst_start_anx_time = current_burst_metadata.azimuth_anx_time;
        let first_line_azimuth_time = current_burst_metadata.azimuth_time; // DateTime<Utc>

        // Duration, in seconds, from the middle of the burst compared to the first azimuth line
        let eta_mid_time = (nl_burst_f64 / 2.0) * delta_t_s_val;

        // f_eta_c: Doppler centroid frequency polynomial
        // Select the polynomial whose azimuth time is closest to eta_mid_anx_time.
        let dc_estimate_list = &slc_metadata.doppler_centroid.dc_estimate_list.dc_estimate;

        if dc_estimate_list.is_empty() {
            panic!("Doppler centroid estimate list is empty. Cannot select f_eta_c polynomial.");
        }

        // Bilinear polynomial f(eta, tau)
        let dc_estimate_polynomial = BilinearPolynomial::new(
            dc_estimate_list
                .iter()
                .map(|dc_estimate| {
                    let duration = dc_estimate
                        .azimuth_time
                        .signed_duration_since(first_line_azimuth_time);
                    let azimuth_time =
                        duration.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                    (
                        azimuth_time,
                        dc_estimate.data_dc_polynomial.polynomial.clone(),
                    )
                })
                .collect(),
        );

        let selected_dc_estimate = dc_estimate_list
            .iter()
            .min_by(|a, b| {
                // Convert DcEstimate's azimuth_time (DateTime<Utc>) to an f64 time interval
                // since the first azimuth line
                let duration_a = a.azimuth_time.signed_duration_since(first_line_azimuth_time);
                let item_a_secs =
                    duration_a.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                let duration_b = b.azimuth_time.signed_duration_since(first_line_azimuth_time);
                let item_b_secs =
                    duration_b.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;

                let diff_a = (item_a_secs - eta_mid_time).abs();
                let diff_b = (item_b_secs - eta_mid_time).abs();
                diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("dc_estimate_list was checked not to be empty but min_by found no minimum. This indicates a data issue or NaN times.");

        let f_eta_c = selected_dc_estimate.data_dc_polynomial.polynomial.clone();
        let f_eta_c_t0_val = selected_dc_estimate.t0;

        // k_a: Azimuth FM rate polynomial
        // Select the polynomial whose azimuth time is closest to eta_mid_anx_time.
        let azimuth_fm_rate_list = &slc_metadata
            .general_annotation
            .azimuth_fm_rate_list
            .azimuth_fm_rate;

        if azimuth_fm_rate_list.is_empty() {
            panic!("Azimuth FM rate list is empty. Cannot select k_a polynomial.");
        }

        // Bilinear polynomial f(eta, tau)
        let azimuth_fm_rate_polynomial = BilinearPolynomial::new(
            azimuth_fm_rate_list
                .iter()
                .map(|azimuth_fm_rate| {
                    let duration = azimuth_fm_rate
                        .azimuth_time
                        .signed_duration_since(first_line_azimuth_time);
                    let azimuth_time =
                        duration.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                    (
                        azimuth_time,
                        azimuth_fm_rate
                            .azimuth_fm_rate_polynomial
                            .coefficients
                            .clone(),
                    )
                })
                .collect(),
        );

        let selected_azimuth_fm_rate_item = azimuth_fm_rate_list
            .iter()
            .min_by(|a, b| {
                // Convert AzimuthFmRate's azimuth_time (DateTime<Utc>) to an f64 ANX-equivalent time
                let duration_a = a.azimuth_time.signed_duration_since(first_line_azimuth_time);
                let item_a_anx_equivalent =
                    duration_a.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                let duration_b = b.azimuth_time.signed_duration_since(first_line_azimuth_time);
                let item_b_anx_equivalent =
                    duration_b.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;

                let diff_a = (item_a_anx_equivalent - eta_mid_time).abs();
                let diff_b = (item_b_anx_equivalent - eta_mid_time).abs();
                diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("azimuth_fm_rate_list was checked not to be empty but min_by found no minimum. This indicates a data issue or NaN times.");

        let k_a_poly = selected_azimuth_fm_rate_item
            .azimuth_fm_rate_polynomial
            .coefficients
            .clone();
        let k_a_t0_val = selected_azimuth_fm_rate_item.t0;

        // f_c: Radar frequency
        let f_c = slc_metadata
            .general_annotation
            .product_information
            .radar_frequency;

        // V_S: Satellite velocity vector [x,y,z]
        let v_s = slc_metadata.general_annotation.orbit_list.orbit[0].velocity;

        // NS_swath: Number of samples in swath
        let ns_swath = slc_metadata
            .image_annotation
            .image_information
            .number_of_samples;

        // Δτ_s: Range sampling interval (inverse of range sampling rate)
        let delta_tau_s = 1.0
            / slc_metadata
                .general_annotation
                .product_information
                .range_sampling_rate;

        // τ(0): Slant range time
        let tau_0 = slc_metadata
            .image_annotation
            .image_information
            .slant_range_time;

        Self {
            k_psi,
            f_eta_c,
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
            dc_estimate_polynomial,
            azimuth_fm_rate_polynomial,
        }
    }
}

impl DerampSlcBurst {
    pub fn process_swath(slc: &Sentinel1SlcIWSwath) -> Array2<Complex<f32>> {
        let burst_count = slc.metadata.swath_timing.burst_list.bursts.len();
        let nl_burst = slc.metadata.swath_timing.lines_per_burst;
        let nl_swath = slc
            .metadata
            .image_annotation
            .image_information
            .number_of_lines;
        let ns_swath = slc
            .metadata
            .image_annotation
            .image_information
            .number_of_samples;
        let mut deramped = Array2::<Complex<f32>>::zeros((nl_swath, ns_swath));
        for burst_index in 0..burst_count {
            let deramp = DerampSlcBurst::new(&slc.metadata, burst_index);
            let mut output_view =
                deramped.slice_mut(s![burst_index * nl_burst..(burst_index + 1) * nl_burst, ..]);
            deramp.buffer_apply(
                &mut output_view,
                slc.burst(burst_index).burst_data.array.view(),
            );
        }
        deramped
    }

    /// Applies the Debursting operator in the default Forward mode to the SLC Burst.
    pub fn process_burst(slc: &Sentinel1SlcIWBurst) -> Array2<Complex<f32>> {
        let burst_index = slc.burst_index;
        DerampSlcBurst::new(&slc.metadata, burst_index).apply(slc.burst_data.array.view())
    }

    pub fn new(metadata: &SlcProductAnnotation, burst_index: usize) -> Self {
        let params = RelevantParameters::new(metadata, burst_index);
        // log::info!("Relevant Deramp Parameters = {params:?}");
        Self {
            mode: DerampingMode::default(),
            params,
            direction: Direction::Forward,
        }
    }

    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    pub fn set_mode(mut self, mode: DerampingMode) -> Self {
        self.mode = mode;
        self
    }

    /// phi(eta, tau)
    pub fn phi(&self, eta: f64, tau: f64) -> f64 {
        let &RelevantParameters {
            nl_burst,
            delta_t_s,
            ..
        } = &self.params;
        let mode = self.mode;
        let direction = self.direction;

        let offset = (nl_burst as f64) * delta_t_s / 2.0;
        let eta_diff = eta - offset - self.eta_ref(tau);
        let phase_deramp_only = -PI * self.k_t(tau) * eta_diff * eta_diff;

        let final_phase = match mode {
            DerampingMode::Standard => phase_deramp_only,
            DerampingMode::FullDemodulation => {
                phase_deramp_only - 2.0 * PI * self.f_eta_c(tau) * eta
            }
        };

        match direction {
            Direction::Forward => final_phase,
            Direction::Backward => -final_phase,
        }
    }

    /// Doppler centroid rate in the focused TOPS SLC data [Hz/s].
    ///
    /// k_t is obtained by scaling the RAW time rate (ks) with the conversion factor (α) between
    /// focused and raw time such that α = 1 - (k_s/k_a(tau))
    pub fn k_t(&self, tau: f64) -> f64 {
        let k_a = self.k_a(tau);
        let k_s = self.k_s();
        k_a * k_s / (k_a - k_s)
    }

    /// Doppler FM rate [Hz/s].
    ///
    /// This is the classical azimuth FM rate which is always negative. The azimuth FM rate is
    /// provided as a sequence of range polynomial regularly updated with azimuth time 𝜂. For
    /// deramping the i-th burst, it is recommended to use closest polynomial to 𝜂_𝑚𝑖𝑑 of the i-th burst.
    pub fn k_a(&self, tau: f64) -> f64 {
        let RelevantParameters { k_a, k_a_t0, .. } = &self.params;
        let tau_diff = tau - k_a_t0;
        k_a.evaluate(tau_diff)
    }

    /// Doppler Centroid rate introduced by the scanning of the antenna 𝑘_𝜓 [Hz/s]. This rate is applicable to the RAW data,
    /// and needs to be converted to k_t before applying to the SLC data.
    pub fn k_s(&self) -> f64 {
        let RelevantParameters {
            k_psi, f_c, v_s, ..
        } = self.params;
        let v_s_magnitude = (v_s.x * v_s.x + v_s.y * v_s.y + v_s.z * v_s.z).sqrt();
        (2.0 * v_s_magnitude * f_c * k_psi) / C_LIGHT
    }

    /// Reference zero-Doppler Azimuth Time
    pub fn eta_ref(&self, tau: f64) -> f64 {
        let RelevantParameters {
            ns_swath,
            delta_tau_s,
            tau_0,
            ..
        } = self.params;
        let tau_mid = tau_0 + (ns_swath / 2) as f64 * delta_tau_s;
        self.eta_c(tau) - self.eta_c(tau_mid)
    }

    /// Beam centre crossing time [s]
    pub fn eta_c(&self, tau: f64) -> f64 {
        -(self.f_eta_c(tau) / self.k_a(tau))
    }

    /// Doppler Centroid frequency [Hz].
    ///
    /// This is provided as a sequence of range polynomial
    /// regularly updated with azimuth time 𝜂. For deramping the i-th burst, it is recommended to
    /// use closest polynomial to 𝜂_𝑚𝑖𝑑 of the i-th burst.
    pub fn f_eta_c(&self, tau: f64) -> f64 {
        let RelevantParameters {
            f_eta_c,
            f_eta_c_t0,
            ..
        } = &self.params;
        let tau_diff = tau - f_eta_c_t0;
        f_eta_c.evaluate(tau_diff)
    }

    // ----- TODO -------
    pub fn k_a_(&self, eta: f64) -> Polynomial {
        let RelevantParameters {
            azimuth_fm_rate_polynomial,
            ..
        } = &self.params;
        // let tau_diff = tau - k_a_t0;
        azimuth_fm_rate_polynomial.interpolate(eta)
    }

    pub fn f_eta_c_(&self, eta: f64) -> Polynomial {
        let RelevantParameters {
            dc_estimate_polynomial,
            ..
        } = &self.params;
        // let tau_diff = tau - f_eta_c_t0;
        dc_estimate_polynomial.interpolate(eta)
    }

    pub fn eta_c_(&self, k_a: &Polynomial, f_eta_c: &Polynomial, tau: f64) -> f64 {
        let RelevantParameters {
            f_eta_c_t0, k_a_t0, ..
        } = &self.params;
        -(f_eta_c.evaluate(tau - f_eta_c_t0) / k_a.evaluate(tau - k_a_t0))
    }

    pub fn eta_ref_(&self, k_a: &Polynomial, f_eta_c: &Polynomial, tau: f64) -> f64 {
        let RelevantParameters {
            ns_swath,
            delta_tau_s,
            tau_0,
            ..
        } = self.params;
        // TODO: understand why SNAP uses firstValidPixel in tmp2, and not tau_mid
        // https://github.com/senbox-org/microwave-toolbox/blob/master/sar-commons/src/main/java/eu/esa/sar/commons/Sentinel1Utils.java#L594
        let tau_mid = tau_0 + (ns_swath / 2) as f64 * delta_tau_s;

        self.eta_c_(k_a, f_eta_c, tau) - self.eta_c_(k_a, f_eta_c, tau_mid)
    }

    pub fn k_t_(&self, k_a: &Polynomial, tau: f64) -> f64 {
        let RelevantParameters { k_a_t0, .. } = &self.params;
        let k_a = k_a.evaluate(tau - k_a_t0);
        let k_s = self.k_s();
        k_a * k_s / (k_a - k_s)
    }

    pub fn phi_(&self, eta: f64, tau: f64, k_a: &Polynomial, f_eta_c: &Polynomial) -> f64 {
        let &RelevantParameters {
            f_eta_c_t0,
            nl_burst,
            delta_t_s,
            ..
        } = &self.params;
        let mode = self.mode;
        let direction = self.direction;

        let offset = (nl_burst as f64) * delta_t_s / 2.0;
        let eta_diff = eta - offset - self.eta_ref_(k_a, f_eta_c, tau);
        let phase_deramp_only = -PI * self.k_t_(k_a, tau) * eta_diff * eta_diff;

        let final_phase = match mode {
            DerampingMode::Standard => phase_deramp_only,
            DerampingMode::FullDemodulation => {
                phase_deramp_only - 2.0 * PI * f_eta_c.evaluate(tau - f_eta_c_t0) * eta
            }
        };

        match direction {
            Direction::Forward => final_phase,
            Direction::Backward => -final_phase,
        }
    }
    // --- END ----
    pub fn buffer_apply(
        &self,
        output_buf: &mut ArrayViewMut2<Complex<f32>>,
        array: ArrayView2<Complex<f32>>,
    ) {
        let &RelevantParameters {
            nl_burst,
            delta_t_s,
            ns_swath,
            delta_tau_s,
            tau_0,
            ref azimuth_fm_rate_polynomial,
            ref dc_estimate_polynomial,
            ..
        } = &self.params;
        // Calculate eta vector (azimuth times)
        let eta: Array1<f64> = Array1::from_iter((0..nl_burst).map(|i| i as f64 * delta_t_s));

        // Calculate tau vector (range times for each sample)
        // tau(i) = tau(0) + i * Δτ_s
        let tau: Array1<f64> =
            Array1::from_iter((0..ns_swath).map(|i| tau_0 + i as f64 * delta_tau_s));

        // let buffer = &slc.burst_data.array;
        //let mut deramped = Array2::<Complex<f32>>::zeros((nl_burst, ns_swath));
        for (i, &eta_val) in eta.iter().enumerate() {
            // TODO: READ THIS: https://github.com/senbox-org/microwave-toolbox/blob/254aa8f5de2cfe65138a8b7edf9d596eb3ba03c1/sar-commons/src/main/java/eu/esa/sar/commons/Sentinel1Utils.java#L719
            let k_a = azimuth_fm_rate_polynomial.interpolate(eta_val);
            let f_eta_c = dc_estimate_polynomial.interpolate(eta_val);
            for (j, &tau_val) in tau.iter().enumerate() {
                // Read the complex value
                let x = array[(i, j)];
                let x = Complex::new(x.re as f64, x.im as f64);

                // Calculate and apply phase
                let phase = self.phi_(eta_val, tau_val, &k_a, &f_eta_c);
                // let phase = self.phi(eta_val, tau_val);
                let phase_cos = phase.cos();
                let phase_sin = phase.sin();
                let x = x * Complex::new(phase_cos, phase_sin);

                // Convert back to Complex<f32> and write
                let x = Complex::<f32>::new(x.re as f32, x.im as f32);
                output_buf[[i, j]] = x;
            }
        }
    }

    pub fn apply(&self, array: ArrayView2<Complex<f32>>) -> Array2<Complex<f32>> {
        let &RelevantParameters {
            nl_burst,
            delta_t_s,
            ns_swath,
            delta_tau_s,
            tau_0,
            ref azimuth_fm_rate_polynomial,
            ref dc_estimate_polynomial,
            ..
        } = &self.params;
        // Calculate eta vector (azimuth times)
        let eta: Array1<f64> = Array1::from_iter((0..nl_burst).map(|i| i as f64 * delta_t_s));

        // Calculate tau vector (range times for each sample)
        // tau(i) = tau(0) + i * Δτ_s
        let tau: Array1<f64> =
            Array1::from_iter((0..ns_swath).map(|i| tau_0 + i as f64 * delta_tau_s));

        // let buffer = &slc.burst_data.array;
        let mut deramped = Array2::<Complex<f32>>::zeros((nl_burst, ns_swath));
        for (i, &eta_val) in eta.iter().enumerate() {
            // TODO: READ THIS: https://github.com/senbox-org/microwave-toolbox/blob/254aa8f5de2cfe65138a8b7edf9d596eb3ba03c1/sar-commons/src/main/java/eu/esa/sar/commons/Sentinel1Utils.java#L719
            let k_a = azimuth_fm_rate_polynomial.interpolate(eta_val);
            let f_eta_c = dc_estimate_polynomial.interpolate(eta_val);
            for (j, &tau_val) in tau.iter().enumerate() {
                // Read the complex value
                let x = array[(i, j)];
                let x = Complex::new(x.re as f64, x.im as f64);

                // Calculate and apply phase
                let phase = self.phi_(eta_val, tau_val, &k_a, &f_eta_c);
                // let phase = self.phi(eta_val, tau_val);
                let phase_cos = phase.cos();
                let phase_sin = phase.sin();
                let x = x * Complex::new(phase_cos, phase_sin);

                // Convert back to Complex<f32> and write
                let x = Complex::<f32>::new(x.re as f32, x.im as f32);
                deramped[[i, j]] = x;
            }
        }

        deramped
    }

    pub fn debug_array(&self) -> Array2<Complex<f32>> {
        let &RelevantParameters {
            nl_burst,
            delta_t_s,
            ns_swath,
            delta_tau_s,
            tau_0,
            ref azimuth_fm_rate_polynomial,
            ref dc_estimate_polynomial,
            ..
        } = &self.params;
        // Calculate eta vector (azimuth times)
        let eta: Array1<f64> = Array1::from_iter((0..nl_burst).map(|i| i as f64 * delta_t_s));

        // Calculate tau vector (range times for each sample)
        // tau(i) = tau(0) + i * Δτ_s
        let tau: Array1<f64> =
            Array1::from_iter((0..ns_swath).map(|i| tau_0 + i as f64 * delta_tau_s));

        // let buffer = &slc.burst_data.array;
        let mut deramped = Array2::<Complex<f32>>::zeros((nl_burst, ns_swath));
        for (i, &eta_val) in eta.iter().enumerate() {
            // TODO: READ THIS: https://github.com/senbox-org/microwave-toolbox/blob/254aa8f5de2cfe65138a8b7edf9d596eb3ba03c1/sar-commons/src/main/java/eu/esa/sar/commons/Sentinel1Utils.java#L719
            let k_a = azimuth_fm_rate_polynomial.interpolate(eta_val);
            let f_eta_c = dc_estimate_polynomial.interpolate(eta_val);
            for (j, &tau_val) in tau.iter().enumerate() {
                // Calculate and apply phase
                let phase = -self.phi_(eta_val, tau_val, &k_a, &f_eta_c);
                // let phase = self.phi(eta_val, tau_val);
                let phase_cos = phase.cos();
                let phase_sin = phase.sin();
                let x = Complex::new(phase_cos, phase_sin);

                // Convert back to Complex<f32> and write
                let x = Complex::<f32>::new(x.re as f32, x.im as f32);
                deramped[[i, j]] = x;
            }
        }

        deramped
    }
}
