use ndarray::ArrayView2;
use num_complex::Complex;
use num_traits::{Float, NumCast};
use std::cmp::{max, min};

/// Maximum support radius allowed for any kernel.
/// This determines the size of the fixed array used for intermediate values.
pub const MAX_SUPPORT: usize = 8;

/// A trait for 1D interpolation kernels.
pub trait Kernel {
    // associated type
    type T: Float;

    /// Evaluates the kernel at a given distance from the center.
    ///
    /// Args:
    ///     t: The distance from the center (can be negative).
    ///
    /// Returns:
    ///     The kernel weight at distance t.
    fn evaluate(&self, t: Self::T) -> Self::T;

    /// Returns the support radius of the kernel.
    /// This is the number of neighboring samples to consider on each side.
    /// Must be less than or equal to MAX_SUPPORT.
    fn support_radius(&self) -> usize;
}

/// Bilinear (Tent/Triangle) kernel.
/// K(t) = max(0, 1 - |t|)
/// Support radius: 1
pub struct BilinearKernel<T: Float> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Kernel for BilinearKernel<T> {
    type T = T;

    fn evaluate(&self, t: T) -> T {
        T::max(T::zero(), T::one() - t.abs())
    }

    fn support_radius(&self) -> usize {
        1
    }
}

impl<T: Float> Default for BilinearKernel<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

/// Cubic (Catmull-Rom type) kernel.
/// Support radius: 2
pub struct CubicKernel<T: Float> {
    a: T,
}

impl<T: Float> CubicKernel<T> {
    pub fn new(a: T) -> Self {
        Self { a }
    }

    pub fn default() -> Self {
        Self {
            a: T::from(-0.5).unwrap(),
        } // Common coefficient value
    }
}

impl<T: Float> Kernel for CubicKernel<T> {
    type T = T;

    fn evaluate(&self, t: T) -> T {
        let t_abs = t.abs();
        let one = T::one();
        let two = T::from(2.0).unwrap();

        if t_abs >= T::zero() && t_abs < one {
            (self.a + two) * t_abs.powi(3) - (self.a + T::from(3.0).unwrap()) * t_abs.powi(2) + one
        } else if t_abs >= one && t_abs < two {
            self.a * t_abs.powi(3) - T::from(5.0).unwrap() * self.a * t_abs.powi(2)
                + T::from(8.0).unwrap() * self.a * t_abs
                - T::from(4.0).unwrap() * self.a
        } else {
            T::zero() // Outside support [-2, 2]
        }
    }

    fn support_radius(&self) -> usize {
        2
    }
}

/// Knab-windowed Sinc kernel.
/// K(t) = sinc(t) * knab_window(t, n, delta)
///
/// The Knab window is defined as:
///
/// w(t) = cosh(chi * sqrt(1 - (t/n)^2)) / cosh(chi)
///
/// where chi = pi * n * delta
///
/// and delta is the sampling interval, related to the oversampling factor beta by delta = 1 - 1/beta
pub struct KnabSincKernel<T: Float> {
    n: usize,
    chi: T,
}

impl<T: Float> KnabSincKernel<T> {
    /// Creates a new Knab-windowed Sinc kernel.
    ///
    /// Args:
    ///     - n: The support radius of the kernel.
    ///     - delta: The sampling interval, a value between 0 and 1, related to the oversampling factor beta by delta = 1 - 1/beta.
    pub fn new(n: usize, delta: T) -> Self {
        use std::f32::consts::PI;
        let zero = T::zero();
        let one = T::one();
        let delta = delta.clamp(zero, one);
        let n_t: T = num_traits::cast(n).unwrap();
        let pi_t = T::from(PI).unwrap();
        let chi = pi_t * n_t * delta;
        if n > MAX_SUPPORT {
            panic!("Kernel support radius exceeds MAX_SUPPORT");
        }
        Self { n, chi }
    }
}

impl<T: Float> Default for KnabSincKernel<T> {
    /// Creates a default Knab-windowed Sinc kernel with N=6 and delta=0.5.
    ///
    /// This configuration provides high-quality interpolation suitable for SAR/InSAR applications:
    /// - N=6 gives a support radius of 6 pixels on each side
    /// - delta=0.5 corresponds to an oversampling factor beta=2
    ///
    /// The resulting kernel has good frequency response characteristics:
    /// - High accuracy in the passband
    /// - Good suppression of aliasing
    /// - Reasonable computational cost
    fn default() -> Self {
        Self::new(6, T::from(0.6).unwrap())
    }
}

impl<T: Float> Kernel for KnabSincKernel<T> {
    type T = T;

    fn evaluate(&self, t: T) -> T {
        use std::f32::consts::PI;
        let pi_t = T::from(PI).unwrap() * t;
        let chi = T::from(self.chi).unwrap();
        let n_t = T::from(self.n as f32).unwrap();
        let t_abs = t.abs();

        // Kernel is zero outside the support [-N, N]
        if t_abs > n_t {
            return T::zero();
        }

        // Handle the t = 0 case separately
        if t_abs < T::from(1e-6).unwrap() {
            return T::one();
        }

        // Calculate sinc part: sin(pi*t) / (pi*t)
        let sinc_val = pi_t.sin() / pi_t;

        // Calculate Knab window part
        let cosh_chi = chi.cosh();
        if cosh_chi.abs() < T::from(1e-6).unwrap() {
            return if chi == T::zero() {
                sinc_val
            } else {
                T::zero()
            };
        }

        let t_over_n_squared = (t / n_t) * (t / n_t);
        let sqrt_term = (T::one() - t_over_n_squared).max(T::zero());
        let cosh_arg = chi * sqrt_term.sqrt();
        let knab_window_val = cosh_arg.cosh() / cosh_chi;

        sinc_val * knab_window_val
    }

    fn support_radius(&self) -> usize {
        self.n
    }
}

/// Clamps an index to be within the valid bounds of an array dimension.
///
/// Args:
///     - idx: The original index (can be negative or out of bounds).
///     - size: The size of the dimension (number of elements).
///
/// Returns:
///     The clamped index, guaranteed to be within `0..size`.
fn clamp_index(idx: isize, size: usize) -> usize {
    // Ensure the index is not negative and not beyond the last element.
    max(0, min(idx, size as isize - 1)) as usize
}

/// Performs 2D interpolation on a discrete 2D signal using a separable kernel.
///
/// Assumes the kernel K(x, y) = kernel_1d(x) * kernel_1d(y).
/// Implements the interpolation via two sequential 1D passes.
/// Each 1D pass uses 2*N + 1 samples centered around the target coordinate.
///
/// Args:
///     - image: A 2D array view representing the discrete signal I(m, n).
///            ndarray uses (row, col) indexing, so image[[m, n]] accesses I(m, n).
///     - i: The continuous vertical coordinate (row direction) to interpolate at.
///     - j: The continuous horizontal coordinate (column direction) to interpolate at.
///     - kernel: A reference to the kernel to use for interpolation.
///
/// Returns:
///     The interpolated value `Complex<f32>` at (i, j).
///
/// Boundary Handling:
///     Uses clamped indices (replicates edge pixel values).
pub fn interpolate_2d<K, T>(image: ArrayView2<Complex<T>>, i: T, j: T, kernel: &K) -> Complex<T>
where
    K: Kernel<T = T>,
    T: Float,
{
    let (num_rows, num_cols) = image.dim();
    let radius = kernel.support_radius() as isize;
    assert!(
        radius as usize <= MAX_SUPPORT,
        "Kernel support radius exceeds MAX_SUPPORT"
    );

    // Determine the range of discrete row indices (m) needed for the vertical pass
    // Indices m range from floor(i) - radius to floor(i) + radius
    let i_floor = i.floor().to_isize().unwrap();
    let m_start = i_floor - radius;
    let m_end = i_floor + radius;

    // Determine the range of discrete column indices (n) needed for the horizontal pass
    // Indices n range from floor(j) - radius to floor(j) + radius
    let j_floor = j.floor().to_isize().unwrap();
    let n_start = j_floor - radius;
    let n_end = j_floor + radius;

    // --- Pass 1: Vertical Interpolation ---
    // For each relevant column n, calculate the intermediate value g(n, i)
    // g(n, i) = sum_m I(m, n) * K(i - m)
    let num_intermediate = (n_end - n_start + 1).max(0) as usize;
    assert!(
        num_intermediate <= 2 * MAX_SUPPORT + 1,
        "Number of intermediate values exceeds array size"
    );

    let mut intermediate_values =
        [(0isize, Complex::<T>::new(T::zero(), T::zero())); 2 * MAX_SUPPORT + 1];
    let mut intermediate_count = 0;

    for n in n_start..=n_end {
        // Clamp column index n to be within image bounds for accessing I(m, n)
        let n_clamped = clamp_index(n, num_cols);

        let mut vertical_sum = Complex::<T>::new(T::zero(), T::zero());
        for m in m_start..=m_end {
            // Clamp row index m to be within image bounds for accessing I(m, n)
            let m_clamped = clamp_index(m, num_rows);

            // Calculate kernel weight K(i - m) using the original (unclamped) m index
            let m_t: T = num_traits::cast(m).unwrap();
            let weight = kernel.evaluate(i - m_t);

            // Add contribution I(m, n) * K(i - m)
            if weight != T::zero() {
                vertical_sum = vertical_sum + image[[m_clamped, n_clamped]] * weight;
            }
        }
        // Store the intermediate result g(n, i) for column n
        intermediate_values[intermediate_count] = (n, vertical_sum);
        intermediate_count += 1;
    }

    // --- Pass 2: Horizontal Interpolation ---
    // Interpolate the intermediate values g(n, i) along the horizontal direction
    // result = sum_n g(n, i) * K(j - n)
    let mut final_value = Complex::<T>::new(T::zero(), T::zero());
    for (n_idx_ref, complex_val_ref) in intermediate_values.iter().take(intermediate_count) {
        // Calculate kernel weight K(j - n) using the original (unclamped) n index
        let n_idx_t: T = num_traits::cast(*n_idx_ref).unwrap();
        let weight: T = kernel.evaluate(j - n_idx_t);

        // Add contribution g(n, i) * K(j - n)
        if weight != T::zero() {
            let val_to_add: Complex<T> = *complex_val_ref * weight;
            final_value = final_value + val_to_add;
        }
    }

    final_value
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bilinear_kernel() {
        let kernel = BilinearKernel::default();
        assert_relative_eq!(kernel.evaluate(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(kernel.evaluate(0.5), 0.5, epsilon = 1e-6);
        assert_relative_eq!(kernel.evaluate(1.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(kernel.evaluate(1.5), 0.0, epsilon = 1e-6);
        assert_eq!(kernel.support_radius(), 1);
        assert!(kernel.support_radius() <= MAX_SUPPORT);
    }

    #[test]
    fn test_cubic_kernel() {
        let kernel: CubicKernel<f64> = CubicKernel::default();
        assert_relative_eq!(kernel.evaluate(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(kernel.evaluate(1.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(kernel.evaluate(2.0), 0.0, epsilon = 1e-6);
        assert_eq!(kernel.support_radius(), 2);
        assert!(kernel.support_radius() <= MAX_SUPPORT);
    }

    #[test]
    fn test_knab_sinc_kernel() {
        let kernel = KnabSincKernel::new(2, 0.5);
        assert_relative_eq!(kernel.evaluate(0.0), 1.0, epsilon = 1e-6);
        assert_eq!(kernel.support_radius(), 2);
        assert!(kernel.support_radius() <= MAX_SUPPORT);
    }

    #[test]
    #[should_panic(expected = "Kernel support radius exceeds MAX_SUPPORT")]
    fn test_max_support_limit() {
        let kernel = KnabSincKernel::new(MAX_SUPPORT + 1, 0.5);
        let _ = kernel.support_radius();
    }
}
