use nalgebra::Vector3;

/// Calculates the polynomial value at t \in [0, 1],
/// p_0: value at starting point
/// m_0: slope/derivative at starting point
/// p_1: value at end point
/// m_1: slope/derivative at end point
#[inline(always)]
pub fn unit_interval_cubic_hermite_spline_interpolation(
    p_0: Vector3<f64>,
    m_0: Vector3<f64>,
    p_1: Vector3<f64>,
    m_1: Vector3<f64>,
    t: f64,
) -> Vector3<f64> {
    let t2 = t * t;
    let t3 = t * t * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * p_0
        + (t3 - 2.0 * t2 + t) * m_0
        + (-2.0 * t3 + 3.0 * t2) * p_1
        + (t3 - t2) * m_1
}

/// Calculates the derivative of the polynomial value at t \in [0, 1],
/// p_0: value at starting point
/// m_0: slope/derivative at starting point
/// p_1: value at end point
/// m_1: slope/derivative at end point
#[inline(always)]
pub fn unit_derivative_interval_cubic_hermite_spline_interpolation(
    p_0: Vector3<f64>,
    m_0: Vector3<f64>,
    p_1: Vector3<f64>,
    m_1: Vector3<f64>,
    t: f64,
) -> Vector3<f64> {
    let t2 = t * t;

    let h00_deriv = 6.0 * t2 - 6.0 * t;
    let h10_deriv = 3.0 * t2 - 4.0 * t + 1.0;
    let h01_deriv = -6.0 * t2 + 6.0 * t;
    let h11_deriv = 3.0 * t2 - 2.0 * t;

    h00_deriv * p_0 + h10_deriv * m_0 + h01_deriv * p_1 + h11_deriv * m_1
}
