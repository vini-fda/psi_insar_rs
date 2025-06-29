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
    t_interval: f64,
    t: f64,
) -> Vector3<f64> {
    let t2 = t * t;
    let t3 = t * t * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p_0 + h01 * p_1 + (h10 * m_0 + h11 * m_1) * t_interval
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
    t_interval: f64,
    t: f64,
) -> Vector3<f64> {
    let t2 = t * t;

    let h00_deriv = 6.0 * t2 - 6.0 * t;
    let h10_deriv = 3.0 * t2 - 4.0 * t + 1.0;
    let h01_deriv = -6.0 * t2 + 6.0 * t;
    let h11_deriv = 3.0 * t2 - 2.0 * t;
    // if t > 0.99 {
    //     println!("h00_deriv = {}", h00_deriv);
    //     let a = h00_deriv * p_0;
    //     let b = h10_deriv * m_0;
    //     let c = h01_deriv * p_1;
    //     let d = h11_deriv * m_1;
    //     let sum = a + b + c + d;
    //     print_vec(a);
    //     print_vec(b);
    //     print_vec(c);
    //     print_vec(d);
    //     print_vec(sum);
    // }

    (h00_deriv * p_0 + h01_deriv * p_1) / t_interval + h10_deriv * m_0 + h11_deriv * m_1
}

/// Calculates the second derivative of the polynomial value at t \in [0, 1],
/// p_0: value at starting point
/// m_0: slope/derivative at starting point
/// p_1: value at end point
/// m_1: slope/derivative at end point
#[inline(always)]
pub fn unit_second_derivative_interval_cubic_hermite_spline_interpolation(
    p_0: Vector3<f64>,
    m_0: Vector3<f64>,
    p_1: Vector3<f64>,
    m_1: Vector3<f64>,
    t_interval: f64,
    t: f64,
) -> Vector3<f64> {
    let h00_second_deriv = 12.0 * t - 6.0;
    let h10_second_deriv = 6.0 * t - 4.0;
    let h01_second_deriv = -12.0 * t + 6.0;
    let h11_second_deriv = 6.0 * t - 2.0;

    (h00_second_deriv * p_0 + h01_second_deriv * p_1) / (t_interval * t_interval)
        + (h10_second_deriv * m_0 + h11_second_deriv * m_1) / t_interval
}

fn print_vec(v: nalgebra::Vector3<f64>) {
    println!("[{}, {}, {}]", v.x, v.y, v.z);
}
