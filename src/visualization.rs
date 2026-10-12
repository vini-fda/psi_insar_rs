/// Colors of the cubehelix cyclic colormap, from the BEAM colour palette.
const CUBEHELIX_COLORS: [[f32; 3]; 8] = [
    [110.0 / 255.0, 60.0 / 255.0, 170.0 / 255.0],
    [210.0 / 255.0, 60.0 / 255.0, 160.0 / 255.0],
    [1.0, 110.0 / 255.0, 70.0 / 255.0],
    [200.0 / 255.0, 200.0 / 255.0, 50.0 / 255.0],
    [80.0 / 255.0, 245.0 / 255.0, 100.0 / 255.0],
    [25.0 / 255.0, 200.0 / 255.0, 180.0 / 255.0],
    [60.0 / 255.0, 130.0 / 255.0, 220.0 / 255.0],
    [100.0 / 255.0, 70.0 / 255.0, 190.0 / 255.0],
];

/// Turbo colormap (polynomial approximation)
/// Reference: <https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html>
/// Original LUT: <https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f>
/// Authors: Anton Mikhailov (mikhailov@google.com), Ruofei Du (ruofei@google.com)
pub fn turbo_colormap(x: f32) -> [f32; 3] {
    // Coefficients for the polynomial approximation
    let k_red_vec4: [f32; 4] = [0.13572138, 4.615_392_7, -42.660_324, 132.131_09];
    let k_green_vec4: [f32; 4] = [0.09140261, 2.194_188_4, 4.842_966_6, -14.185_034];
    let k_blue_vec4: [f32; 4] = [0.106_673_3, 12.641_946, -60.582_047, 110.362_77];
    let k_red_vec2: [f32; 2] = [-152.942_4, 59.286_38];
    let k_green_vec2: [f32; 2] = [4.277_298_5, 2.829_566];
    let k_blue_vec2: [f32; 2] = [-89.903_11, 27.348_25];

    // Clamp input to [0, 1]
    let y = x.clamp(0.0, 1.0);

    // Compute polynomial terms
    let v4: [f32; 4] = [1.0, y, y * y, y * y * y];
    let v2: [f32; 2] = [v4[2] * v4[2], v4[3] * v4[2]];

    fn dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
        a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>()
    }

    // Compute dot products
    let red = dot(&v4, &k_red_vec4) + dot(&v2, &k_red_vec2);
    let green = dot(&v4, &k_green_vec4) + dot(&v2, &k_green_vec2);
    let blue = dot(&v4, &k_blue_vec4) + dot(&v2, &k_blue_vec2);

    [red, green, blue]
}

#[inline(always)]
fn to_byte(x: f32) -> u8 {
    (255.0 * x) as u8
}

pub fn normalize_values(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min = values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let range = max - min;
    if range > 0.0 {
        values.iter().map(|x| (x - min) / range).collect()
    } else {
        let n = values.len();
        vec![0.0; n]
    }
}

pub fn turbo_colorized_values(values: &[f32]) -> Vec<u32> {
    let max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite()) // filters out NaNs and infinities
        .max_by(|a, b| a.partial_cmp(b).unwrap());
    let min = values
        .iter()
        .copied()
        .filter(|v| v.is_finite()) // filters out NaNs and infinities
        .min_by(|a, b| a.partial_cmp(b).unwrap());
    if max.is_none() || min.is_none() {
        return vec![0xFFFFFFFF];
    }
    let max = max.unwrap();
    let min = min.unwrap();
    let range = max - min;
    if range > 0.0 {
        values
            .iter()
            .map(|x| turbo_colormap_bytes((x - min) / range))
            .collect()
    } else {
        vec![0xFFFFFFFF]
    }
}

pub fn turbo_colormap_bytes(x: f32) -> u32 {
    if x.is_nan() {
        return 0x00000000;
    }
    let [red, green, blue] = turbo_colormap(x).map(to_byte);
    let alpha = 255u8;
    // Pack into u32 as 0xAARRGGBB or 0xRRGGBBAA depending on your convention.
    // Here we use 0xRRGGBBAA (common in WebGL, etc.)
    ((red as u32) << 24) | ((green as u32) << 16) | ((blue as u32) << 8) | (alpha as u32)
}

/// Uniform Catmull–Rom interpolation between the control points `p1` and `p2`.
///
/// `p0` and `p3` are the neighbouring control points.
///
/// The spline passes exactly through its control points (`f(0) == p1`,
/// `f(1) == p2`) and is C1 continuous where segments meet.
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (3.0f32.mul_add(p1, -p0) - 3.0 * p2 + p3) * t3)
}

/// Evaluates the closed cubehelix loop at `t`, taken modulo `1`.
fn cubehelix_colormap(t: f32) -> [f32; 3] {
    const CONTROL_POINTS: usize = CUBEHELIX_COLORS.len();

    // The loop has as many segments as control points, because the last segment closes it
    // by running from point 7 back to point 0.
    let x = t.rem_euclid(1.0) * CONTROL_POINTS as f32;
    let segment = (x.floor() as usize) % CONTROL_POINTS;
    let local = x - x.floor();

    let point = |offset: usize| CUBEHELIX_COLORS[(segment + offset) % CONTROL_POINTS];
    let (p0, p1, p2, p3) = (point(CONTROL_POINTS - 1), point(0), point(1), point(2));

    let mut rgb = [0.0; 3];
    for (channel, out) in rgb.iter_mut().enumerate() {
        *out =
            catmull_rom(p0[channel], p1[channel], p2[channel], p3[channel], local).clamp(0.0, 1.0);
    }
    rgb
}

/// The cubehelix color of an arbitrary phase, in the RGB format with u8 samples (0..=255).
pub fn cubehelix_phase_color(phase: f32) -> [u8; 3] {
    let normalized_phase = phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
    cubehelix_colormap(normalized_phase).map(|x| (x * 255.0) as u8)
}
