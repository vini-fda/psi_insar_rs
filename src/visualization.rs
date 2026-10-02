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

/// Cubehelix cycle colormap
/// Based on the BEAM Color Palette Definition
pub fn cubehelix_colormap(x: f32) -> [f32; 3] {
    // Color points from the palette (normalized to [0,1])
    let colors: [[f32; 3]; 8] = [
        [110.0 / 255.0, 60.0 / 255.0, 170.0 / 255.0], // color0
        [210.0 / 255.0, 60.0 / 255.0, 160.0 / 255.0], // color1
        [1.0, 110.0 / 255.0, 70.0 / 255.0],           // color2
        [200.0 / 255.0, 200.0 / 255.0, 50.0 / 255.0], // color3
        [80.0 / 255.0, 245.0 / 255.0, 100.0 / 255.0], // color4
        [25.0 / 255.0, 200.0 / 255.0, 180.0 / 255.0], // color5
        [60.0 / 255.0, 130.0 / 255.0, 220.0 / 255.0], // color6
        [100.0 / 255.0, 70.0 / 255.0, 190.0 / 255.0], // color7
    ];

    // Wrap input to [0, 1]
    let x = x.rem_euclid(1.0);

    // Scale x to [0, 7] to match our 8 color points
    let x_scaled = x * 7.0;

    // Get the two colors to interpolate between
    let idx = x_scaled.floor() as usize;
    let t = x_scaled - idx as f32;

    // Handle the wrap-around case
    let (c1, c2) = if idx == 7 {
        (colors[7], colors[0])
    } else {
        (colors[idx], colors[idx + 1])
    };

    // Cubic interpolation
    let t2 = t * t;
    let t3 = t2 * t;

    // Cubic interpolation coefficients
    let a = -0.5 * c1[0] + 1.5 * c2[0] - 1.5 * c1[0] + 0.5 * c2[0];
    let b = c1[0] - 2.5 * c2[0] + 2.0 * c1[0] - 0.5 * c2[0];
    let c = -0.5 * c1[0] + 0.5 * c2[0];
    let d = c1[0];

    let red = a * t3 + b * t2 + c * t + d;

    let a = -0.5 * c1[1] + 1.5 * c2[1] - 1.5 * c1[1] + 0.5 * c2[1];
    let b = c1[1] - 2.5 * c2[1] + 2.0 * c1[1] - 0.5 * c2[1];
    let c = -0.5 * c1[1] + 0.5 * c2[1];
    let d = c1[1];

    let green = a * t3 + b * t2 + c * t + d;

    let a = -0.5 * c1[2] + 1.5 * c2[2] - 1.5 * c1[2] + 0.5 * c2[2];
    let b = c1[2] - 2.5 * c2[2] + 2.0 * c1[2] - 0.5 * c2[2];
    let c = -0.5 * c1[2] + 0.5 * c2[2];
    let d = c1[2];

    let blue = a * t3 + b * t2 + c * t + d;

    [red, green, blue]
}
