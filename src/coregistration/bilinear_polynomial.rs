use crate::metadata::annotation_xml::Polynomial;

/// A bilinear polynomial interpolates a family of polynomials with respect to `y`.
///
/// Internally stores pairs `(y, Polynomial)` and interpolates coefficients linearly
/// between the nearest neighbors of a query `y`.
#[derive(Debug, Clone)]
pub struct BilinearPolynomial {
    /// Each entry is `(y_value, Polynomial)` where polynomials are assumed to have same degree
    pub polys: Vec<(f64, Polynomial)>,
}

impl BilinearPolynomial {
    /// Create a new bilinear polynomial from pairs `(y, Polynomial)`.
    /// The input will be sorted by `y`.
    pub fn new(mut polys: Vec<(f64, Polynomial)>) -> Self {
        polys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        BilinearPolynomial { polys }
    }

    /// Linearly interpolate polynomial coefficients at a given `y`.
    pub fn interpolate(&self, y: f64) -> Polynomial {
        if self.polys.is_empty() {
            return Polynomial {
                coefficients: vec![],
            };
        }

        // If y is outside the range, clamp to nearest endpoint
        if y <= self.polys[0].0 {
            return self.polys[0].1.clone();
        }
        if y >= self.polys.last().unwrap().0 {
            return self.polys.last().unwrap().1.clone();
        }

        // Find neighbors for interpolation
        for w in self.polys.windows(2) {
            let (y0, p0) = &w[0];
            let (y1, p1) = &w[1];
            if y >= *y0 && y <= *y1 {
                let t = (y - y0) / (y1 - y0);
                let coeffs: Vec<f64> = p0
                    .coefficients
                    .iter()
                    .zip(&p1.coefficients)
                    .map(|(a, b)| a + t * (b - a))
                    .collect();
                return Polynomial {
                    coefficients: coeffs,
                };
            }
        }

        unreachable!("Interpolation logic should have returned earlier");
    }
}
