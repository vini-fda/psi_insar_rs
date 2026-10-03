//! Terrain Correction (Geocoding)
//!
//! Resamples an image in radar geometry (`[azimuth, range]`) onto a regular latitude/longitude
//! grid, like SNAP's Range-Doppler Terrain Correction, which is the last step of NASA's
//! Sentinel-1 interferogram recipe.
//!
//! The radar coordinates of the ground are given by tie points: DEM points (latitude,
//! longitude) with their radar coordinates, which are what the DEM-assisted coregistration
//! computes (see
//! [`EnhancedDelaunayWarpFunction`](crate::perp_baseline::EnhancedDelaunayWarpFunction)).
//! Since the tie points come from the DEM, the mapping includes the terrain, unlike a mapping
//! to an ellipsoid. [`GeocodingFunction`] triangulates them in latitude/longitude and
//! interpolates their radar coordinates at each output pixel (natural neighbor interpolation),
//! where the image is then sampled bilinearly.
//!
//! The mapping from the ground to radar coordinates is a function (each ground point is seen at
//! one azimuth time and range), so it is well defined in layover areas too, where several
//! ground points share radar coordinates: they all get the same, mixed, radar value.

use std::ops::{Add, Mul};

use ndarray::{Array2, ArrayView2, Axis};
use num_traits::Zero;
use rayon::prelude::*;
use spade::{DelaunayTriangulation, HasPosition, HierarchyHintGenerator, Point2, Triangulation};

/// Mean Earth radius (IUGG), used to convert pixel spacings from meters to degrees.
const MEAN_EARTH_RADIUS: f64 = 6_371_008.8;

/// A ground point with its coordinates in the radar image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TiePoint {
    /// Latitude, in degrees.
    pub lat: f64,
    /// Longitude, in degrees.
    pub lon: f64,
    /// Azimuth (line) coordinate in the radar image.
    pub azimuth: f64,
    /// Range (sample) coordinate in the radar image.
    pub range: f64,
}

impl HasPosition for TiePoint {
    type Scalar = f64;

    fn position(&self) -> Point2<f64> {
        Point2::new(self.lon, self.lat)
    }
}

/// A regular, north-up latitude/longitude grid: row 0 is the northernmost and column 0 the
/// westernmost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLonGrid {
    /// Latitude of the northern edge of the grid, in degrees.
    pub north: f64,
    /// Longitude of the western edge of the grid, in degrees.
    pub west: f64,
    /// Height of a pixel, in degrees of latitude.
    pub lat_spacing: f64,
    /// Width of a pixel, in degrees of longitude.
    pub lon_spacing: f64,
    pub rows: usize,
    pub cols: usize,
}

impl LatLonGrid {
    /// The grid covering `bounds` (`[min lat, max lat, min lon, max lon]`, in degrees) with
    /// pixels of about `spacing` meters on each side at the center of the bounds (on a sphere
    /// with the mean Earth radius).
    pub fn covering(bounds: [f64; 4], spacing: f64) -> Self {
        let [min_lat, max_lat, min_lon, max_lon] = bounds;
        let lat_spacing = (spacing / MEAN_EARTH_RADIUS).to_degrees();
        let lon_spacing = lat_spacing / ((min_lat + max_lat) / 2.0).to_radians().cos();
        LatLonGrid {
            north: max_lat,
            west: min_lon,
            lat_spacing,
            lon_spacing,
            rows: ((max_lat - min_lat) / lat_spacing).ceil().max(1.0) as usize,
            cols: ((max_lon - min_lon) / lon_spacing).ceil().max(1.0) as usize,
        }
    }

    /// The `[lat, lon]` of the center of pixel (`row`, `col`).
    pub fn pixel_center(&self, row: usize, col: usize) -> [f64; 2] {
        [
            self.north - (row as f64 + 0.5) * self.lat_spacing,
            self.west + (col as f64 + 0.5) * self.lon_spacing,
        ]
    }

    /// The `[lat, lon]` of the outer corners of the grid: upper left, lower left, lower right,
    /// upper right and upper left again, e.g. to draw its extent as a closed line.
    pub fn closed_corners_lat_lon(&self) -> [[f64; 2]; 5] {
        let south = self.north - self.rows as f64 * self.lat_spacing;
        let east = self.west + self.cols as f64 * self.lon_spacing;
        [
            [self.north, self.west],
            [south, self.west],
            [south, east],
            [self.north, east],
            [self.north, self.west],
        ]
    }
}

type GeoTriangulation = DelaunayTriangulation<TiePoint, (), (), (), HierarchyHintGenerator<f64>>;

/// Maps latitude/longitude to radar image coordinates, by natural neighbor interpolation of
/// the radar coordinates of tie points.
pub struct GeocodingFunction {
    triangulation: GeoTriangulation,
}

impl GeocodingFunction {
    /// Triangulates the `tie_points`. Points with non-finite coordinates are skipped, and of
    /// several points at the same latitude/longitude, only one is kept.
    pub fn new(tie_points: impl IntoIterator<Item = TiePoint>) -> Self {
        let tie_points: Vec<TiePoint> = tie_points
            .into_iter()
            .filter(|p| {
                [p.lat, p.lon, p.azimuth, p.range]
                    .iter()
                    .all(|x| x.is_finite())
            })
            .collect();
        let triangulation =
            GeoTriangulation::bulk_load(tie_points).expect("Finite tie points are always valid");
        GeocodingFunction { triangulation }
    }

    /// Number of (distinct) tie points.
    pub fn len(&self) -> usize {
        self.triangulation.num_vertices()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `[min lat, max lat, min lon, max lon]` of the tie points, or `None` without tie
    /// points.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        self.triangulation.vertices().fold(None, |bounds, vertex| {
            let TiePoint { lat, lon, .. } = *vertex.data();
            let [min_lat, max_lat, min_lon, max_lon] = bounds.unwrap_or([lat, lat, lon, lon]);
            Some([
                min_lat.min(lat),
                max_lat.max(lat),
                min_lon.min(lon),
                max_lon.max(lon),
            ])
        })
    }

    /// The `[azimuth, range]` image coordinates of (`lat`, `lon`), or `None` outside the convex
    /// hull of the tie points.
    pub fn radar_coords(&self, lat: f64, lon: f64) -> Option<[f64; 2]> {
        Self::interpolate(&self.triangulation.natural_neighbor(), lat, lon)
    }

    fn interpolate(
        nn: &spade::NaturalNeighbor<'_, GeoTriangulation>,
        lat: f64,
        lon: f64,
    ) -> Option<[f64; 2]> {
        let point = Point2::new(lon, lat);
        let azimuth = nn.interpolate(|v| v.data().azimuth, point)?;
        let range = nn.interpolate(|v| v.data().range, point)?;
        Some([azimuth, range])
    }

    /// `image` (`[azimuth, range]`, in the radar coordinates of the tie points) resampled on
    /// `grid`. Each output pixel is the bilinear interpolation of `image` at the radar
    /// coordinates of its center, or zero if they are outside the tie points or the image.
    pub fn geocode<T>(&self, image: ArrayView2<'_, T>, grid: &LatLonGrid) -> Array2<T>
    where
        T: Copy + Zero + Add<Output = T> + Mul<f32, Output = T> + Send + Sync,
    {
        let mut output = Array2::zeros((grid.rows, grid.cols));
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each_init(
                || self.triangulation.natural_neighbor(),
                |nn, (row, mut output_row)| {
                    for (col, value) in output_row.iter_mut().enumerate() {
                        let [lat, lon] = grid.pixel_center(row, col);
                        if let Some([azimuth, range]) = Self::interpolate(nn, lat, lon)
                            && let Some(sample) = bilinear(image, azimuth, range)
                        {
                            *value = sample;
                        }
                    }
                },
            );
        output
    }
}

/// The bilinear interpolation of `image` at (`azimuth`, `range`), or `None` outside the image.
fn bilinear<T>(image: ArrayView2<'_, T>, azimuth: f64, range: f64) -> Option<T>
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    let (rows, cols) = image.dim();
    let inside = |coord: f64, len: usize| len > 0 && (0.0..=(len - 1) as f64).contains(&coord);
    if !inside(azimuth, rows) || !inside(range, cols) {
        return None;
    }
    // The top-left neighbor, kept one pixel before the last row/column so that the bottom-right
    // neighbor exists (with a zero weight on the last row/column).
    let i = (azimuth.floor() as usize).min(rows.saturating_sub(2));
    let j = (range.floor() as usize).min(cols.saturating_sub(2));
    let (i1, j1) = ((i + 1).min(rows - 1), (j + 1).min(cols - 1));
    let (u, v) = ((azimuth - i as f64) as f32, (range - j as f64) as f32);
    Some(
        image[[i, j]] * ((1.0 - u) * (1.0 - v))
            + image[[i, j1]] * ((1.0 - u) * v)
            + image[[i1, j]] * (u * (1.0 - v))
            + image[[i1, j1]] * (u * v),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Tie points on a 0.01° grid over [10°, 10.5°] × [20°, 20.6°], with radar coordinates that
    /// are linear in latitude/longitude: azimuth = 100 · (lat - 10), range = 50 · (lon - 20),
    /// i.e. a 50 × 30 radar image.
    fn linear_tie_points() -> Vec<TiePoint> {
        let mut points = Vec::new();
        for i in 0..=50 {
            for j in 0..=60 {
                let (lat, lon) = (10.0 + 0.01 * i as f64, 20.0 + 0.01 * j as f64);
                points.push(TiePoint {
                    lat,
                    lon,
                    azimuth: 100.0 * (lat - 10.0),
                    range: 50.0 * (lon - 20.0),
                });
            }
        }
        points
    }

    #[test]
    fn grid_covers_bounds_with_square_pixels() {
        let grid = LatLonGrid::covering([32.6, 33.0, 130.5, 131.0], 30.0);
        // 30 m is 2.7e-4° of latitude, and 3.2e-4° of longitude at 32.8° N.
        assert_relative_eq!(grid.lat_spacing, 2.698e-4, epsilon = 1e-7);
        assert_relative_eq!(
            grid.lon_spacing,
            grid.lat_spacing / 32.8f64.to_radians().cos()
        );
        assert_eq!(grid.rows, (0.4 / grid.lat_spacing).ceil() as usize);
        assert_eq!(grid.cols, (0.5 / grid.lon_spacing).ceil() as usize);

        let [lat, lon] = grid.pixel_center(0, 0);
        assert_relative_eq!(lat, 33.0 - grid.lat_spacing / 2.0);
        assert_relative_eq!(lon, 130.5 + grid.lon_spacing / 2.0);
        let corners = grid.closed_corners_lat_lon();
        assert_eq!(corners[0], corners[4]);
        assert!(corners[2][0] <= 32.6 && corners[2][1] >= 131.0);
    }

    #[test]
    fn interpolates_radar_coords() {
        let function = GeocodingFunction::new(linear_tie_points());
        assert_eq!(function.len(), 51 * 61);
        assert_eq!(function.bounds(), Some([10.0, 10.5, 20.0, 20.6]));

        let [azimuth, range] = function.radar_coords(10.123, 20.456).unwrap();
        assert_relative_eq!(azimuth, 12.3, epsilon = 1e-9);
        assert_relative_eq!(range, 22.8, epsilon = 1e-9);
        assert_eq!(function.radar_coords(9.9, 20.3), None);
    }

    #[test]
    fn geocodes_onto_grid() {
        let function = GeocodingFunction::new(linear_tie_points());
        // A radar image whose value is linear in its coordinates: 1000 · azimuth + range.
        let image = Array2::from_shape_fn((51, 31), |(i, j)| (1000 * i + j) as f32);
        // Wider than the tie points to the north and east, where the output must be zero.
        let grid = LatLonGrid {
            north: 10.6,
            west: 20.0,
            lat_spacing: 0.01,
            lon_spacing: 0.01,
            rows: 60,
            cols: 70,
        };
        let geocoded = function.geocode(image.view(), &grid);

        for ((row, col), &value) in geocoded.indexed_iter() {
            let [lat, lon] = grid.pixel_center(row, col);
            if lat < 10.5 && lon < 20.6 {
                let expected = 1000.0 * 100.0 * (lat - 10.0) + 50.0 * (lon - 20.0);
                assert_relative_eq!(value as f64, expected, epsilon = 0.05);
            } else {
                assert_eq!(value, 0.0, "({row}, {col}) at {lat}, {lon}");
            }
        }
    }

    #[test]
    fn skips_non_finite_tie_points() {
        let mut points = linear_tie_points();
        points.push(TiePoint {
            lat: 10.25,
            lon: f64::NAN,
            azimuth: 0.0,
            range: 0.0,
        });
        assert_eq!(GeocodingFunction::new(points).len(), 51 * 61);
        assert!(GeocodingFunction::new([]).is_empty());
        assert_eq!(GeocodingFunction::new([]).bounds(), None);
    }

    #[test]
    fn bilinear_interpolation() {
        let image = ndarray::array![[0.0f32, 1.0], [2.0, 3.0]];
        assert_eq!(bilinear(image.view(), 0.5, 0.5), Some(1.5));
        assert_eq!(bilinear(image.view(), 1.0, 1.0), Some(3.0));
        assert_eq!(bilinear(image.view(), 0.0, 1.0), Some(1.0));
        assert_eq!(bilinear(image.view(), 1.01, 0.0), None);
        assert_eq!(bilinear(image.view(), 0.0, -0.01), None);
    }
}
