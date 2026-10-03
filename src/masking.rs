//! Mask Operations
//!
//! Masks are boolean images (`true` = selected pixel). This module labels their connected
//! components, uses them to clean masks up (removing small islands of selected pixels and
//! filling small holes), and builds a sea mask from a DEM.
//!
//! # Sea masks
//!
//! The Copernicus DEM flattens water bodies [\[1\]], and in the tiles used by the tests the sea
//! is exactly 0 m (observed in the Kumamoto DEMs: the Ariake Sea is 0 m, about 20% of the
//! pixels, while land is up to 1686 m, and a few coastal land pixels are below sea level, down
//! to -12 m, so the test must be `== 0`, not `<= 0`). So the pixels whose height is exactly 0
//! are the sea, apart from a few land pixels that may happen to be at 0 m, which keeping only
//! the large connected regions removes. The Copernicus DEM also has a Water Body Mask layer
//! [\[1\]], which would be more precise, but it is not part of the elevation GeoTIFFs that this
//! crate downloads. NASA's Sentinel-1 interferogram recipe [\[2\]]
//! masks the ocean the same way ("Areas that are not covered by the DEM or are located in the
//! ocean may optionally be masked out"), where SNAP's default SRTM DEM has no data.
//!
//! # Sources
//!
//! 1. Copernicus Data Space Ecosystem, Copernicus DEM collection description:
//!    <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>
//! 2. NASA Earthdata, "Create an Interferogram Using ESA's Sentinel-1 Toolbox":
//!    <https://www.earthdata.nasa.gov/learn/data-recipes/create-interferogram-using-esas-sentinel-1-toolbox>
//!
//! [\[1\]]: https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
//! [\[2\]]: https://www.earthdata.nasa.gov/learn/data-recipes/create-interferogram-using-esas-sentinel-1-toolbox

use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::{dem::DEM, geocoding::LatLonGrid};

/// Which neighbors of a pixel are connected to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// The pixels sharing an edge with it.
    Four,
    /// The pixels sharing an edge or a corner with it.
    Eight,
}

impl Connectivity {
    fn offsets(self) -> &'static [(isize, isize)] {
        match self {
            Connectivity::Four => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            Connectivity::Eight => &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
        }
    }
}

/// The connected components of the selected pixels of a mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectedComponents {
    /// The label of each pixel: 0 for unselected pixels, and `1..=count` for the components of
    /// selected pixels, in row-major order of their first pixel.
    pub labels: Array2<u32>,
    /// The number of pixels of each label (`sizes[0]` is the number of unselected pixels).
    pub sizes: Vec<usize>,
}

impl ConnectedComponents {
    /// The number of components.
    pub fn count(&self) -> usize {
        self.sizes.len() - 1
    }
}

/// Labels the connected components of the `true` pixels of `mask` (flood fill, linear in the
/// number of pixels).
///
/// # Panics
///
/// If there are more than `u32::MAX` components.
pub fn connected_components(
    mask: ArrayView2<'_, bool>,
    connectivity: Connectivity,
) -> ConnectedComponents {
    let (rows, cols) = mask.dim();
    let mut labels = Array2::<u32>::zeros((rows, cols));
    let mut sizes = vec![mask.iter().filter(|&&selected| !selected).count()];
    let mut stack = Vec::new();
    for ((i, j), &selected) in mask.indexed_iter() {
        if !selected || labels[[i, j]] != 0 {
            continue;
        }
        let label = u32::try_from(sizes.len()).expect("Too many connected components");
        labels[[i, j]] = label;
        stack.push((i, j));
        let mut size = 0;
        while let Some((row, col)) = stack.pop() {
            size += 1;
            for &(d_row, d_col) in connectivity.offsets() {
                let (Some(r), Some(c)) =
                    (row.checked_add_signed(d_row), col.checked_add_signed(d_col))
                else {
                    continue;
                };
                if r < rows && c < cols && mask[[r, c]] && labels[[r, c]] == 0 {
                    labels[[r, c]] = label;
                    stack.push((r, c));
                }
            }
        }
        sizes.push(size);
    }
    ConnectedComponents { labels, sizes }
}

/// `mask` without its connected components of `true` pixels smaller than `min_size` pixels
/// (e.g. isolated valid pixels in a coherence mask).
pub fn remove_small_components(
    mask: ArrayView2<'_, bool>,
    min_size: usize,
    connectivity: Connectivity,
) -> Array2<bool> {
    let components = connected_components(mask, connectivity);
    components
        .labels
        .mapv(|label| label != 0 && components.sizes[label as usize] >= min_size)
}

/// `mask` with its holes (connected components of `false` pixels) smaller than `min_size`
/// pixels set to `true`. Holes touching the image edge are filled too.
///
/// To treat a mask and its holes consistently, use 8-connectivity for one and 4-connectivity
/// for the other, e.g. [`remove_small_components`] with [`Connectivity::Eight`] and this
/// function with [`Connectivity::Four`].
pub fn fill_small_holes(
    mask: ArrayView2<'_, bool>,
    min_size: usize,
    connectivity: Connectivity,
) -> Array2<bool> {
    let holes = mask.mapv(|selected| !selected);
    let large_holes = remove_small_components(holes.view(), min_size, connectivity);
    large_holes.mapv(|in_large_hole| !in_large_hole)
}

/// The pixels of `grid` covered by `dem` whose (bilinearly interpolated) height is exactly 0 m,
/// which for the Copernicus DEM is the sea, plus possibly a few land pixels at 0 m (see the
/// [module documentation](self)). Since the height is interpolated, a pixel is only selected if
/// the 4 DEM posts around it are all at 0 m, so the coast is kept.
pub fn zero_height_mask(dem: &DEM, grid: &LatLonGrid) -> Array2<bool> {
    let mut mask = Array2::from_elem((grid.rows, grid.cols), false);
    mask.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut mask_row)| {
            for (col, selected) in mask_row.iter_mut().enumerate() {
                let [lat, lon] = grid.pixel_center(row, col);
                *selected = dem.contains(lat, lon) && dem.get_height_at_lat_lon(lat, lon) == 0.0;
            }
        });
    mask
}

/// The sea pixels of `grid` according to `dem` (a Copernicus DEM): the connected regions of at
/// least `min_size` pixels of [`zero_height_mask`].
pub fn sea_mask(dem: &DEM, grid: &LatLonGrid, min_size: usize) -> Array2<bool> {
    remove_small_components(
        zero_height_mask(dem, grid).view(),
        min_size,
        Connectivity::Eight,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn mask(rows: &[&str]) -> Array2<bool> {
        let cols = rows[0].len();
        Array2::from_shape_fn((rows.len(), cols), |(i, j)| rows[i].as_bytes()[j] == b'#')
    }

    #[test]
    fn labels_components() {
        let m = mask(&["##..#", "#...#", "...#.", "##..."]);
        let four = connected_components(m.view(), Connectivity::Four);
        assert_eq!(
            four.labels,
            array![
                [1, 1, 0, 0, 2],
                [1, 0, 0, 0, 2],
                [0, 0, 0, 3, 0],
                [4, 4, 0, 0, 0]
            ]
        );
        assert_eq!(four.sizes, [12, 3, 2, 1, 2]);
        assert_eq!(four.count(), 4);

        // With 8-connectivity, the diagonal pixel joins the component on the right.
        let eight = connected_components(m.view(), Connectivity::Eight);
        assert_eq!(eight.sizes, [12, 3, 3, 2]);
        assert_eq!(eight.labels[[2, 3]], 2);
    }

    #[test]
    fn empty_and_full_masks() {
        let empty = Array2::from_elem((3, 4), false);
        let components = connected_components(empty.view(), Connectivity::Four);
        assert_eq!(components.count(), 0);
        assert_eq!(components.sizes, [12]);

        let full = Array2::from_elem((3, 4), true);
        let components = connected_components(full.view(), Connectivity::Four);
        assert_eq!(components.sizes, [0, 12]);
    }

    #[test]
    fn removes_small_islands() {
        let m = mask(&["###..", "###.#", "###..", ".....", "#...."]);
        let cleaned = remove_small_components(m.view(), 2, Connectivity::Eight);
        assert_eq!(
            cleaned,
            mask(&["###..", "###..", "###..", ".....", "....."])
        );
    }

    #[test]
    fn fills_small_holes() {
        let m = mask(&["#####", "#.###", "#####", "##..#", "##..#"]);
        // The 1-pixel hole is filled; the 4-pixel one, touching the edge, is kept.
        let filled = fill_small_holes(m.view(), 2, Connectivity::Four);
        assert_eq!(filled, mask(&["#####", "#####", "#####", "##..#", "##..#"]));
    }
}
