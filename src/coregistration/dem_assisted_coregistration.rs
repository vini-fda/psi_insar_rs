//! Sentinel-1 DEM-Based Coregistration
//!
//! Implementation for coregistering Sentinel-1 SLC burst data
//! using orbit information and Digital Elevation Models (DEM).
//!
//! This module provides functionality specifically tailored for Sentinel-1
//! TOPS (Terrain Observation with Progressive Scans) imagery where data is
//! organized in bursts.
//! # Coregistering InSAR Products
//!
//! The sub-pixel coregistration of SAR images is a strict requirement
//! and critical component of any interferometric processing chain. It is
//! an essential step for the accurate determination of phase difference,
//! and applications such as DEM map generation, interferometric
//! deformation analysis, etc.
//!
//! This module will co-register
//! one or more secondary images with respect to a reference image. The
//! co-registration procedure is completely automatic. Apart from defining
//! the processing parameters, no additional input nor intervention from the
//! user is required. For example, the distribution of correlation (optimization)
//! windows are done in automatic manner for both reference and secondary image.
//! Also, the refinement of the coregistration offsets is done in a fully
//! automatic way, including downloading and interpolation of the a-priori
//! digital-elevation-model.
//!
//! ## Brief Implementation Details
//!
//! The implementation of the coregistration procedure is based on the
//! cross-correlation technique. Since this technique for an optimal
//! alignment tend to be slow for very large search windows, the procedure
//! is usually separated in two main steps: *coarse* and *fine*
//! coregistration. In the coarse coregistration, the offsets are
//! approximated either by using the satellite orbits and timing as a
//! reference, and/or by defining an approximate common points in
//! reference/secondary images and performing correlation matching with large
//! windows. The subsequent fine coregistration applies automation
//! correlation technique to obtain sub-pixel alignment accuracy. After the
//! coregistration offsets are computed, the estimation of the
//! coregistration polynomial (CPM) and interferometric resampling of secondary
//! images to the reference geometry is performed.
//!
//! ## Overview of Coregistration Processing Chain
//!
//! The interferometric coregistration is performed by create stack, coarse
//! fine coregistration and resampling.
//!
//! ```text
//! Reference Image  Secondary Image(s)
//!       |               |
//!       v               v
//!    Create Stack (initial coarse alignment)
//!           |
//!           v
//!    Cross Correlation (compute offsets)
//!           |
//!           v
//!    Warp (resample to reference geometry)
//!           |
//!           v
//!    Coregistered Stack
//! ```
//!
//! ## Input Images and Data Support
//!
//! Input SAR images may be fully ("full frame") or only partially
//! overlapping ("subset"), they have to be from acquisitions taken at
//! different times using compatible, in the interferometric sense,
//! sensors, and input images must belong to the same type (i.e., they must
//! be complex).
//!
//! While in principle the implementation of the InSAR coregistration is
//! flexible enough to allow processing of real (detected) products, for
//! now only complex (single-look-complex) data is supported.
//!
//! ## Create Stack
//!
//! The Create Stack operation collocates the reference and secondary
//! images based into a single reference geometry. Basically the
//! secondary image data is subset into geometry of the reference image. With
//! performing this operation the reference and secondary images share the same
//! geo-positioning information, and have the similar dimensions. For
//! overlap and geometry calculation either orbital data, or annotated
//! tie-point-grids (i.e., ground-control-points) can be used. In other
//! words the coarse coregistration is performed using orbital information
//! or annotated GCPs. The method based on orbits is recommended for all
//! platforms, since especially in case of old sensors (ERS1/2) annotated
//! GCPs prove not to be reliable through-out the whole mission lifetime.
//!
//! ## Cross Correlation
//!
//! The Cross Correlation operation creates an alignment
//! between reference and secondary images by matching automatically distributed
//! correlation optimization windows to their corresponding secondary windows.
//! There are two steps: coarse and fine registration. The offsets between
//! reference and secondary are computed by maximizing the cross-correlation
//! between reference and secondary images on a series of imagettes defined across
//! the images. First on coarse level, with large windows and lower
//! oversampling factors, later on fine level, with smaller windows and
//! higher oversampling factors.
//!
//! ## Warp
//!
//! With the reference-secondary offsets computed, a coregistration polynomial
//! (CPM) is estimated by the Warp operation, which resamples pixels in
//! the secondary image into pixels in the reference image.
//!
//! This resampling is performed in two-steps:
//! 1. Reconstruction of the continuous signal from its sampled version by convolution with an
//!    interpolation kernel
//! 2. Sampling of the constructed signal at the new sampling locations.
//!

use thiserror::Error;

use crate::dem_gdal::DEMGdal;
use crate::sentinel::Sentinel1SlcBurst;

/// Error types specific to Sentinel-1 DEM coregistration
#[derive(Error, Debug)]
pub enum S1CoregistrationError {
    /// Invalid product format
    #[error("Invalid product format: {0}")]
    InvalidFormat(String),

    /// Input product validation error
    #[error("Input validation error: {0}")]
    ValidationError(String),

    /// Burst error
    #[error("Burst error: {0}")]
    BurstError(String),

    /// Error during computation
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// DEM error
    #[error("DEM error: {0}")]
    DEMError(String),

    /// Input/output error
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
}

/// Result type for S1 coregistration operations
pub type Result<T> = std::result::Result<T, S1CoregistrationError>;

/// Resampling methods for interpolation between grids
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingMethod {
    /// Nearest neighbor interpolation (fast but low quality)
    NearestNeighbor,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
    /// Bisinc 5-point interpolation (high quality)
    Bisinc5Point,
    /// Bisinc 11-point interpolation (highest quality)
    Bisinc11Point,
}

/// Configures the DEM-based coregistration for Sentinel-1 SLC products
#[derive(Debug)]
pub struct S1DEMCoregistrationConfig {
    /// No data value for external DEM
    pub external_dem_no_data_value: f64,
    /// Resampling method for DEM
    pub dem_resampling_method: ResamplingMethod,
    /// Resampling method for image interpolation
    pub image_resampling_method: ResamplingMethod,
    /// Whether to mask out areas without elevation data
    pub mask_out_area_without_elevation: bool,
    /// Whether to output range and azimuth offset bands
    pub output_range_azimuth_offset: bool,
    /// Whether to output deramp/demod phase
    pub output_deramp_demod_phase: bool,
    /// Whether to disable reramp
    pub disable_reramp: bool,
    /// Whether to output DEM height band
    pub output_dem: bool,
}

impl Default for S1DEMCoregistrationConfig {
    fn default() -> Self {
        Self {
            external_dem_no_data_value: 0.0,
            dem_resampling_method: ResamplingMethod::Bicubic,
            image_resampling_method: ResamplingMethod::Bisinc5Point,
            mask_out_area_without_elevation: true,
            output_range_azimuth_offset: false,
            output_deramp_demod_phase: false,
            disable_reramp: false,
            output_dem: false,
        }
    }
}

/// Coregistered Stack of SLC bursts
pub struct CoregisteredSLCStack {}

pub struct DEMAssistedCoregistration {
    config: S1DEMCoregistrationConfig,
    external_dem: DEMGdal,
}

impl DEMAssistedCoregistration {
    pub fn new(config: S1DEMCoregistrationConfig, external_dem: DEMGdal) -> Self {
        Self {
            config,
            external_dem,
        }
    }

    pub fn coregister_stack(
        &self,
        reference: &Sentinel1SlcBurst,
        secondaries: &[Sentinel1SlcBurst],
    ) -> Result<CoregisteredSLCStack> {
        // 1. Create Stack of SLC bursts
        // 2. Apply orbit correction (precise orbit ephemerides)
        // 3. Coarse coregistration
        // 4. Fine coregistration
        // 5. Resample to common grid
        // 6. Result: Coregistered SLC stack
        unimplemented!()
    }
}
