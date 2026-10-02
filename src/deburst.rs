//! TOPS Deburst
//!
//! Joins the bursts of a Sentinel-1 IW subswath into a single image on a regular azimuth time
//! grid, like SNAP's S-1 TOPS Deburst operator [\[2\]].
//!
//! Each burst of an IW SLC is stored as a separate block of `linesPerBurst` lines, but
//! consecutive bursts image overlapping areas: the last valid lines of a burst and the first
//! valid lines of the next one have (almost) the same zero-Doppler times. The first and last
//! lines and samples of each burst are also invalid (zero-filled), which the annotation marks
//! with `-1` in the per-line `firstValidSample`/`lastValidSample` arrays of the `burstList`.
//!
//! [`deburst`] places every burst on the output grid by its first-line azimuth time, keeps
//! only its valid lines and samples, and in each overlap switches from one burst to the next
//! at the middle of the overlap of their valid lines. SNAP switches at the middle of the
//! overlap of the whole bursts instead, which is about one line away, since the invalid lines
//! at the start and end of a burst are about as many. Lines are placed at the nearest output
//! line, as in SNAP, so bursts whose times are not an exact number of lines apart are shifted
//! by less than half a line.
//!
//! The input arrays may be any per-burst product in the geometry of the bursts described by
//! the [`BurstGeometry`]s, e.g. the SLC bursts themselves, or the interferogram and coherence of
//! each reference burst after coregistration.
//!
//! # Sources
//!
//! 1. Sentinel-1 Product Specification (S1-RS-MDA-52-7441), `swathTiming` annotation.
//! 2. SNAP microwave toolbox, `TOPSARDeburstOp` (burst selection in
//!    `getLineIndicesInSourceProduct`):
//!    <https://github.com/senbox-org/microwave-toolbox/blob/master/sar-op-sentinel1/src/main/java/eu/esa/sar/sentinel1/gpf/TOPSARDeburstOp.java>
//!
//! [\[2\]]: https://github.com/senbox-org/microwave-toolbox/blob/master/sar-op-sentinel1/src/main/java/eu/esa/sar/sentinel1/gpf/TOPSARDeburstOp.java

use std::ops::RangeInclusive;

use chrono::{DateTime, TimeDelta, Utc};
use ndarray::{Array2, ArrayView2, Axis, s};
use num_traits::Zero;
use rayon::prelude::*;
use thiserror::Error;

use crate::metadata::annotation_xml::SlcProductAnnotation;

#[derive(Debug, Error, PartialEq)]
pub enum DeburstError {
    #[error("No bursts to deburst")]
    NoBursts,
    #[error("Burst {burst} has {array} lines, but its geometry describes {geometry}")]
    LineCountMismatch {
        burst: usize,
        array: usize,
        geometry: usize,
    },
    #[error("Burst {burst} has {samples} samples, but burst 0 has {expected}")]
    SampleCountMismatch {
        burst: usize,
        samples: usize,
        expected: usize,
    },
    #[error("Burst {0} has no valid lines")]
    NoValidLines(usize),
    #[error("Burst {0} does not start after the previous burst")]
    NotInAzimuthOrder(usize),
    #[error("Invalid burst annotation: {0}")]
    InvalidAnnotation(String),
}

/// Timing and valid region of one burst.
#[derive(Debug, Clone, PartialEq)]
pub struct BurstGeometry {
    /// Zero-Doppler azimuth time of the burst's first line.
    pub azimuth_time: DateTime<Utc>,
    /// For each line of the burst, its first valid sample, or `-1` if the line is invalid.
    pub first_valid_sample: Vec<i32>,
    /// For each line of the burst, its last valid sample (inclusive), or `-1` if the line is
    /// invalid.
    pub last_valid_sample: Vec<i32>,
}

impl BurstGeometry {
    /// The geometry of burst `burst_index` of an SLC product, from its `swathTiming` annotation.
    pub fn from_annotation(
        metadata: &SlcProductAnnotation,
        burst_index: usize,
    ) -> Result<Self, DeburstError> {
        let swath_timing = &metadata.swath_timing;
        let burst = swath_timing
            .burst_list
            .bursts
            .get(burst_index)
            .ok_or_else(|| {
                DeburstError::InvalidAnnotation(format!("No burst {burst_index} in the burstList"))
            })?;
        let lines = swath_timing.lines_per_burst;
        let parse = |name: &str, text: &Option<String>| {
            let values = text
                .as_deref()
                .unwrap_or_default()
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|err| {
                    DeburstError::InvalidAnnotation(format!(
                        "Burst {burst_index} {name} is not a list of integers: {err}"
                    ))
                })?;
            if values.len() != lines {
                return Err(DeburstError::InvalidAnnotation(format!(
                    "Burst {burst_index} {name} has {} values, expected one per line ({lines})",
                    values.len()
                )));
            }
            Ok(values)
        };
        Ok(BurstGeometry {
            azimuth_time: burst.azimuth_time,
            first_valid_sample: parse("firstValidSample", &burst.first_valid_sample.text)?,
            last_valid_sample: parse("lastValidSample", &burst.last_valid_sample.text)?,
        })
    }

    /// Number of lines of the burst.
    pub fn lines(&self) -> usize {
        self.first_valid_sample.len()
    }

    /// The range from the first to the last valid line, or `None` if no line is valid.
    pub fn valid_lines(&self) -> Option<RangeInclusive<usize>> {
        let is_valid = |line: &usize| self.valid_samples(*line).is_some();
        let first = (0..self.lines()).find(is_valid)?;
        let last = (0..self.lines()).rfind(is_valid)?;
        Some(first..=last)
    }

    /// The valid samples of `line`, or `None` if the line is invalid.
    pub fn valid_samples(&self, line: usize) -> Option<RangeInclusive<usize>> {
        let first = usize::try_from(*self.first_valid_sample.get(line)?).ok()?;
        let last = usize::try_from(*self.last_valid_sample.get(line)?).ok()?;
        (first <= last).then_some(first..=last)
    }
}

/// A debursted image.
#[derive(Debug, Clone, PartialEq)]
pub struct Debursted<T> {
    /// The image, `[azimuth line, range sample]`. Invalid samples are zero.
    pub data: Array2<T>,
    /// Zero-Doppler azimuth time of the first line. Line `i` is at
    /// `first_line_time + i * azimuth_time_interval`.
    pub first_line_time: DateTime<Utc>,
}

/// Joins consecutive bursts of a subswath (in azimuth order) into a single image.
///
/// `bursts` pairs each burst's array (`[line, sample]`, with the burst's `linesPerBurst` lines)
/// with its geometry, and `azimuth_time_interval` is the line spacing in seconds (the
/// `azimuthTimeInterval` annotation). The output starts at the first valid line of the first
/// burst and ends at the last valid line of the last burst. In the overlap between two bursts,
/// the lines before the middle of the overlap come from the first burst and the others from
/// the second. Lines in a gap between two bursts that do not overlap, and samples outside a
/// line's valid samples, are zero.
pub fn deburst<T>(
    bursts: &[(ArrayView2<'_, T>, &BurstGeometry)],
    azimuth_time_interval: f64,
) -> Result<Debursted<T>, DeburstError>
where
    T: Copy + Zero + Send + Sync,
{
    let (first_array, first_geometry) = bursts.first().ok_or(DeburstError::NoBursts)?;
    let samples = first_array.ncols();
    let t0 = first_geometry.azimuth_time;
    // Burst start times and valid line ranges, with times in seconds since t0.
    let mut timings = Vec::with_capacity(bursts.len());
    for (k, (array, geometry)) in bursts.iter().enumerate() {
        if array.nrows() != geometry.lines() {
            return Err(DeburstError::LineCountMismatch {
                burst: k,
                array: array.nrows(),
                geometry: geometry.lines(),
            });
        }
        if array.ncols() != samples {
            return Err(DeburstError::SampleCountMismatch {
                burst: k,
                samples: array.ncols(),
                expected: samples,
            });
        }
        let valid_lines = geometry
            .valid_lines()
            .ok_or(DeburstError::NoValidLines(k))?;
        let start = (geometry.azimuth_time - t0).as_seconds_f64();
        if let Some(&(previous_start, _)) = timings.last()
            && start <= previous_start
        {
            return Err(DeburstError::NotInAzimuthOrder(k));
        }
        timings.push((start, valid_lines));
    }

    let line_time = |(start, _): &(f64, RangeInclusive<usize>), line: usize| {
        start + line as f64 * azimuth_time_interval
    };
    let first_time = line_time(&timings[0], *timings[0].1.start());
    let last = timings.last().unwrap();
    let last_time = line_time(last, *last.1.end());
    let lines = ((last_time - first_time) / azimuth_time_interval).round() as usize + 1;
    // Burst k is used up to the middle of its overlap with burst k + 1.
    let switch_times: Vec<f64> = timings
        .windows(2)
        .map(|pair| {
            let end = line_time(&pair[0], *pair[0].1.end());
            let next_start = line_time(&pair[1], *pair[1].1.start());
            (end + next_start) / 2.0
        })
        .collect();

    let mut data = Array2::zeros((lines, samples));
    data.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let time = first_time + i as f64 * azimuth_time_interval;
            let k = switch_times.partition_point(|&switch_time| switch_time <= time);
            let (start, valid_lines) = &timings[k];
            let line = ((time - start) / azimuth_time_interval).round();
            if line < 0.0 || !valid_lines.contains(&(line as usize)) {
                return;
            }
            let line = line as usize;
            let (array, geometry) = &bursts[k];
            if let Some(valid_samples) = geometry.valid_samples(line) {
                let end = (*valid_samples.end()).min(samples - 1);
                let valid_samples = *valid_samples.start()..=end;
                row.slice_mut(s![valid_samples.clone()])
                    .assign(&array.slice(s![line, valid_samples]));
            }
        });

    let first_line_time = t0 + TimeDelta::nanoseconds((first_time * 1e9).round() as i64);
    Ok(Debursted {
        data,
        first_line_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.002;

    fn utc(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().to_utc()
    }

    /// A burst geometry with `lines` lines starting `start_lines` lines after t = 0, whose lines
    /// in `valid_lines` have the valid samples `valid_samples`.
    fn geometry(
        start_lines: f64,
        lines: usize,
        valid_lines: RangeInclusive<usize>,
        valid_samples: RangeInclusive<i32>,
    ) -> BurstGeometry {
        let per_line = |value: i32| {
            (0..lines)
                .map(|line| {
                    if valid_lines.contains(&line) {
                        value
                    } else {
                        -1
                    }
                })
                .collect()
        };
        BurstGeometry {
            azimuth_time: utc("2016-04-08T09:14:00Z")
                + TimeDelta::nanoseconds((start_lines * DT * 1e9).round() as i64),
            first_valid_sample: per_line(*valid_samples.start()),
            last_valid_sample: per_line(*valid_samples.end()),
        }
    }

    /// A burst whose samples hold `100 * burst + line`, so that the output shows where each
    /// value came from.
    fn burst_array(burst: usize, lines: usize, samples: usize) -> Array2<f32> {
        Array2::from_shape_fn((lines, samples), |(line, _)| (100 * burst + line) as f32)
    }

    #[test]
    fn parses_annotation_geometry() {
        let metadata: SlcProductAnnotation =
            quick_xml::de::from_str(include_str!("metadata/test_data/annotation_example.xml"))
                .unwrap();
        let geometry = BurstGeometry::from_annotation(&metadata, 0).unwrap();
        assert_eq!(geometry.azimuth_time, utc("2015-10-22T12:25:46.721151Z"));
        assert_eq!(geometry.lines(), 1507);
        assert_eq!(geometry.valid_lines(), Some(27..=1481));
        assert_eq!(geometry.valid_samples(0), None);
        assert_eq!(geometry.valid_samples(27), Some(29..=23722));
        assert!(matches!(
            BurstGeometry::from_annotation(&metadata, 1),
            Err(DeburstError::InvalidAnnotation(_))
        ));
    }

    #[test]
    fn single_burst_keeps_valid_region() {
        let geometry = geometry(0.0, 6, 1..=4, 1..=2);
        let array = burst_array(0, 6, 4);
        let debursted = deburst(&[(array.view(), &geometry)], DT).unwrap();
        // Valid lines 1..=4, valid samples 1..=2, zero elsewhere.
        assert_eq!(debursted.data.dim(), (4, 4));
        for (i, row) in debursted.data.outer_iter().enumerate() {
            let line = (i + 1) as f32;
            assert_eq!(row.to_vec(), vec![0.0, line, line, 0.0]);
        }
        assert_eq!(
            debursted.first_line_time,
            geometry.azimuth_time + TimeDelta::milliseconds(2)
        );
    }

    #[test]
    fn switches_bursts_in_the_middle_of_the_overlap() {
        // Burst 0 is valid at times (in lines) 1..=8, burst 1 starts at 6 and is valid at
        // 7..=14: they overlap at 7..=8, and the switch is at 7.5.
        let geometries = [
            geometry(0.0, 10, 1..=8, 0..=2),
            geometry(6.0, 10, 1..=8, 0..=2),
        ];
        let arrays = [burst_array(0, 10, 3), burst_array(1, 10, 3)];
        let bursts: Vec<_> = arrays.iter().map(|a| a.view()).zip(&geometries).collect();
        let debursted = deburst(&bursts, DT).unwrap();

        let column: Vec<f32> = debursted.data.column(0).to_vec();
        // Times 1..=7 from burst 0 (lines 1..=7), times 8..=14 from burst 1 (lines 2..=8).
        let expected = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
        ];
        assert_eq!(column, expected);
        assert_eq!(
            debursted.first_line_time,
            geometries[0].azimuth_time + TimeDelta::milliseconds(2)
        );
    }

    #[test]
    fn rounds_burst_start_to_the_nearest_line() {
        // Burst 1 starts 0.3 lines after a line of the output grid.
        let geometries = [
            geometry(0.0, 6, 0..=5, 0..=0),
            geometry(4.3, 6, 0..=5, 0..=0),
        ];
        let arrays = [burst_array(0, 6, 1), burst_array(1, 6, 1)];
        let bursts: Vec<_> = arrays.iter().map(|a| a.view()).zip(&geometries).collect();
        let debursted = deburst(&bursts, DT).unwrap();
        // Switch at (5 + 4.3) / 2 = 4.65; output line i at time i, burst 1 line round(i - 4.3).
        let column: Vec<f32> = debursted.data.column(0).to_vec();
        assert_eq!(
            column,
            [0.0, 1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        );
    }

    #[test]
    fn gap_between_bursts_is_zero() {
        let geometries = [
            geometry(0.0, 4, 0..=3, 0..=0),
            geometry(6.0, 4, 0..=3, 0..=0),
        ];
        let arrays = [burst_array(0, 4, 1).mapv(|v| v + 1.0), burst_array(1, 4, 1)];
        let bursts: Vec<_> = arrays.iter().map(|a| a.view()).zip(&geometries).collect();
        let debursted = deburst(&bursts, DT).unwrap();
        let column: Vec<f32> = debursted.data.column(0).to_vec();
        assert_eq!(
            column,
            [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 100.0, 101.0, 102.0, 103.0]
        );
    }

    #[test]
    fn rejects_inconsistent_bursts() {
        let g = geometry(0.0, 4, 0..=3, 0..=0);
        let array = burst_array(0, 4, 2);
        assert_eq!(deburst::<f32>(&[], DT), Err(DeburstError::NoBursts));

        let short = burst_array(0, 3, 2);
        assert!(matches!(
            deburst(&[(short.view(), &g)], DT),
            Err(DeburstError::LineCountMismatch { burst: 0, .. })
        ));

        let narrow = burst_array(1, 4, 1);
        let later = geometry(2.0, 4, 0..=3, 0..=0);
        assert!(matches!(
            deburst(&[(array.view(), &g), (narrow.view(), &later)], DT),
            Err(DeburstError::SampleCountMismatch { burst: 1, .. })
        ));

        assert_eq!(
            deburst(&[(array.view(), &later), (array.view(), &g)], DT),
            Err(DeburstError::NotInAzimuthOrder(1))
        );

        let invalid = BurstGeometry {
            first_valid_sample: vec![-1; 4],
            last_valid_sample: vec![-1; 4],
            ..g.clone()
        };
        assert_eq!(
            deburst(&[(array.view(), &invalid)], DT),
            Err(DeburstError::NoValidLines(0))
        );
    }
}
