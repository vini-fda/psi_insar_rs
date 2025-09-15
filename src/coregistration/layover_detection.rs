use crate::{dem::DEM, satellite_orbit::radar_coords_to_pixel_coords, sentinel::Sentinel1SlcProduct};

/// Checks for layover in the radar image by comparing vertex ordering in geographic and radar coordinates.
/// Layover occurs when the relative ordering of points in geographic space is inverted in radar coordinates.
/// This function checks for layover in both azimuth and slant range dimensions.
///
/// # Arguments
/// * `burst` - The SLC burst to check for layover
/// * `dem` - The DEM to use for coordinate conversion
///
/// # Returns
/// * `bool` - True if layover is detected in either dimension, false otherwise
pub fn check_for_layover(burst: &Sentinel1SlcProduct, dem: &DEM) -> bool {
    let osh = burst.orbital_state_history();

    // State machine states for each dimension
    #[derive(Debug, PartialEq)]
    enum OrderingState {
        Initial,
        Increasing,
        Decreasing,
    }

    // Helper function to get radar coordinates for a point
    let get_radar_coords = |lat: f64, lon: f64| -> (f32, f32) {
        let pos = dem.get_ecef_at_lat_lon(lat, lon);
        let zero_doppler = osh.find_zero_doppler_state(pos.into());
        let [azimuth_idx, slant_range_idx] =
            radar_coords_to_pixel_coords(zero_doppler, &burst.metadata);
        (azimuth_idx, slant_range_idx)
    };

    // Track states and last points for both dimensions
    let mut azimuth_state = OrderingState::Initial;
    let mut slant_range_state = OrderingState::Initial;
    let mut last_lat = None;
    let mut last_lon = None;
    let mut last_azimuth_idx = None;
    let mut last_slant_range_idx = None;
    let mut last_i = None;
    let mut last_j = None;

    // Track the established ordering relationships
    let mut azimuth_ordering = None;
    let mut slant_range_ordering = None;

    // Iterate through DEM points in latitude order
    let mut points: Vec<(usize, usize, f64, f64)> = dem
        .indexed_lat_lon_height_iter()
        .map(|(i, j, lat, lon, _)| (i, j, lat, lon))
        .collect();
    points.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for (i, j, lat, lon) in points {
        let (azimuth_idx, slant_range_idx) = get_radar_coords(lat, lon);

        // Check azimuth dimension
        match azimuth_state {
            OrderingState::Initial => {
                last_lat = Some(lat);
                last_azimuth_idx = Some(azimuth_idx);
                last_i = Some(i);
                last_j = Some(j);
                azimuth_state = OrderingState::Increasing;
            }
            OrderingState::Increasing => {
                if let (Some(prev_lat), Some(prev_azimuth), Some(prev_i), Some(prev_j)) =
                    (last_lat, last_azimuth_idx, last_i, last_j)
                {
                    // Establish ordering if not yet determined
                    if azimuth_ordering.is_none() && lat != prev_lat {
                        azimuth_ordering = Some(if azimuth_idx > prev_azimuth {
                            "increasing"
                        } else {
                            "decreasing"
                        });
                    }

                    if lat > prev_lat {
                        let expected_ordering = azimuth_ordering.unwrap_or("unknown");
                        let actual_ordering = if azimuth_idx > prev_azimuth {
                            "increasing"
                        } else {
                            "decreasing"
                        };

                        if expected_ordering != actual_ordering {
                            println!(
                                "Azimuth layover detected: Geographic ordering violation\n\
                                 Established relationship: Higher latitude → {} azimuth index\n\
                                 Violation: Higher latitude ({:.6}°N) → {} azimuth index ({:.2})\n\
                                 Location: DEM grid position ({}, {}) → ({}, {})\n\
                                 Geographic coordinates: ({:.6}°N, {:.6}°E) → ({:.6}°N, {:.6}°E)",
                                expected_ordering,
                                prev_lat,
                                actual_ordering,
                                azimuth_idx,
                                prev_i,
                                prev_j,
                                i,
                                j,
                                prev_lat,
                                last_lon.unwrap(),
                                lat,
                                lon
                            );
                            return true;
                        }
                    }
                    if azimuth_idx < prev_azimuth {
                        azimuth_state = OrderingState::Decreasing;
                    }
                }
                last_lat = Some(lat);
                last_azimuth_idx = Some(azimuth_idx);
                last_i = Some(i);
                last_j = Some(j);
            }
            OrderingState::Decreasing => {
                if let (Some(prev_lat), Some(prev_azimuth), Some(prev_i), Some(prev_j)) =
                    (last_lat, last_azimuth_idx, last_i, last_j)
                {
                    // Establish ordering if not yet determined
                    if azimuth_ordering.is_none() && lat != prev_lat {
                        azimuth_ordering = Some(if azimuth_idx > prev_azimuth {
                            "increasing"
                        } else {
                            "decreasing"
                        });
                    }

                    if lat > prev_lat {
                        let expected_ordering = azimuth_ordering.unwrap_or("unknown");
                        let actual_ordering = if azimuth_idx > prev_azimuth {
                            "increasing"
                        } else {
                            "decreasing"
                        };

                        if expected_ordering != actual_ordering {
                            println!(
                                "Azimuth layover detected: Geographic ordering violation\n\
                                 Established relationship: Higher latitude → {} azimuth index\n\
                                 Violation: Higher latitude ({:.6}°N) → {} azimuth index ({:.2})\n\
                                 Location: DEM grid position ({}, {}) → ({}, {})\n\
                                 Geographic coordinates: ({:.6}°N, {:.6}°E) → ({:.6}°N, {:.6}°E)",
                                expected_ordering,
                                prev_lat,
                                actual_ordering,
                                azimuth_idx,
                                prev_i,
                                prev_j,
                                i,
                                j,
                                prev_lat,
                                last_lon.unwrap(),
                                lat,
                                lon
                            );
                            return true;
                        }
                    }
                    if azimuth_idx > prev_azimuth {
                        azimuth_state = OrderingState::Increasing;
                    }
                }
                last_lat = Some(lat);
                last_azimuth_idx = Some(azimuth_idx);
                last_i = Some(i);
                last_j = Some(j);
            }
        }

        // Check slant range dimension
        match slant_range_state {
            OrderingState::Initial => {
                last_lon = Some(lon);
                last_slant_range_idx = Some(slant_range_idx);
                slant_range_state = OrderingState::Increasing;
            }
            OrderingState::Increasing => {
                if let (Some(prev_lon), Some(prev_slant_range), Some(prev_i), Some(prev_j)) =
                    (last_lon, last_slant_range_idx, last_i, last_j)
                {
                    // Establish ordering if not yet determined
                    if slant_range_ordering.is_none() && lon != prev_lon {
                        slant_range_ordering = Some(if slant_range_idx > prev_slant_range {
                            "increasing"
                        } else {
                            "decreasing"
                        });
                    }

                    if lon > prev_lon {
                        let expected_ordering = slant_range_ordering.unwrap_or("unknown");
                        let actual_ordering = if slant_range_idx > prev_slant_range {
                            "increasing"
                        } else {
                            "decreasing"
                        };

                        if expected_ordering != actual_ordering {
                            println!(
                                "Slant range layover detected: Geographic ordering violation\n\
                                 Established relationship: Higher longitude → {} slant range index\n\
                                 Violation: Higher longitude ({:.6}°E) → {} slant range index ({:.2})\n\
                                 Location: DEM grid position ({}, {}) → ({}, {})\n\
                                 Geographic coordinates: ({:.6}°N, {:.6}°E) → ({:.6}°N, {:.6}°E)",
                                expected_ordering,
                                prev_lon,
                                actual_ordering,
                                slant_range_idx,
                                prev_i,
                                prev_j,
                                i,
                                j,
                                last_lat.unwrap(),
                                prev_lon,
                                lat,
                                lon
                            );
                            return true;
                        }
                    }
                    if slant_range_idx < prev_slant_range {
                        slant_range_state = OrderingState::Decreasing;
                    }
                }
                last_lon = Some(lon);
                last_slant_range_idx = Some(slant_range_idx);
            }
            OrderingState::Decreasing => {
                if let (Some(prev_lon), Some(prev_slant_range), Some(prev_i), Some(prev_j)) =
                    (last_lon, last_slant_range_idx, last_i, last_j)
                {
                    // Establish ordering if not yet determined
                    if slant_range_ordering.is_none() && lon != prev_lon {
                        slant_range_ordering = Some(if slant_range_idx > prev_slant_range {
                            "increasing"
                        } else {
                            "decreasing"
                        });
                    }

                    if lon > prev_lon {
                        let expected_ordering = slant_range_ordering.unwrap_or("unknown");
                        let actual_ordering = if slant_range_idx > prev_slant_range {
                            "increasing"
                        } else {
                            "decreasing"
                        };

                        if expected_ordering != actual_ordering {
                            println!(
                                "Slant range layover detected: Geographic ordering violation\n\
                                 Established relationship: Higher longitude → {} slant range index\n\
                                 Violation: Higher longitude ({:.6}°E) → {} slant range index ({:.2})\n\
                                 Location: DEM grid position ({}, {}) → ({}, {})\n\
                                 Geographic coordinates: ({:.6}°N, {:.6}°E) → ({:.6}°N, {:.6}°E)",
                                expected_ordering,
                                prev_lon,
                                actual_ordering,
                                slant_range_idx,
                                prev_i,
                                prev_j,
                                i,
                                j,
                                last_lat.unwrap(),
                                prev_lon,
                                lat,
                                lon
                            );
                            return true;
                        }
                    }
                    if slant_range_idx > prev_slant_range {
                        slant_range_state = OrderingState::Increasing;
                    }
                }
                last_lon = Some(lon);
                last_slant_range_idx = Some(slant_range_idx);
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Needs to open external files"]
    fn test_check_layover() {
        let dem = DEM::open_file("dem.tif");
        let reference = Sentinel1SlcProduct::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151022T122546_20151022T122546_008265_00BA51_422D.SAFE",
        )
        .unwrap();
        let secondary = Sentinel1SlcProduct::load_first_from_directory(
            "download/S1A_IW_SLC__1SSV_20151010T122546_20151010T122546_008090_00B578_BFAD.SAFE",
        )
        .unwrap();

        println!("\nChecking layover in reference image...");
        let ref_layover = check_for_layover(&reference, &dem);
        println!("Reference image layover detected: {ref_layover}\n");

        println!("Checking layover in secondary image...");
        let sec_layover = check_for_layover(&secondary, &dem);
        println!("Secondary image layover detected: {sec_layover}\n");

        // If layover is detected in either image, we should fail the test
        assert!(
            !ref_layover && !sec_layover,
            "Layover detected in {} image",
            if ref_layover {
                "reference"
            } else {
                "secondary"
            }
        );
    }
}
