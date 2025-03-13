use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Product {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "adsHeader")]
    pub ads_header: AdsHeader,
    #[serde(rename = "qualityInformation")]
    pub quality_information: QualityInformation,
    #[serde(rename = "generalAnnotation")]
    pub general_annotation: GeneralAnnotation,
    #[serde(rename = "imageAnnotation")]
    pub image_annotation: ImageAnnotation,
    #[serde(rename = "dopplerCentroid")]
    pub doppler_centroid: DopplerCentroid,
    #[serde(rename = "antennaPattern")]
    pub antenna_pattern: ProductAntennaPattern,
    #[serde(rename = "swathTiming")]
    pub swath_timing: SwathTiming,
    #[serde(rename = "geolocationGrid")]
    pub geolocation_grid: GeolocationGrid,
    #[serde(rename = "coordinateConversion")]
    pub coordinate_conversion: CoordinateConversion,
    #[serde(rename = "swathMerging")]
    pub swath_merging: SwathMerging,
}

#[derive(Serialize, Deserialize)]
pub struct AdsHeader {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "missionId")]
    pub mission_id: String,
    #[serde(rename = "productType")]
    pub product_type: String,
    pub polarisation: String,
    pub mode: String,
    pub swath: String,
    #[serde(rename = "startTime")]
    pub start_time: String,
    #[serde(rename = "stopTime")]
    pub stop_time: String,
    #[serde(rename = "absoluteOrbitNumber")]
    pub absolute_orbit_number: String,
    #[serde(rename = "missionDataTakeId")]
    pub mission_data_take_id: String,
    #[serde(rename = "imageNumber")]
    pub image_number: String,
}

#[derive(Serialize, Deserialize)]
pub struct QualityInformation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "productQualityIndex")]
    pub product_quality_index: String,
    #[serde(rename = "qualityDataList")]
    pub quality_data_list: QualityDataList,
}

#[derive(Serialize, Deserialize)]
pub struct QualityDataList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "qualityData")]
    pub quality_data: QualityData,
}

#[derive(Serialize, Deserialize)]
pub struct QualityData {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "downlinkQuality")]
    pub downlink_quality: DownlinkQuality,
    #[serde(rename = "rawDataAnalysisQuality")]
    pub raw_data_analysis_quality: RawDataAnalysisQuality,
    #[serde(rename = "dopplerCentroidQuality")]
    pub doppler_centroid_quality: DopplerCentroidQuality,
    #[serde(rename = "imageQuality")]
    pub image_quality: ImageQuality,
}

#[derive(Serialize, Deserialize)]
pub struct DownlinkQuality {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "iInputDataMean")]
    pub i_input_data_mean: String,
    #[serde(rename = "qInputDataMean")]
    pub q_input_data_mean: String,
    #[serde(rename = "inputDataMeanOutsideNominalRangeFlag")]
    pub input_data_mean_outside_nominal_range_flag: String,
    #[serde(rename = "iInputDataStdDev")]
    pub i_input_data_std_dev: String,
    #[serde(rename = "qInputDataStdDev")]
    pub q_input_data_std_dev: String,
    #[serde(rename = "inputDataStDevOutsideNominalRangeFlag")]
    pub input_data_st_dev_outside_nominal_range_flag: String,
    #[serde(rename = "numDownlinkInputDataGaps")]
    pub num_downlink_input_data_gaps: String,
    #[serde(rename = "downlinkGapsInInputDataSignificantFlag")]
    pub downlink_gaps_in_input_data_significant_flag: String,
    #[serde(rename = "numDownlinkInputMissingLines")]
    pub num_downlink_input_missing_lines: String,
    #[serde(rename = "downlinkMissingLinesSignificantFlag")]
    pub downlink_missing_lines_significant_flag: String,
    #[serde(rename = "numInstrumentInputDataGaps")]
    pub num_instrument_input_data_gaps: String,
    #[serde(rename = "instrumentGapsInInputDataSignificantFlag")]
    pub instrument_gaps_in_input_data_significant_flag: String,
    #[serde(rename = "numInstrumentInputMissingLines")]
    pub num_instrument_input_missing_lines: String,
    #[serde(rename = "instrumentMissingLinesSignificantFlag")]
    pub instrument_missing_lines_significant_flag: String,
    #[serde(rename = "numSsbErrorInputDataGaps")]
    pub num_ssb_error_input_data_gaps: String,
    #[serde(rename = "ssbErrorGapsInInputDataSignificantFlag")]
    pub ssb_error_gaps_in_input_data_significant_flag: String,
    #[serde(rename = "numSsbErrorInputMissingLines")]
    pub num_ssb_error_input_missing_lines: String,
    #[serde(rename = "ssbErrorMissingLinesSignificantFlag")]
    pub ssb_error_missing_lines_significant_flag: String,
    #[serde(rename = "chirpSourceUsed")]
    pub chirp_source_used: String,
    #[serde(rename = "pgSourceUsed")]
    pub pg_source_used: String,
    #[serde(rename = "rrfSpectrumUsed")]
    pub rrf_spectrum_used: String,
    #[serde(rename = "replicaReconstructionFailedFlag")]
    pub replica_reconstruction_failed_flag: String,
    #[serde(rename = "meanPgProductAmplitude")]
    pub mean_pg_product_amplitude: String,
    #[serde(rename = "stdDevPgProductAmplitude")]
    pub std_dev_pg_product_amplitude: String,
    #[serde(rename = "meanPgProductPhase")]
    pub mean_pg_product_phase: String,
    #[serde(rename = "stdDevPgProductPhase")]
    pub std_dev_pg_product_phase: String,
    #[serde(rename = "pgProductDerivationFailedFlag")]
    pub pg_product_derivation_failed_flag: String,
    #[serde(rename = "invalidDownlinkParamsFlag")]
    pub invalid_downlink_params_flag: String,
}

#[derive(Serialize, Deserialize)]
pub struct RawDataAnalysisQuality {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "iBias")]
    pub i_bias: String,
    #[serde(rename = "iBiasSignificanceFlag")]
    pub i_bias_significance_flag: String,
    #[serde(rename = "qBias")]
    pub q_bias: String,
    #[serde(rename = "qBiasSignificanceFlag")]
    pub q_bias_significance_flag: String,
    #[serde(rename = "iqGainImbalance")]
    pub iq_gain_imbalance: String,
    #[serde(rename = "iqGainSignificanceFlag")]
    pub iq_gain_significance_flag: String,
    #[serde(rename = "iqQuadratureDeparture")]
    pub iq_quadrature_departure: String,
    #[serde(rename = "iqQuadratureDepartureSignificanceFlag")]
    pub iq_quadrature_departure_significance_flag: String,
}

#[derive(Serialize, Deserialize)]
pub struct DopplerCentroidQuality {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "dcMethod")]
    pub dc_method: String,
    #[serde(rename = "dopplerCentroidUncertainFlag")]
    pub doppler_centroid_uncertain_flag: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageQuality {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "imageStatistics")]
    pub image_statistics: ImageQualityImageStatistics,
    #[serde(rename = "outputDataMeanOutsideNominalRangeFlag")]
    pub output_data_mean_outside_nominal_range_flag: String,
    #[serde(rename = "outputDataStDevOutsideNominalRangeFlag")]
    pub output_data_st_dev_outside_nominal_range_flag: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageQualityImageStatistics {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "outputDataMean")]
    pub output_data_mean: ImageQualityImageStatisticsOutputDataMean,
    #[serde(rename = "outputDataStdDev")]
    pub output_data_std_dev: ImageQualityImageStatisticsOutputDataStdDev,
}

#[derive(Serialize, Deserialize)]
pub struct ImageQualityImageStatisticsOutputDataMean {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub re: String,
    pub im: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageQualityImageStatisticsOutputDataStdDev {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub re: String,
    pub im: String,
}

#[derive(Serialize, Deserialize)]
pub struct GeneralAnnotation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "productInformation")]
    pub product_information: ProductInformation,
    #[serde(rename = "downlinkInformationList")]
    pub downlink_information_list: DownlinkInformationList,
    #[serde(rename = "orbitList")]
    pub orbit_list: OrbitList,
    #[serde(rename = "attitudeList")]
    pub attitude_list: AttitudeList,
    #[serde(rename = "rawDataAnalysisList")]
    pub raw_data_analysis_list: RawDataAnalysisList,
    #[serde(rename = "replicaInformationList")]
    pub replica_information_list: ReplicaInformationList,
    #[serde(rename = "noiseList")]
    pub noise_list: NoiseList,
    #[serde(rename = "terrainHeightList")]
    pub terrain_height_list: TerrainHeightList,
    #[serde(rename = "azimuthFmRateList")]
    pub azimuth_fm_rate_list: AzimuthFmRateList,
}

#[derive(Serialize, Deserialize)]
pub struct ProductInformation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub pass: String,
    #[serde(rename = "timelinessCategory")]
    pub timeliness_category: String,
    #[serde(rename = "platformHeading")]
    pub platform_heading: String,
    pub projection: String,
    #[serde(rename = "rangeSamplingRate")]
    pub range_sampling_rate: String,
    #[serde(rename = "radarFrequency")]
    pub radar_frequency: String,
    #[serde(rename = "azimuthSteeringRate")]
    pub azimuth_steering_rate: String,
}

#[derive(Serialize, Deserialize)]
pub struct DownlinkInformationList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "downlinkInformation")]
    pub downlink_information: DownlinkInformation,
}

#[derive(Serialize, Deserialize)]
pub struct DownlinkInformation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub swath: String,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "firstLineSensingTime")]
    pub first_line_sensing_time: String,
    #[serde(rename = "lastLineSensingTime")]
    pub last_line_sensing_time: String,
    pub prf: String,
    #[serde(rename = "bitErrorCount")]
    pub bit_error_count: BitErrorCount,
    #[serde(rename = "downlinkValues")]
    pub downlink_values: DownlinkValues,
}

#[derive(Serialize, Deserialize)]
pub struct BitErrorCount {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "numErrSyncMarker")]
    pub num_err_sync_marker: String,
    #[serde(rename = "numErrDataTakeId")]
    pub num_err_data_take_id: String,
    #[serde(rename = "numErrEccNumber")]
    pub num_err_ecc_number: String,
    #[serde(rename = "numErrTestMode")]
    pub num_err_test_mode: String,
    #[serde(rename = "numErrRxChannelId")]
    pub num_err_rx_channel_id: String,
    #[serde(rename = "numErrInstrumentConfigId")]
    pub num_err_instrument_config_id: String,
    #[serde(rename = "numErrPacketCount")]
    pub num_err_packet_count: u32,
    #[serde(rename = "numErrPriCount")]
    pub num_err_pri_count: u32,
    #[serde(rename = "numErrSsbErrorFlag")]
    pub num_err_ssb_error_flag: String,
    #[serde(rename = "numErrBaqMode")]
    pub num_err_baq_mode: String,
    #[serde(rename = "numErrBaqBlockLength")]
    pub num_err_baq_block_length: String,
    #[serde(rename = "numErrRangeDecimation")]
    pub num_err_range_decimation: String,
    #[serde(rename = "numErrRxGain")]
    pub num_err_rx_gain: String,
    #[serde(rename = "numErrTxRampRate")]
    pub num_err_tx_ramp_rate: String,
    #[serde(rename = "numErrTxPulseStartFrequency")]
    pub num_err_tx_pulse_start_frequency: String,
    #[serde(rename = "numErrRank")]
    pub num_err_rank: String,
    #[serde(rename = "numErrPri")]
    pub num_err_pri: String,
    #[serde(rename = "numErrSwst")]
    pub num_err_swst: String,
    #[serde(rename = "numErrSwl")]
    pub num_err_swl: String,
    #[serde(rename = "numErrPolarisation")]
    pub num_err_polarisation: String,
    #[serde(rename = "numErrTempComp")]
    pub num_err_temp_comp: String,
    #[serde(rename = "numErrElevationBeamAddress")]
    pub num_err_elevation_beam_address: String,
    #[serde(rename = "numErrAzimuthBeamAddress")]
    pub num_err_azimuth_beam_address: String,
    #[serde(rename = "numErrSasTestMode")]
    pub num_err_sas_test_mode: String,
    #[serde(rename = "numErrCalType")]
    pub num_err_cal_type: String,
    #[serde(rename = "numErrCalibrationBeamAddress")]
    pub num_err_calibration_beam_address: String,
    #[serde(rename = "numErrCalMode")]
    pub num_err_cal_mode: String,
    #[serde(rename = "numErrTxPulseNumber")]
    pub num_err_tx_pulse_number: String,
    #[serde(rename = "numErrSignalType")]
    pub num_err_signal_type: String,
    #[serde(rename = "numErrSwapFlag")]
    pub num_err_swap_flag: String,
    #[serde(rename = "numErrSwathNumber")]
    pub num_err_swath_number: String,
    #[serde(rename = "numErrNumberOfQuads")]
    pub num_err_number_of_quads: String,
    #[serde(rename = "numIspHeaderErrors")]
    pub num_isp_header_errors: String,
}

#[derive(Serialize, Deserialize)]
pub struct DownlinkValues {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub pri: String,
    pub rank: String,
    #[serde(rename = "dataTakeId")]
    pub data_take_id: String,
    #[serde(rename = "eccNumber")]
    pub ecc_number: String,
    #[serde(rename = "rxChannelId")]
    pub rx_channel_id: String,
    #[serde(rename = "instrumentConfigId")]
    pub instrument_config_id: String,
    #[serde(rename = "dataFormat")]
    pub data_format: DataFormat,
    #[serde(rename = "rangeDecimation")]
    pub range_decimation: RangeDecimation,
    #[serde(rename = "rxGain")]
    pub rx_gain: String,
    #[serde(rename = "txPulseLength")]
    pub tx_pulse_length: String,
    #[serde(rename = "txPulseStartFrequency")]
    pub tx_pulse_start_frequency: String,
    #[serde(rename = "txPulseRampRate")]
    pub tx_pulse_ramp_rate: String,
    #[serde(rename = "swathNumber")]
    pub swath_number: String,
    #[serde(rename = "swlList")]
    pub swl_list: SwlList,
    #[serde(rename = "swstList")]
    pub swst_list: SwstList,
    #[serde(rename = "pointingStatusList")]
    pub pointing_status_list: PointingStatusList,
}

#[derive(Serialize, Deserialize)]
pub struct DataFormat {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "baqBlockLength")]
    pub baq_block_length: String,
    #[serde(rename = "echoFormat")]
    pub echo_format: String,
    #[serde(rename = "noiseFormat")]
    pub noise_format: String,
    #[serde(rename = "calibrationFormat")]
    pub calibration_format: String,
    #[serde(rename = "meanBitRate")]
    pub mean_bit_rate: String,
}

#[derive(Serialize, Deserialize)]
pub struct RangeDecimation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "decimationFilterBandwidth")]
    pub decimation_filter_bandwidth: String,
    #[serde(rename = "samplingFrequencyAfterDecimation")]
    pub sampling_frequency_after_decimation: String,
    #[serde(rename = "filterLength")]
    pub filter_length: String,
}

#[derive(Serialize, Deserialize)]
pub struct SwlList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub swl: Vec<Swl>,
}

#[derive(Serialize, Deserialize)]
pub struct Swl {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub value: String,
}

#[derive(Serialize, Deserialize)]
pub struct SwstList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub swst: Vec<Swst>,
}

#[derive(Serialize, Deserialize)]
pub struct Swst {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub value: String,
}

#[derive(Serialize, Deserialize)]
pub struct PointingStatusList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "pointingStatus")]
    pub pointing_status: PointingStatus,
}

#[derive(Serialize, Deserialize)]
pub struct PointingStatus {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "aocsOpMode")]
    pub aocs_op_mode: String,
    #[serde(rename = "rollErrorFlag")]
    pub roll_error_flag: String,
    #[serde(rename = "pitchErrorFlag")]
    pub pitch_error_flag: String,
    #[serde(rename = "yawErrorFlag")]
    pub yaw_error_flag: String,
}

#[derive(Serialize, Deserialize)]
pub struct OrbitList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub orbit: Vec<Orbit>,
}

#[derive(Serialize, Deserialize)]
pub struct Orbit {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub time: String,
    pub frame: String,
    pub position: Position,
    pub velocity: Velocity,
}

#[derive(Serialize, Deserialize)]
pub struct Position {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub x: String,
    pub y: String,
    pub z: String,
}

#[derive(Serialize, Deserialize)]
pub struct Velocity {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub x: String,
    pub y: String,
    pub z: String,
}

#[derive(Serialize, Deserialize)]
pub struct AttitudeList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub attitude: Vec<Attitude>,
}

#[derive(Serialize, Deserialize)]
pub struct Attitude {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub time: String,
    pub frame: String,
    pub q0: String,
    pub q1: String,
    pub q2: String,
    pub q3: String,
    pub wx: String,
    pub wy: String,
    pub wz: String,
    pub roll: String,
    pub pitch: String,
    pub yaw: String,
}

#[derive(Serialize, Deserialize)]
pub struct RawDataAnalysisList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "rawDataAnalysis")]
    pub raw_data_analysis: RawDataAnalysis,
}

#[derive(Serialize, Deserialize)]
pub struct RawDataAnalysis {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "iBias")]
    pub i_bias: String,
    #[serde(rename = "qBias")]
    pub q_bias: String,
    #[serde(rename = "iqQuadratureDeparture")]
    pub iq_quadrature_departure: String,
    #[serde(rename = "iqGainImbalance")]
    pub iq_gain_imbalance: String,
    pub support: Support,
}

#[derive(Serialize, Deserialize)]
pub struct Support {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "iBiasUpperBound")]
    pub i_bias_upper_bound: String,
    #[serde(rename = "iBiasLowerBound")]
    pub i_bias_lower_bound: String,
    #[serde(rename = "qBiasUpperBound")]
    pub q_bias_upper_bound: String,
    #[serde(rename = "qBiasLowerBound")]
    pub q_bias_lower_bound: String,
    #[serde(rename = "iqGainUpperBound")]
    pub iq_gain_upper_bound: String,
    #[serde(rename = "iqGainLowerBound")]
    pub iq_gain_lower_bound: String,
    #[serde(rename = "iqQuadratureDepartureUpperBound")]
    pub iq_quadrature_departure_upper_bound: String,
    #[serde(rename = "iqQuadratureDepartureLowerBound")]
    pub iq_quadrature_departure_lower_bound: String,
    #[serde(rename = "iBiasUsedForCorrection")]
    pub i_bias_used_for_correction: String,
    #[serde(rename = "qBiasUsedForCorrection")]
    pub q_bias_used_for_correction: String,
    #[serde(rename = "iqGainImbalanceUsedForCorrection")]
    pub iq_gain_imbalance_used_for_correction: String,
    #[serde(rename = "iqQuadratureDepartureUsedForCorrection")]
    pub iq_quadrature_departure_used_for_correction: String,
}

#[derive(Serialize, Deserialize)]
pub struct ReplicaInformationList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "replicaInformation")]
    pub replica_information: ReplicaInformation,
}

#[derive(Serialize, Deserialize)]
pub struct ReplicaInformation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub swath: String,
    #[serde(rename = "referenceReplica")]
    pub reference_replica: ReferenceReplica,
    #[serde(rename = "replicaList")]
    pub replica_list: ReplicaList,
}

#[derive(Serialize, Deserialize)]
pub struct ReferenceReplica {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "chirpSource")]
    pub chirp_source: String,
    #[serde(rename = "pgSource")]
    pub pg_source: String,
    #[serde(rename = "amplitudeCoefficients")]
    pub amplitude_coefficients: AmplitudeCoefficients,
    #[serde(rename = "phaseCoefficients")]
    pub phase_coefficients: PhaseCoefficients,
    #[serde(rename = "timeDelay")]
    pub time_delay: String,
    pub gain: Gain,
}

#[derive(Serialize, Deserialize)]
pub struct AmplitudeCoefficients {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct PhaseCoefficients {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Gain {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub re: String,
    pub im: String,
}

#[derive(Serialize, Deserialize)]
pub struct ReplicaList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub replica: Vec<Replica>,
}

#[derive(Serialize, Deserialize)]
pub struct Replica {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "crossCorrelationBandwidth")]
    pub cross_correlation_bandwidth: String,
    #[serde(rename = "crossCorrelationPslr")]
    pub cross_correlation_pslr: String,
    #[serde(rename = "crossCorrelationIslr")]
    pub cross_correlation_islr: String,
    #[serde(rename = "crossCorrelationPeakLocation")]
    pub cross_correlation_peak_location: String,
    #[serde(rename = "reconstructedReplicaValidFlag")]
    pub reconstructed_replica_valid_flag: String,
    #[serde(rename = "pgProductAmplitude")]
    pub pg_product_amplitude: String,
    #[serde(rename = "pgProductPhase")]
    pub pg_product_phase: String,
    #[serde(rename = "modelPgProductAmplitude")]
    pub model_pg_product_amplitude: String,
    #[serde(rename = "modelPgProductPhase")]
    pub model_pg_product_phase: String,
    #[serde(rename = "relativePgProductValidFlag")]
    pub relative_pg_product_valid_flag: String,
    #[serde(rename = "absolutePgProductValidFlag")]
    pub absolute_pg_product_valid_flag: String,
    #[serde(rename = "internalTimeDelay")]
    pub internal_time_delay: String,
}

#[derive(Serialize, Deserialize)]
pub struct NoiseList {
    #[serde(rename = "@count")]
    pub count: u32,
}

#[derive(Serialize, Deserialize)]
pub struct TerrainHeightList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "terrainHeight")]
    pub terrain_height: Vec<TerrainHeightListTerrainHeight>,
}

#[derive(Serialize, Deserialize)]
pub struct TerrainHeightListTerrainHeight {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub value: String,
}

#[derive(Serialize, Deserialize)]
pub struct AzimuthFmRateList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthFmRate")]
    pub azimuth_fm_rate: Vec<AzimuthFmRate>,
}

#[derive(Serialize, Deserialize)]
pub struct AzimuthFmRate {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub t0: String,
    #[serde(rename = "azimuthFmRatePolynomial")]
    pub azimuth_fm_rate_polynomial: AzimuthFmRatePolynomial,
}

#[derive(Serialize, Deserialize)]
pub struct AzimuthFmRatePolynomial {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ImageAnnotation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "imageInformation")]
    pub image_information: ImageInformation,
    #[serde(rename = "processingInformation")]
    pub processing_information: ProcessingInformation,
}

#[derive(Serialize, Deserialize)]
pub struct ImageInformation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "productFirstLineUtcTime")]
    pub product_first_line_utc_time: String,
    #[serde(rename = "productLastLineUtcTime")]
    pub product_last_line_utc_time: String,
    #[serde(rename = "ascendingNodeTime")]
    pub ascending_node_time: String,
    #[serde(rename = "anchorTime")]
    pub anchor_time: String,
    #[serde(rename = "productComposition")]
    pub product_composition: String,
    #[serde(rename = "sliceNumber")]
    pub slice_number: String,
    #[serde(rename = "sliceList")]
    pub slice_list: SliceList,
    #[serde(rename = "slantRangeTime")]
    pub slant_range_time: String,
    #[serde(rename = "pixelValue")]
    pub pixel_value: String,
    #[serde(rename = "outputPixels")]
    pub output_pixels: String,
    #[serde(rename = "rangePixelSpacing")]
    pub range_pixel_spacing: String,
    #[serde(rename = "azimuthPixelSpacing")]
    pub azimuth_pixel_spacing: String,
    #[serde(rename = "azimuthTimeInterval")]
    pub azimuth_time_interval: String,
    #[serde(rename = "azimuthFrequency")]
    pub azimuth_frequency: String,
    #[serde(rename = "numberOfSamples")]
    pub number_of_samples: String,
    #[serde(rename = "numberOfLines")]
    pub number_of_lines: String,
    #[serde(rename = "zeroDopMinusAcqTime")]
    pub zero_dop_minus_acq_time: String,
    #[serde(rename = "incidenceAngleMidSwath")]
    pub incidence_angle_mid_swath: String,
    #[serde(rename = "imageStatistics")]
    pub image_statistics: ImageInformationImageStatistics,
}

#[derive(Serialize, Deserialize)]
pub struct SliceList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ImageInformationImageStatistics {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "outputDataMean")]
    pub output_data_mean: ImageInformationImageStatisticsOutputDataMean,
    #[serde(rename = "outputDataStdDev")]
    pub output_data_std_dev: ImageInformationImageStatisticsOutputDataStdDev,
}

#[derive(Serialize, Deserialize)]
pub struct ImageInformationImageStatisticsOutputDataMean {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub re: String,
    pub im: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageInformationImageStatisticsOutputDataStdDev {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub re: String,
    pub im: String,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessingInformation {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "rawDataAnalysisUsed")]
    pub raw_data_analysis_used: String,
    #[serde(rename = "orbitDataFileUsed")]
    pub orbit_data_file_used: String,
    #[serde(rename = "attitudeDataFileUsed")]
    pub attitude_data_file_used: String,
    #[serde(rename = "rxVariationCorrectionApplied")]
    pub rx_variation_correction_applied: String,
    #[serde(rename = "antennaElevationPatternApplied")]
    pub antenna_elevation_pattern_applied: String,
    #[serde(rename = "antennaAzimuthPatternApplied")]
    pub antenna_azimuth_pattern_applied: String,
    #[serde(rename = "antennaAzimuthElementPatternApplied")]
    pub antenna_azimuth_element_pattern_applied: String,
    #[serde(rename = "dcMethod")]
    pub dc_method: String,
    #[serde(rename = "dcInputData")]
    pub dc_input_data: String,
    #[serde(rename = "rangeSpreadingLossCompensationApplied")]
    pub range_spreading_loss_compensation_applied: String,
    #[serde(rename = "srgrConversionApplied")]
    pub srgr_conversion_applied: String,
    #[serde(rename = "detectionPerformed")]
    pub detection_performed: String,
    #[serde(rename = "thermalNoiseCorrectionPerformed")]
    pub thermal_noise_correction_performed: String,
    #[serde(rename = "chirpSource")]
    pub chirp_source: String,
    #[serde(rename = "pgSource")]
    pub pg_source: String,
    #[serde(rename = "rrfSpectrum")]
    pub rrf_spectrum: String,
    #[serde(rename = "applicationLutId")]
    pub application_lut_id: String,
    #[serde(rename = "swathProcParamsList")]
    pub swath_proc_params_list: SwathProcParamsList,
    #[serde(rename = "inputDimensionsList")]
    pub input_dimensions_list: InputDimensionsList,
    #[serde(rename = "referenceRange")]
    pub reference_range: String,
    #[serde(rename = "ellipsoidName")]
    pub ellipsoid_name: String,
    #[serde(rename = "ellipsoidSemiMajorAxis")]
    pub ellipsoid_semi_major_axis: String,
    #[serde(rename = "ellipsoidSemiMinorAxis")]
    pub ellipsoid_semi_minor_axis: String,
    #[serde(rename = "bistaticDelayCorrectionApplied")]
    pub bistatic_delay_correction_applied: String,
    #[serde(rename = "topsFilterConvention")]
    pub tops_filter_convention: String,
    #[serde(rename = "orbitSource")]
    pub orbit_source: String,
    #[serde(rename = "attitudeSource")]
    pub attitude_source: String,
}

#[derive(Serialize, Deserialize)]
pub struct SwathProcParamsList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "swathProcParams")]
    pub swath_proc_params: SwathProcParams,
}

#[derive(Serialize, Deserialize)]
pub struct SwathProcParams {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub swath: String,
    #[serde(rename = "rangeProcessing")]
    pub range_processing: RangeProcessing,
    #[serde(rename = "azimuthProcessing")]
    pub azimuth_processing: AzimuthProcessing,
    #[serde(rename = "processorScalingFactor")]
    pub processor_scaling_factor: String,
}

#[derive(Serialize, Deserialize)]
pub struct RangeProcessing {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "windowType")]
    pub window_type: String,
    #[serde(rename = "windowCoefficient")]
    pub window_coefficient: String,
    #[serde(rename = "totalBandwidth")]
    pub total_bandwidth: String,
    #[serde(rename = "processingBandwidth")]
    pub processing_bandwidth: String,
    #[serde(rename = "lookBandwidth")]
    pub look_bandwidth: String,
    #[serde(rename = "numberOfLooks")]
    pub number_of_looks: String,
    #[serde(rename = "lookOverlap")]
    pub look_overlap: String,
}

#[derive(Serialize, Deserialize)]
pub struct AzimuthProcessing {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "windowType")]
    pub window_type: String,
    #[serde(rename = "windowCoefficient")]
    pub window_coefficient: String,
    #[serde(rename = "totalBandwidth")]
    pub total_bandwidth: String,
    #[serde(rename = "processingBandwidth")]
    pub processing_bandwidth: String,
    #[serde(rename = "lookBandwidth")]
    pub look_bandwidth: String,
    #[serde(rename = "numberOfLooks")]
    pub number_of_looks: String,
    #[serde(rename = "lookOverlap")]
    pub look_overlap: String,
}

#[derive(Serialize, Deserialize)]
pub struct InputDimensionsList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "inputDimensions")]
    pub input_dimensions: InputDimensions,
}

#[derive(Serialize, Deserialize)]
pub struct InputDimensions {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub swath: String,
    #[serde(rename = "numberOfInputSamples")]
    pub number_of_input_samples: String,
    #[serde(rename = "numberOfInputLines")]
    pub number_of_input_lines: String,
}

#[derive(Serialize, Deserialize)]
pub struct DopplerCentroid {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "dcEstimateList")]
    pub dc_estimate_list: DcEstimateList,
}

#[derive(Serialize, Deserialize)]
pub struct DcEstimateList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "dcEstimate")]
    pub dc_estimate: Vec<DcEstimate>,
}

#[derive(Serialize, Deserialize)]
pub struct DcEstimate {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    pub t0: String,
    #[serde(rename = "geometryDcPolynomial")]
    pub geometry_dc_polynomial: GeometryDcPolynomial,
    #[serde(rename = "dataDcPolynomial")]
    pub data_dc_polynomial: DataDcPolynomial,
    #[serde(rename = "dataDcRmsError")]
    pub data_dc_rms_error: String,
    #[serde(rename = "dataDcRmsErrorAboveThreshold")]
    pub data_dc_rms_error_above_threshold: String,
    #[serde(rename = "fineDceAzimuthStartTime")]
    pub fine_dce_azimuth_start_time: String,
    #[serde(rename = "fineDceAzimuthStopTime")]
    pub fine_dce_azimuth_stop_time: String,
    #[serde(rename = "fineDceList")]
    pub fine_dce_list: FineDceList,
}

#[derive(Serialize, Deserialize)]
pub struct GeometryDcPolynomial {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct DataDcPolynomial {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct FineDceList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "fineDce")]
    pub fine_dce: Vec<FineDce>,
}

#[derive(Serialize, Deserialize)]
pub struct FineDce {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "slantRangeTime")]
    pub slant_range_time: String,
    pub frequency: String,
}

#[derive(Serialize, Deserialize)]
pub struct ProductAntennaPattern {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "antennaPatternList")]
    pub antenna_pattern_list: AntennaPatternList,
}

#[derive(Serialize, Deserialize)]
pub struct AntennaPatternList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "antennaPattern")]
    pub antenna_pattern: Vec<AntennaPatternListAntennaPattern>,
}

#[derive(Serialize, Deserialize)]
pub struct AntennaPatternListAntennaPattern {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub swath: String,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "slantRangeTime")]
    pub slant_range_time: AntennaPatternSlantRangeTime,
    #[serde(rename = "elevationAngle")]
    pub elevation_angle: AntennaPatternElevationAngle,
    #[serde(rename = "elevationPattern")]
    pub elevation_pattern: ElevationPattern,
    #[serde(rename = "incidenceAngle")]
    pub incidence_angle: AntennaPatternIncidenceAngle,
    #[serde(rename = "terrainHeight")]
    pub terrain_height: String,
    pub roll: String,
}

#[derive(Serialize, Deserialize)]
pub struct AntennaPatternSlantRangeTime {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AntennaPatternElevationAngle {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ElevationPattern {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AntennaPatternIncidenceAngle {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SwathTiming {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "linesPerBurst")]
    pub lines_per_burst: String,
    #[serde(rename = "samplesPerBurst")]
    pub samples_per_burst: String,
    #[serde(rename = "burstList")]
    pub burst_list: BurstList,
}

#[derive(Serialize, Deserialize)]
pub struct BurstList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    pub burst: Burst,
}

#[derive(Serialize, Deserialize)]
pub struct Burst {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "azimuthAnxTime")]
    pub azimuth_anx_time: String,
    #[serde(rename = "sensingTime")]
    pub sensing_time: String,
    #[serde(rename = "byteOffset")]
    pub byte_offset: String,
    #[serde(rename = "firstValidSample")]
    pub first_valid_sample: FirstValidSample,
    #[serde(rename = "lastValidSample")]
    pub last_valid_sample: LastValidSample,
}

#[derive(Serialize, Deserialize)]
pub struct FirstValidSample {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct LastValidSample {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct GeolocationGrid {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "geolocationGridPointList")]
    pub geolocation_grid_point_list: GeolocationGridPointList,
}

#[derive(Serialize, Deserialize)]
pub struct GeolocationGridPointList {
    #[serde(rename = "@count")]
    pub count: u32,
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "geolocationGridPoint")]
    pub geolocation_grid_point: Vec<GeolocationGridPoint>,
}

#[derive(Serialize, Deserialize)]
pub struct GeolocationGridPoint {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "azimuthTime")]
    pub azimuth_time: String,
    #[serde(rename = "slantRangeTime")]
    pub slant_range_time: String,
    pub line: String,
    pub pixel: String,
    pub latitude: String,
    pub longitude: String,
    pub height: String,
    #[serde(rename = "incidenceAngle")]
    pub incidence_angle: String,
    #[serde(rename = "elevationAngle")]
    pub elevation_angle: String,
}

#[derive(Serialize, Deserialize)]
pub struct CoordinateConversion {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "coordinateConversionList")]
    pub coordinate_conversion_list: CoordinateConversionList,
}

#[derive(Serialize, Deserialize)]
pub struct CoordinateConversionList {
    #[serde(rename = "@count")]
    pub count: u32,
}

#[derive(Serialize, Deserialize)]
pub struct SwathMerging {
    #[serde(rename = "$text")]
    pub text: Option<String>,
    #[serde(rename = "swathMergeList")]
    pub swath_merge_list: SwathMergeList,
}

#[derive(Serialize, Deserialize)]
pub struct SwathMergeList {
    #[serde(rename = "@count")]
    pub count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::de::from_str;

    #[test]
    fn test_annotation_parsing() {
        // Read the file content
        let xml_content = include_str!("test_data/annotation_example.xml");

        // Parse the XML into our Product struct
        let calibration: Product = from_str(&xml_content).expect("Failed to parse annotation XML");
    }
}
