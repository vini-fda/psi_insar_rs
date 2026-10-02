//! The snaphu-rs native `.phase` raster container.
//!
//! `.phase` is a snaphu-rs extension (it is *not* understood by the original
//! SNAPHU C program). It wraps a single-band `f32` raster in a small,
//! self-describing header so that the row count and the column count travel
//! with the samples instead of being passed on the command line:
//!
//! ```text
//! byte offset  size  contents
//! 0            4     u32 nrows, native endian
//! 4            4     u32 ncols, native endian
//! 8            56    reserved, zero-filled padding
//! 64           4*n   f32 samples, native endian, row-major (n = nrows*ncols)
//! ```
//!
//! The header is padded to [`PHASE_HEADER_LEN`] bytes so the sample block
//! starts at a 64-byte boundary, which keeps it aligned for `f32` (and for
//! typical SIMD/cache-line access) when a reader memory-maps the file.
//!
//! `nrows` and `ncols` are each `u32`, so their product can exceed `u32::MAX`;
//! every size computation here is done in `u64` (or with checked arithmetic)
//! and is rejected if it cannot be represented on the host.
//!
//! Everything is native endian, so a `.phase` file is not portable between
//! hosts of different byte order.

use ndarray::{Array2, ArrayView2};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Total size of the zero-padded `.phase` header, in bytes.
pub const PHASE_HEADER_LEN: usize = 64;

/// Number of header bytes that carry meaning today (`u32` rows, `u32` cols).
///
/// The remaining `PHASE_HEADER_LEN - PHASE_HEADER_USED_LEN` bytes are reserved
/// and must be written as zeros.
pub const PHASE_HEADER_USED_LEN: usize = 8;

/// Conventional file extension for the format.
pub const PHASE_FILE_EXTENSION: &str = "phase";

/// Raster shape declared by a `.phase` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseDims {
    pub nrows: usize,
    pub ncols: usize,
}

impl PhaseDims {
    /// Number of samples in the file, computed in `u64` because `nrows` and
    /// `ncols` are both `u32` and their product can overflow 32 bits.
    fn sample_count(&self) -> io::Result<u64> {
        let nrows = self.nrows as u64;
        let ncols = self.ncols as u64;
        nrows
            .checked_mul(ncols)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "sample count overflow"))
    }

    /// Total on-disk size of a `.phase` file with this shape.
    fn expected_file_len(&self) -> io::Result<u64> {
        self.sample_count()?
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .and_then(|bytes| bytes.checked_add(PHASE_HEADER_LEN as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "file size overflow"))
    }
}

fn read_u32_ne(bytes: &[u8]) -> u32 {
    u32::from_ne_bytes(bytes.try_into().expect("4-byte slice"))
}

/// Read and validate a `.phase` header from an open file of size `filesize`.
fn read_header_from(fp: &mut File, filesize: u64, path: &Path) -> io::Result<PhaseDims> {
    if filesize < PHASE_HEADER_LEN as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "file {} is {} bytes, too short for a {}-byte .phase header",
                path.display(),
                filesize,
                PHASE_HEADER_LEN
            ),
        ));
    }

    let mut header = [0u8; PHASE_HEADER_LEN];
    fp.read_exact(&mut header)?;
    let nrows = read_u32_ne(&header[0..4]);
    let ncols = read_u32_ne(&header[4..8]);
    if nrows == 0 || ncols == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "file {} declares an empty raster ({} x {})",
                path.display(),
                nrows,
                ncols
            ),
        ));
    }

    let to_host = |value: u32, what: &str| {
        usize::try_from(value).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file {} declares {} = {}, too large for this host",
                    path.display(),
                    what,
                    value
                ),
            )
        })
    };
    let dims = PhaseDims {
        nrows: to_host(nrows, "nrows")?,
        ncols: to_host(ncols, "ncols")?,
    };

    let expected = dims.expected_file_len().map_err(|err| {
        io::Error::new(
            err.kind(),
            format!(
                "file {} declares {} x {}: {}",
                path.display(),
                dims.nrows,
                dims.ncols,
                err
            ),
        )
    })?;
    if filesize != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "file {} is {} bytes but its header declares {} x {} ({} bytes expected)",
                path.display(),
                filesize,
                dims.nrows,
                dims.ncols,
                expected
            ),
        ));
    }
    // The sample block must also be addressable in memory once read.
    usize::try_from(dims.sample_count()?).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "file {} declares {} x {}, too many samples for this host",
                path.display(),
                dims.nrows,
                dims.ncols
            ),
        )
    })?;

    Ok(dims)
}

/// Read and validate the header of a `.phase` file.
///
/// Fails when the file is too short, declares an empty raster, declares
/// dimensions too large for the host, or has a size that disagrees with the
/// declared shape.
pub fn read_phase_header(path: &Path) -> io::Result<PhaseDims> {
    let mut fp = File::open(path)?;
    let filesize = fp.metadata()?.len();
    read_header_from(&mut fp, filesize, path)
}

/// Read a whole `.phase` file into an `nrows x ncols` array, taking its shape
/// from the header.
pub fn read_phase_file(path: &Path) -> io::Result<Array2<f32>> {
    let mut fp = File::open(path)?;
    let filesize = fp.metadata()?.len();
    let dims = read_header_from(&mut fp, filesize, path)?;

    let mut reader = BufReader::new(fp);
    let mut data = Vec::with_capacity(dims.nrows * dims.ncols);
    let mut buf = [0u8; 4];
    for _ in 0..dims.nrows * dims.ncols {
        reader.read_exact(&mut buf)?;
        data.push(f32::from_ne_bytes(buf));
    }
    Array2::from_shape_vec((dims.nrows, dims.ncols), data)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// Write a 2D array (rows x columns) as a `.phase` file.
///
/// Samples are written in logical row-major order regardless of the array's
/// memory layout.
pub fn write_phase_file(raster: ArrayView2<f32>, outfile: &Path) -> io::Result<()> {
    let (height, width) = raster.dim();
    let nrows = u32::try_from(height).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("row count {height} does not fit in the u32 header field"),
        )
    })?;
    let ncols = u32::try_from(width).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("column count {width} does not fit in the u32 header field"),
        )
    })?;
    if nrows == 0 || ncols == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot write an empty .phase raster",
        ));
    }

    let mut header = [0u8; PHASE_HEADER_LEN];
    header[0..4].copy_from_slice(&nrows.to_ne_bytes());
    header[4..8].copy_from_slice(&ncols.to_ne_bytes());

    // Buffer the sample block: a raster can be hundreds of megabytes.
    let mut writer = BufWriter::new(File::create(outfile)?);
    writer.write_all(&header)?;
    for sample in raster.iter() {
        writer.write_all(&sample.to_ne_bytes())?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), nanos))
    }

    fn write_raw(path: &Path, header_rows: u32, header_cols: u32, samples: &[f32]) {
        let mut header = [0u8; PHASE_HEADER_LEN];
        header[0..4].copy_from_slice(&header_rows.to_ne_bytes());
        header[4..8].copy_from_slice(&header_cols.to_ne_bytes());
        let mut fp = File::create(path).unwrap();
        fp.write_all(&header).unwrap();
        for value in samples {
            fp.write_all(&value.to_ne_bytes()).unwrap();
        }
    }

    #[test]
    fn round_trip_preserves_shape_and_samples() {
        let path = temp_file("psi_insar_phase_roundtrip");
        let raster =
            Array2::from_shape_vec((3, 4), (0..12).map(|v| v as f32 * 0.25).collect()).unwrap();
        write_phase_file(raster.view(), &path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            (PHASE_HEADER_LEN + 12 * 4) as u64
        );

        let dims = read_phase_header(&path).unwrap();
        assert_eq!(dims, PhaseDims { nrows: 3, ncols: 4 });
        assert_eq!(read_phase_file(&path).unwrap(), raster);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_uses_logical_row_major_order_for_non_contiguous_views() {
        let path = temp_file("psi_insar_phase_view");
        let full = Array2::from_shape_vec((3, 4), (0..12).map(|v| v as f32).collect()).unwrap();
        // Transposed (column-major) view and a strided slice.
        let transposed = full.t();
        write_phase_file(transposed, &path).unwrap();
        assert_eq!(read_phase_file(&path).unwrap(), transposed);

        let sliced = full.slice(s![1.., ..;2]);
        write_phase_file(sliced, &path).unwrap();
        let read = read_phase_file(&path).unwrap();
        assert_eq!(read, sliced);
        assert_eq!(read.as_slice().unwrap(), &[4.0, 6.0, 8.0, 10.0]);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn header_reserved_bytes_are_zero_filled() {
        let path = temp_file("psi_insar_phase_padding");
        write_phase_file(Array2::<f32>::zeros((2, 2)).view(), &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(
            bytes[PHASE_HEADER_USED_LEN..PHASE_HEADER_LEN]
                .iter()
                .all(|b| *b == 0)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_rejects_empty_raster() {
        let path = temp_file("psi_insar_phase_write_empty");
        let err = write_phase_file(Array2::<f32>::zeros((0, 4)).view(), &path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn read_phase_header_rejects_truncated_file() {
        let path = temp_file("psi_insar_phase_short");
        File::create(&path).unwrap().write_all(&[0u8; 16]).unwrap();
        let err = read_phase_header(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("too short"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_phase_header_rejects_size_mismatch() {
        let path = temp_file("psi_insar_phase_badsize");
        // Header claims 3x4 but only 11 samples follow.
        write_raw(&path, 3, 4, &(0..11).map(|v| v as f32).collect::<Vec<_>>());
        let err = read_phase_header(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("bytes expected"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_phase_header_rejects_zero_dimension() {
        let path = temp_file("psi_insar_phase_empty");
        write_raw(&path, 0, 4, &[]);
        let err = read_phase_header(&path).unwrap_err();
        assert!(err.to_string().contains("empty raster"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_phase_header_rejects_dimensions_whose_product_overflows() {
        let path = temp_file("psi_insar_phase_huge");
        // 2^31 x 2^31 samples: each dimension fits its u32 field, but the
        // sample count (2^62) times 4 bytes does not fit in u64.
        write_raw(&path, 1 << 31, 1 << 31, &[]);
        let err = read_phase_header(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("file size overflow"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_phase_header_rejects_oversized_but_representable_shape() {
        let path = temp_file("psi_insar_phase_big");
        // 1 x 2^30: the declared size is representable, just not this file's.
        write_raw(&path, 1, 1 << 30, &[]);
        let err = read_phase_header(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("bytes expected"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn expected_file_len_rejects_u64_overflow() {
        let dims = PhaseDims {
            nrows: u32::MAX as usize,
            ncols: u32::MAX as usize,
        };
        // The sample count still fits in u64; the byte count does not.
        assert!(dims.sample_count().is_ok());
        assert!(dims.expected_file_len().is_err());
    }
}
