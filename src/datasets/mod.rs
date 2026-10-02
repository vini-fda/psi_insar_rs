//! Access to external datasets, cached on disk.

use std::path::PathBuf;

pub mod asf;
pub mod cdse;
pub(crate) mod http;
#[cfg(test)]
pub(crate) mod mock_server;

/// Root of the on-disk dataset caches: `{user cache dir}/psi_insar_rs`.
///
/// The user cache directory comes from [`dirs::cache_dir`]: `~/Library/Caches` on macOS,
/// `$XDG_CACHE_HOME` or `~/.cache` on Linux, and `%LOCALAPPDATA%` on Windows. If the platform
/// has none (e.g. `$HOME` is unset), the system temporary directory is used instead.
pub fn cache_root() -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(|| {
        let temp_dir = std::env::temp_dir();
        log::warn!(
            "No user cache directory found, caching datasets under {}",
            temp_dir.display()
        );
        temp_dir
    });
    base.join("psi_insar_rs")
}
