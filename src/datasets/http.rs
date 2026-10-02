//! HTTP and file helpers shared by the dataset downloaders.

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use log::warn;
use ureq::Body;
use ureq::http::Response;

/// User agent sent by the dataset downloaders.
pub(crate) const USER_AGENT: &str = concat!("psi_insar_rs/", env!("CARGO_PKG_VERSION"));

/// Whether a request error may go away when the request is repeated.
pub(crate) fn is_transient(err: &ureq::Error) -> bool {
    matches!(
        err,
        ureq::Error::Io(_)
            | ureq::Error::Timeout(_)
            | ureq::Error::HostNotFound
            | ureq::Error::ConnectionFailed
            | ureq::Error::Protocol(_)
    )
}

/// The start of an error response body, for diagnostics.
pub(crate) fn body_snippet(resp: &mut Response<Body>) -> String {
    resp.body_mut()
        .with_config()
        .limit(1024)
        .read_to_string()
        .unwrap_or_else(|err| format!("<unreadable body: {err}>"))
}

/// Why [`call_with_retries`] did not return a response.
pub(crate) enum RetryError {
    /// A request error that is not worth retrying.
    Request(ureq::Error),
    /// Every attempt failed with a transient error; `last_error` describes the last one.
    Exhausted { retries: u32, last_error: String },
}

/// Calls `request`, retrying transport errors, `429` and `5xx` responses up to `max_retries`
/// times with exponential backoff starting at `initial_backoff`. Other responses are returned
/// as they are. `what` names the request in log messages.
pub(crate) fn call_with_retries(
    what: &str,
    max_retries: u32,
    initial_backoff: Duration,
    mut request: impl FnMut() -> Result<Response<Body>, ureq::Error>,
) -> Result<Response<Body>, RetryError> {
    let mut backoff = initial_backoff;
    let mut retries = 0;
    loop {
        let error = match request() {
            Ok(mut resp) if resp.status() == 429 || resp.status().is_server_error() => {
                format!("HTTP {}: {}", resp.status(), body_snippet(&mut resp))
            }
            Ok(resp) => return Ok(resp),
            Err(err) if is_transient(&err) => err.to_string(),
            Err(err) => return Err(RetryError::Request(err)),
        };
        if retries == max_retries {
            return Err(RetryError::Exhausted {
                retries,
                last_error: error,
            });
        }
        retries += 1;
        warn!("{what} failed ({error}), retry {retries}/{max_retries} in {backoff:?}");
        thread::sleep(backoff);
        backoff *= 2;
    }
}

/// Removes a temporary file or directory when dropped.
pub(crate) struct TempPath(pub(crate) PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = if self.0.is_dir() {
            fs::remove_dir_all(&self.0)
        } else {
            fs::remove_file(&self.0)
        };
    }
}

/// Appends `suffix` to the file name of `path`, e.g. `x.SAFE` -> `x.SAFE.part`.
pub(crate) fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}
