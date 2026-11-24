// SPDX-License-Identifier: MIT OR Apache-2.0

//! Set up logging through a [`tracing`](https://docs.rs/tracing) subscriber.
//!
//! This module configures [`tracing_subscriber`](https://docs.rs/tracing_subscriber) with up to two output layers:
//! - **stdout** – human-friendly, ANSI-coloured when attached to a real TTY.
//! - **file** – plain-text, appended via a non-blocking writer to the path
//!   [`resolve_log_file`] picks. File logging is on by default, writing to
//!   [`LOG_FILE`] inside the net-specific `data_dir`.
//!
//! The active log level is controlled (in descending priority) by:
//! 1. The `RUST_LOG` environment variable.
//! 2. The `--debug` flag (`debug` level) or its absence (`info` level).
//!
//! When the `tokio-console` feature is enabled, the registry also enables
//! `tokio=trace` and `runtime=trace` so that `tokio-console` can connect,
//! while the human-facing layers stay quiet through their own per-layer filters.

use core::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use jiff::ZonedDateTime;
use tracing::Level;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::fmt::FormatEvent;
use tracing_subscriber::fmt::FormatFields;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::layer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

/// The file which logging events are written to.
pub(crate) const LOG_FILE: &str = "debug.log";

/// The string used to format the timestamp, via [`LocalTimer`].
///
/// Format: `YYYY-MM-DD HH:mm:ss`
pub(crate) const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

/// The string used to format the timestamp when
/// the log level [`Level::DEBUG`] or higher, via [`LocalTimer`].
///
/// Format: `YYYY-MM-DD HH:mm:ss.sss`
pub(crate) const TIMESTAMP_FORMAT_DEBUG: &str = "%Y-%m-%d %H:%M:%S%.3f";

/// Colored `ERROR` in bright red.
pub(crate) const COLORED_ERROR: &str = "\x1b[0;31mERROR\x1b[0m";

/// Colored `WARN` in bright yellow
pub(crate) const COLORED_WARN: &str = "\x1b[0;33m WARN\x1b[0m";

/// Colored `INFO` in dim green.
pub(crate) const COLORED_INFO: &str = "\x1b[0;32m INFO\x1b[0m";

/// Colored `DEBUG` in dim blue.
pub(crate) const COLORED_DEBUG: &str = "\x1b[0;34mDEBUG\x1b[0m";

/// Colored `TRACE` in dim magenta.
pub(crate) const COLORED_TRACE: &str = "\x1b[0;35mTRACE\x1b[0m";

/// A [`FormatTime`] implementation that renders the local time with `jiff`.
struct LocalTimer {
    format: &'static str,
}

impl LocalTimer {
    fn new(format: &'static str) -> Self {
        Self { format }
    }
}

impl FormatTime for LocalTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        write!(w, "{}", ZonedDateTime::now().strftime(self.format))
    }
}

/// A custom [`FormatEvent`] implementation for [`tracing-subscriber`]'s `fmt` layer.
///
/// Formats log events as:
/// ```text
/// YYYY-MM-DD HH:MM:SS LEVEL target: message
/// ```
///
/// ## Target shortening
///
/// At `INFO` level and above, well-known `floresta_*` crate prefixes are
/// replaced with short aliases for readability:
///
/// | Module Path Prefix         | Alias        |
/// |----------------------------|--------------|
/// | `florestad`                | `florestad`  |
/// | `floresta_chain`           | `chain`      |
/// | `floresta_common`          | `common`     |
/// | `floresta_electrum`        | `electrum`   |
/// | `floresta_compact_filters` | `filters`    |
/// | `floresta_mempool`         | `mempool`    |
/// | `floresta_node`            | `node`       |
/// | `floresta_rpc`             | `rpc`        |
/// | `floresta_watch_only`      | `watch_only` |
/// | `floresta_wire`            | `wire`       |
///
/// When the log level is `DEBUG` and below, the full module path is preserved.
///
/// ## ANSI colors
///
/// When writing to an interactive terminal, log levels are colorized.
/// Colors are suppressed when writing to a file.
///
/// [`tracing-subscriber`]: https://crates.io/crates/tracing-subscriber
pub struct ShortTargetFormatter {
    timer: LocalTimer,
}

impl Default for ShortTargetFormatter {
    fn default() -> Self {
        Self {
            timer: LocalTimer::new(TIMESTAMP_FORMAT),
        }
    }
}

impl ShortTargetFormatter {
    /// Create a new [`ShortTargetFormatter`].
    ///
    /// If `debug` is `true`, it will also log milliseconds.
    pub fn new(debug: bool) -> Self {
        let fmt = if debug {
            TIMESTAMP_FORMAT_DEBUG
        } else {
            TIMESTAMP_FORMAT
        };
        Self {
            timer: LocalTimer::new(fmt),
        }
    }

    /// Maps a full module path to a short human-friendly alias.
    ///
    /// Returns the original target unchanged if no alias is defined for it.
    fn short_target(target: &str) -> &str {
        if target.starts_with("florestad") {
            "florestad"
        } else if target.starts_with("floresta_chain") {
            "chain"
        } else if target.starts_with("floresta_common") {
            "common"
        } else if target.starts_with("floresta_electrum") {
            "electrum"
        } else if target.starts_with("floresta_compact_filters") {
            "compact_filters"
        } else if target.starts_with("floresta_mempool") {
            "mempool"
        } else if target.starts_with("floresta_node") {
            "node"
        } else if target.starts_with("floresta_rpc") {
            "rpc"
        } else if target.starts_with("floresta_watch_only") {
            "watch_only"
        } else if target.starts_with("floresta_wire") {
            "wire"
        } else {
            target
        }
    }
}

impl<S, N> FormatEvent<S, N> for ShortTargetFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> fmt::Result {
        // Check if the `Writer` has support for ANSI escape codes.
        let writer_supports_ansi_escaping = writer.has_ansi_escapes();

        // Get the event's metadata.
        let event_metadata = event.metadata();

        // Timestamp (ANSI colored if the TTY supports it).
        if writer_supports_ansi_escaping {
            write!(writer, "\x1b[2m")?;
        }
        self.timer.format_time(&mut writer)?;
        if writer_supports_ansi_escaping {
            write!(writer, "\x1b[0m ")?;
        } else {
            write!(writer, " ")?;
        }

        // Level (ANSI colored if the TTY supports it).
        if writer_supports_ansi_escaping {
            let colored_level = match *event_metadata.level() {
                Level::ERROR => COLORED_ERROR,
                Level::WARN => COLORED_WARN,
                Level::INFO => COLORED_INFO,
                Level::DEBUG => COLORED_DEBUG,
                Level::TRACE => COLORED_TRACE,
            };
            write!(writer, "{} ", colored_level)?;
        } else {
            write!(writer, "{:>5} ", event_metadata.level())?;
        }

        // Target (ANSI colored if the TTY supports it).
        let target = if tracing::enabled!(Level::DEBUG) {
            event_metadata.target()
        } else {
            Self::short_target(event_metadata.target())
        };
        if writer_supports_ansi_escaping {
            write!(writer, "\x1b[2m{}\x1b[0m: ", target)?;
        } else {
            write!(writer, "{}: ", target)?;
        }

        // Log Message and Fields.
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// Resolves the log file path from the CLI flags, mirroring Bitcoin Core.
///
/// File logging is on by default. `--no-log-file` turns it off and wins over any
/// path that was also passed. `--log-file=<file>` overrides the default
/// [`LOG_FILE`] name: absolute paths are used as given, while relative ones are
/// prefixed by `datadir`, which is already net-specific.
///
/// Returns `None` when file logging is disabled.
pub(crate) fn resolve_log_file(
    no_log_file: bool,
    log_file: Option<impl AsRef<Path>>,
    datadir: impl AsRef<Path>,
) -> Option<PathBuf> {
    if no_log_file {
        return None;
    }

    let path = log_file
        .as_ref()
        .map(AsRef::as_ref)
        .unwrap_or(Path::new(LOG_FILE));

    // Like Core, absolute paths are taken as given; relative ones are joined
    // onto the datadir.
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        Some(datadir.as_ref().join(path))
    }
}

/// Initialises the global [`tracing`] subscriber for the application.
///
/// Depending on the flags provided, the subscriber can write to stdout, a log
/// file, both, or neither. The two layers are independent and may be enabled in
/// any combination at runtime.
///
/// # Arguments
///
/// * `log_file` – Full path of the file log output is appended to, as resolved
///   by [`resolve_log_file`]. `None` disables file logging. The parent
///   directory must already exist.
/// * `log_to_stdout` – Emit log output to stdout.
/// * `debug` – Set the default log level to `debug`. When `false` the
///   level defaults to `info`. In both cases `RUST_LOG` overrides the default.
///
/// # Returns
///
/// Returns `Ok(Some(guard))` when file logging is active. The [`WorkerGuard`]
/// must be kept alive for the duration of the program; dropping it flushes and
/// shuts down the non-blocking file-writer thread. Returns `Ok(None)` when file
/// logging is disabled.
///
/// # Errors
///
/// Returns [`io::Error`] if `log_file` is `Some` and the path cannot be created
/// or opened for appending.
///
/// # Panics
///
/// Panics if a global [`tracing`] subscriber has already been
/// installed (e.g. if this function is called more than once).
pub fn start_logger(
    log_file: Option<impl AsRef<Path>>,
    log_to_stdout: bool,
    log_level: Level,
) -> Result<Option<WorkerGuard>, io::Error> {
    let is_debug = log_level >= Level::DEBUG;
    let make_filter = || {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level.to_string()))
    };

    // Formatter for events destined to `stdout`.
    let ansi_tty = io::IsTerminal::is_terminal(&io::stdout());
    let fmt_layer_stdout = log_to_stdout.then(|| {
        layer()
            .with_writer(io::stdout)
            .with_ansi(ansi_tty)
            .event_format(ShortTargetFormatter::new(is_debug))
            .with_filter(make_filter())
    });

    // Formatter for events destined to the log file.
    let mut guard = None;
    let fmt_layer_logfile = match log_file.as_ref().map(AsRef::as_ref) {
        None => None,
        Some(path) => {
            // Opened before installing the subscriber, so an unusable path fails here.
            let file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;

            let (non_blocking, file_guard) = tracing_appender::non_blocking(file);
            guard = Some(file_guard);
            Some(
                layer()
                    .with_writer(non_blocking)
                    .with_ansi(false)
                    .event_format(ShortTargetFormatter::new(is_debug))
                    .with_filter(make_filter()),
            )
        }
    };

    let registry = tracing_subscriber::registry()
        .with(fmt_layer_stdout)
        .with(fmt_layer_logfile);

    #[cfg(feature = "tokio-console")]
    // Spawns the tokio-console gRPC server on a dedicated background thread
    // with its own runtime.
    let registry = registry.with(console_subscriber::spawn());

    registry.init();

    Ok(guard)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use super::LOG_FILE;
    use super::resolve_log_file;

    /// With no flags, we log to [`LOG_FILE`] inside the datadir.
    #[test]
    fn resolve_log_file_defaults_to_datadir() {
        let resolved = resolve_log_file(false, None::<&Path>, "/home/user/.floresta");

        assert_eq!(
            resolved,
            Some(PathBuf::from("/home/user/.floresta").join(LOG_FILE))
        );
    }

    /// The datadir is already net-specific, so the default lands under it.
    #[test]
    fn resolve_log_file_uses_the_net_specific_datadir() {
        let resolved = resolve_log_file(false, None::<&Path>, "/home/user/.floresta/signet");

        assert_eq!(
            resolved,
            Some(PathBuf::from("/home/user/.floresta/signet/debug.log"))
        );
    }

    /// Relative paths are prefixed by the datadir, like Core.
    #[test]
    fn resolve_log_file_joins_relative_paths_onto_the_datadir() {
        let resolved = resolve_log_file(false, Some("logs/node.log"), "/home/user/.floresta");

        assert_eq!(
            resolved,
            Some(PathBuf::from("/home/user/.floresta/logs/node.log"))
        );
    }

    /// Absolute paths are taken as given, ignoring the datadir.
    #[test]
    fn resolve_log_file_keeps_absolute_paths_untouched() {
        let resolved = resolve_log_file(
            false,
            Some("/var/log/florestad.log"),
            "/home/user/.floresta",
        );

        assert_eq!(resolved, Some(PathBuf::from("/var/log/florestad.log")));
    }

    /// `--no-log-file` disables file logging.
    #[test]
    fn resolve_log_file_returns_none_when_disabled() {
        let resolved = resolve_log_file(true, None::<&Path>, "/home/user/.floresta");

        assert_eq!(resolved, None);
    }

    /// `--no-log-file` wins when a path was passed too.
    #[test]
    fn resolve_log_file_disabled_overrides_a_supplied_path() {
        let resolved =
            resolve_log_file(true, Some("/var/log/florestad.log"), "/home/user/.floresta");

        assert_eq!(resolved, None);
    }
}
