/// Run a task and warn any errors that might occur.
///
/// `try_and_log!` variant for tasks that can fail safely.
macro_rules! try_and_warn {
    ($exerpt: literal, $what:expr) => {
        if let Err(warning) = $what {
            tracing::warn!("{}: {warning}", $exerpt);
        }
    };
}

/// Run a task and log any errors that might occur with `Debug` level
macro_rules! try_and_debug {
    ($excerpt: literal, $what:expr) => {
        if let Err(error) = $what {
            tracing::debug!("{}: {error:?}" , $excerpt);
        }
    };
}

/// Run a task and warn any errors that might occur.
///
/// `try_and_log!` variant for tasks that can fail safely.
macro_rules! try_and_error {
    ($excerpt: literal, $what:expr) => {
        if let Err(warning) = $what {
            tracing::error!("{}: {}", $excerpt, warning);
        }
    };
}


macro_rules! periodic_job {
    ($what:expr, $timer:expr, $interval:ident, $context:ty) => {
        if $timer.elapsed() > Duration::from_secs(<$context>::$interval) {
            if let Err(error) = $what {
                tracing::debug!("Periodic job error ({}): {:?}", stringify!($what), error);
            }
            $timer = Instant::now();
        }
    };
    ($what:expr, $timer:expr, $interval:ident, $context:ty, $no_log:literal) => {
        if $timer.elapsed() > Duration::from_secs(<$context>::$interval) {
            $what;
            $timer = Instant::now();
        }
    };
}

pub(crate) use periodic_job;
pub(crate) use try_and_debug;
pub(crate) use try_and_warn;
pub(crate) use try_and_error;
