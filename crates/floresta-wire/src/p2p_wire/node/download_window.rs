// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapts SwiftSync's rolling byte cap on requested or downloaded blocks awaiting processing.
//!
//! The cap is maintained in MiB-sized units and defaults to one GiB. Requested blocks reserve
//! their rolling estimated size; downloaded blocks are charged their actual serialized size.
//! The caller records useful block bytes, periodically calls `update`, and reads the cap from
//! `limit_bytes`. This module chooses the cap; it does not send requests or manage peers.
//!
//! Each cycle measures a baseline, tries a nearby cap, then keeps or reverts it before measuring
//! a fresh baseline. Smaller windows win when speeds are similar; only useful throughput decides.
//! Two consecutive rejected trials pause probing for three minutes, while measurements continue.
//! After lowering the cap, observation waits for the old backlog to drain and settle.
//! `Measurement` handles byte counting and timing; `DownloadWindow` handles these decisions.

use std::time::Duration;
use std::time::Instant;

// Window sizes, in MiB. The public admission cap is converted to bytes.
const WINDOW_UNIT_BYTES: usize = 1 << 20;
const MIN_WINDOW: usize = 64;
const DEFAULT_WINDOW: usize = 1 << 10;
const MAX_WINDOW: usize = 1 << 10;
const INITIAL_BLOCK_RESERVATION_BYTES: usize = 1 << 20;
const MIN_BLOCK_RESERVATION_BYTES: usize = 1 << 10;
const MAX_BLOCK_RESERVATION_BYTES: usize = 4 << 20;

// Throughput observation periods.
const SAMPLE_PERIOD: Duration = Duration::from_secs(30);
const MAX_SAMPLE_PERIOD: Duration = Duration::from_secs(120);
const STABLE_PROBE_INTERVAL: Duration = Duration::from_secs(180);

/// A completed throughput measurement and its resulting decision, for logging.
pub(super) struct WindowSample {
    /// Cap during the measurement, in MiB, before this decision changes it.
    pub measured_limit: usize,
    /// Useful throughput during the observation period, excluding draining and settling.
    pub bytes_per_second: f64,
    /// Actual observation duration, excluding draining and settling.
    pub sample_duration: Duration,
    /// Useful bytes counted during this observation.
    pub sample_bytes: u64,
    /// Previous cap in MiB being compared against, only when this sample completes a trial.
    pub baseline_limit: Option<usize>,
    /// Baseline throughput for the completed trial, if any.
    pub baseline_bytes_per_second: Option<f64>,
    /// Logging label for the probe decision or idle/hold state.
    pub action: &'static str,
}

/// Finds a small byte window that sustains throughput by trying nearby limits.
/// This is a continuously refilled pending-byte cap, not a fixed range of heights.
pub(super) struct DownloadWindow {
    /// Current pending-byte cap in MiB.
    limit: usize,
    /// Useful throughput being measured at the current cap.
    measurement: Measurement,
    /// A smaller cap cannot be measured until the old pending blocks drain to it.
    waiting_for_drain: bool,
    /// Direction to try after collecting the next baseline.
    next_direction: Direction,
    /// Baseline for the active trial, or `None` while collecting a fresh baseline.
    probe: Option<Probe>,
    /// Consecutive rejected trials around the same accepted cap; a kept trial resets this.
    rejected_probes: u8,
    /// Earliest time to try another cap after stability is detected; sampling is unaffected.
    probe_after: Instant,
    /// Previous probing eligibility, used to detect when the baseline must reset.
    can_probe: bool,
    /// Whether current worker pressure permits trials that enlarge the window.
    growth_allowed: bool,
    /// Rolling reservation for a requested block whose serialized size is not known yet.
    estimated_block_bytes: usize,
}

impl Default for DownloadWindow {
    fn default() -> Self {
        Self::new(Instant::now())
    }
}

impl DownloadWindow {
    /// Starts at the one-GiB default cap.
    fn new(now: Instant) -> Self {
        Self {
            limit: DEFAULT_WINDOW,
            measurement: Measurement::new(now),
            waiting_for_drain: false,
            next_direction: Direction::Up,
            probe: None,
            rejected_probes: 0,
            probe_after: now,
            can_probe: true,
            growth_allowed: true,
            estimated_block_bytes: INITIAL_BLOCK_RESERVATION_BYTES,
        }
    }

    /// Returns the internal cap in MiB.
    #[cfg(test)]
    pub(super) fn limit(&self) -> usize {
        self.limit
    }

    /// Returns the admission cap in bytes.
    pub(super) fn limit_bytes(&self) -> usize {
        self.limit.saturating_mul(WINDOW_UNIT_BYTES)
    }

    /// Rounds a byte backlog up to the controller's whole-MiB units.
    pub(super) fn units_for_bytes(bytes: usize) -> usize {
        bytes.div_ceil(WINDOW_UNIT_BYTES)
    }

    /// Returns the current reservation for one requested block.
    pub(super) fn estimated_block_bytes(&self) -> usize {
        self.estimated_block_bytes
    }

    /// Updates the requested-block reservation from an accepted block's actual serialized size.
    pub(super) fn observe_block_size(&mut self, bytes: usize) {
        let bytes = bytes.clamp(MIN_BLOCK_RESERVATION_BYTES, MAX_BLOCK_RESERVATION_BYTES);
        self.estimated_block_bytes = (self.estimated_block_bytes.saturating_mul(7) + bytes) / 8;
    }

    #[cfg(test)]
    pub(super) fn probe_down_next(&mut self) {
        self.next_direction = Direction::Down;
    }

    #[cfg(test)]
    pub(super) fn set_limit_for_test(&mut self, limit_mib: usize) {
        self.limit = limit_mib;
    }

    /// Prevents larger-window trials while validation workers already have queued work.
    pub(super) fn set_growth_allowed(&mut self, allowed: bool) {
        self.growth_allowed = allowed;
        if !allowed {
            self.next_direction = Direction::Down;
        }
    }

    /// Describes the current observation state for diagnostic logs without advancing it.
    pub(super) fn phase(&self, now: Instant) -> &'static str {
        if !self.can_probe {
            "hold"
        } else if self.waiting_for_drain {
            "draining"
        } else if now < self.measurement.starts_at {
            "settling"
        } else {
            match self.probe.as_ref().map(|probe| probe.direction) {
                Some(Direction::Up) => "probe-up",
                Some(Direction::Down) => "probe-down",
                None if now < self.probe_after => "stable",
                None => "baseline",
            }
        }
    }

    /// Counts useful bytes outside draining/settling; the caller excludes invalid or duplicate blocks.
    pub(super) fn record_bytes(&mut self, bytes: usize, now: Instant) {
        if !self.waiting_for_drain {
            self.measurement.record_bytes(bytes, now);
        }
    }

    /// Adjusts the cap after an observation period, returning a sample for logging.
    ///
    /// `response_time` is the median peer latency and sets observation and settling times.
    /// `can_probe` is false while draining the download tail or lacking usable peers.
    /// `waiting_window_units` is the requested/downloaded backlog rounded up to whole MiB.
    /// Returns `None` while draining, settling, or collecting a full observation period.
    pub(super) fn update(
        &mut self,
        now: Instant,
        response_time: Duration,
        can_probe: bool,
        waiting_window_units: usize,
    ) -> Option<WindowSample> {
        if can_probe != self.can_probe {
            // A draining tail or a lack of peers cannot tell us the window's capacity.
            // Discard the comparison, but keep the current cap.
            self.can_probe = can_probe;
            self.probe = None;
            self.rejected_probes = 0;
            self.probe_after = now;
            self.waiting_for_drain = can_probe && waiting_window_units > self.limit;
            self.measurement = Measurement::new(now);
        }

        let settling_time = response_time.clamp(Duration::from_secs(5), MAX_SAMPLE_PERIOD);
        if self.waiting_for_drain {
            if waiting_window_units > self.limit {
                return None;
            }
            // Refilling can resume now. Preserve the comparison, but exclude the drain and settling.
            self.waiting_for_drain = false;
            self.measurement = Measurement::new(now + settling_time);
        }

        let throughput = self.measurement.throughput(now, response_time)?;
        let measured_limit = self.limit;
        let sample_duration = now.duration_since(self.measurement.starts_at);
        let sample_bytes = self.measurement.bytes;
        let baseline_limit = self.probe.as_ref().map(|probe| probe.previous_limit);
        let baseline_bytes_per_second = self.probe.as_ref().map(|probe| probe.baseline_throughput);

        // Even a zero-throughput trial must be kept or reverted, not treated as idle.
        let action = if !can_probe {
            "hold"
        } else if let Some(probe) = self.probe.take() {
            self.finish_probe(probe, throughput, now)
        } else if throughput == 0.0 {
            "idle"
        } else if now < self.probe_after {
            // Keep emitting fresh measurements without paying for another unhelpful trial.
            "stable"
        } else {
            self.start_probe(throughput)
        };

        // Shrinking cannot cancel requests: wait for the cap to take effect before settling.
        self.waiting_for_drain = self.limit < measured_limit && waiting_window_units > self.limit;
        let settle = if self.limit != measured_limit {
            settling_time
        } else {
            Duration::ZERO
        };
        self.measurement = Measurement::new(now + settle);
        Some(WindowSample {
            measured_limit,
            bytes_per_second: throughput,
            sample_duration,
            sample_bytes,
            baseline_limit,
            baseline_bytes_per_second,
            action,
        })
    }

    /// Saves this baseline and starts a trial, reversing direction at a window bound.
    fn start_probe(&mut self, throughput: f64) -> &'static str {
        let mut direction = self.next_direction;
        if !self.growth_allowed && matches!(direction, Direction::Up) {
            direction = Direction::Down;
        }
        if direction.candidate(self.limit) == self.limit {
            if !self.growth_allowed {
                return "stable";
            }
            direction = direction.opposite();
        }
        self.probe = Some(Probe {
            previous_limit: self.limit,
            baseline_throughput: throughput,
            direction,
        });
        self.limit = direction.candidate(self.limit);
        match direction {
            Direction::Up => "probe-up",
            Direction::Down => "probe-down",
        }
    }

    /// Keeps or reverts the completed trial and chooses a direction for the next cycle.
    fn finish_probe(&mut self, probe: Probe, throughput: f64, now: Instant) -> &'static str {
        let keep = match probe.direction {
            // More pending blocks must improve useful throughput by over 3%.
            Direction::Up => self.growth_allowed && throughput > probe.baseline_throughput * 1.03,
            // Fewer pending blocks may cost at most 3% of useful throughput.
            Direction::Down => throughput >= probe.baseline_throughput * 0.97,
        };
        self.next_direction = probe.direction;
        if keep {
            self.rejected_probes = 0;
        } else {
            self.limit = probe.previous_limit;
            self.next_direction = probe.direction.opposite();
            self.rejected_probes += 1;
            if self.rejected_probes == 2 {
                // Usually both neighbours failed; at a bound only one can be tried.
                // Restore immediately, then wait before exploring this cap again.
                self.rejected_probes = 0;
                self.probe_after = now + STABLE_PROBE_INTERVAL;
            }
        }
        match (probe.direction, keep) {
            (Direction::Up, true) => "keep-up",
            (Direction::Up, false) => "revert-up",
            (Direction::Down, true) => "keep-down",
            (Direction::Down, false) => "revert-down",
        }
    }
}

/// Counts useful bytes over one observation period, excluding any initial settling time.
struct Measurement {
    /// Observation start; a future instant allows old requests to settle first.
    starts_at: Instant,
    /// Useful bytes recorded since `starts_at`.
    bytes: u64,
}

impl Measurement {
    /// Starts an empty observation at the given instant, possibly after a settling delay.
    fn new(starts_at: Instant) -> Self {
        Self {
            starts_at,
            bytes: 0,
        }
    }

    /// Records useful bytes only after the observation has started.
    fn record_bytes(&mut self, bytes: usize, now: Instant) {
        if now >= self.starts_at {
            self.bytes = self.bytes.saturating_add(bytes as u64);
        }
    }

    /// Returns bytes per second once the latency-adjusted observation period (30–120s) is complete.
    fn throughput(&self, now: Instant, response_time: Duration) -> Option<f64> {
        let period = response_time
            .saturating_mul(2)
            .clamp(SAMPLE_PERIOD, MAX_SAMPLE_PERIOD);
        let elapsed = now.checked_duration_since(self.starts_at)?;
        if elapsed < period {
            return None;
        }

        Some(self.bytes as f64 / elapsed.as_secs_f64())
    }
}

/// Saves the baseline to compare against throughput at a trial window.
struct Probe {
    /// Cap in MiB to restore if the trial does not help.
    previous_limit: usize,
    /// Useful bytes per second measured before starting the trial.
    baseline_throughput: f64,
    /// Direction being tested by the current window.
    direction: Direction,
}

/// Whether a probe tries a larger or smaller pending-byte cap.
#[derive(Clone, Copy)]
enum Direction {
    Up,
    Down,
}

impl Direction {
    /// Reverses the probe direction after an unhelpful trial or at a window bound.
    fn opposite(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }

    /// Proposes a new pending-byte cap in whole MiB.
    fn candidate(self, current_window: usize) -> usize {
        let candidate = match self {
            // Add 25%, adding at least one MiB.
            Self::Up => current_window + (current_window / 4).max(1),
            // Keep 80%, rounding down.
            Self::Down => current_window * 4 / 5,
        };

        candidate.clamp(MIN_WINDOW, MAX_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Completes one observation period at a constant useful byte rate.
    fn sample(window: &mut DownloadWindow, bytes_per_second: usize) -> WindowSample {
        let now = window.measurement.starts_at + SAMPLE_PERIOD;
        window.record_bytes(bytes_per_second * 30, now);
        window.update(now, Duration::from_secs(1), true, 0).unwrap()
    }

    #[test]
    fn one_gib_window_reserves_more_early_blocks() {
        let mut window = DownloadWindow::default();
        assert_eq!(window.limit_bytes(), 1 << 30);
        let initial_capacity = window.limit_bytes() / window.estimated_block_bytes();

        for _ in 0..64 {
            window.observe_block_size(1 << 10);
        }
        let early_capacity = window.limit_bytes() / window.estimated_block_bytes();
        assert!(early_capacity > initial_capacity * 100);

        for _ in 0..64 {
            window.observe_block_size(4 << 20);
        }
        assert!(window.estimated_block_bytes() > 3 << 20);
    }

    #[test]
    fn reports_observation_metadata_and_phases() {
        let start = Instant::now();
        let mut window = DownloadWindow {
            limit: 500,
            ..DownloadWindow::new(start)
        };
        assert_eq!(window.phase(start), "baseline");

        // Maintenance need not run exactly at the observation deadline.
        let now = start + Duration::from_secs(35);
        window.record_bytes(3_500, now);
        let result = window.update(now, Duration::ZERO, true, 0).unwrap();
        assert_eq!(result.sample_duration, Duration::from_secs(35));
        assert_eq!(result.sample_bytes, 3_500);
        assert_eq!(result.bytes_per_second, 100.0);
        assert_eq!(result.baseline_limit, None);
        assert_eq!(result.baseline_bytes_per_second, None);
        assert_eq!(window.phase(now), "settling");
        assert_eq!(window.phase(window.measurement.starts_at), "probe-up");

        let result = sample(&mut window, 120);
        assert_eq!(result.measured_limit, 625);
        assert_eq!(result.sample_duration, SAMPLE_PERIOD);
        assert_eq!(result.sample_bytes, 3_600);
        assert_eq!(result.baseline_limit, Some(500));
        assert_eq!(result.baseline_bytes_per_second, Some(100.0));
        assert_eq!(window.phase(window.measurement.starts_at), "baseline");

        let now = window.measurement.starts_at;
        assert!(window.update(now, Duration::ZERO, false, 0).is_none());
        assert_eq!(window.phase(now), "hold");
    }

    #[test]
    fn keeps_useful_growth_and_reverts_an_ineffective_probe() {
        let mut window = DownloadWindow {
            limit: 500,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(window.limit(), 625);
        assert_eq!(sample(&mut window, 120).action, "keep-up");
        assert_eq!(window.limit(), 625);
        // A fresh baseline follows each decision; it is not an all-time best score.
        assert_eq!(sample(&mut window, 80).action, "probe-up");
        assert_eq!(window.limit(), 781);
        assert_eq!(sample(&mut window, 81).action, "revert-up");
        assert_eq!(window.limit(), 625);
    }

    #[test]
    fn probe_acceptance_respects_the_three_percent_boundaries() {
        for (direction, throughput, action, limit) in [
            (Direction::Up, 103, "revert-up", 500),
            (Direction::Up, 104, "keep-up", 625),
            (Direction::Down, 97, "keep-down", 400),
            (Direction::Down, 96, "revert-down", 500),
        ] {
            let mut window = DownloadWindow {
                limit: 500,
                next_direction: direction,
                ..Default::default()
            };
            sample(&mut window, 100);
            assert_eq!(sample(&mut window, throughput).action, action);
            assert_eq!(window.limit(), limit);
        }
    }

    #[test]
    fn worker_pressure_forces_downward_probes_and_rejects_growth() {
        let mut window = DownloadWindow {
            limit: 500,
            ..Default::default()
        };
        window.set_growth_allowed(false);
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(window.limit(), 400);

        let mut window = DownloadWindow {
            limit: 500,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(window.limit(), 625);
        window.set_growth_allowed(false);
        assert_eq!(sample(&mut window, 120).action, "revert-up");
        assert_eq!(window.limit(), 500);
    }

    #[test]
    fn keeps_a_smaller_window_with_equivalent_throughput() {
        let mut window = DownloadWindow {
            limit: 300,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(sample(&mut window, 100).action, "revert-up");
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(window.limit(), 240);
        assert_eq!(sample(&mut window, 98).action, "keep-down");
        assert_eq!(window.limit(), 240);
        assert_eq!(sample(&mut window, 98).action, "probe-down");
    }

    #[test]
    fn reverts_a_smaller_window_that_reduces_throughput() {
        let mut window = DownloadWindow {
            limit: 300,
            next_direction: Direction::Down,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(sample(&mut window, 90).action, "revert-down");
        assert_eq!(window.limit(), 300);
        assert_eq!(sample(&mut window, 90).action, "probe-up");
    }

    #[test]
    fn probes_inward_at_bounds_and_preserves_mib_units() {
        let mut window = DownloadWindow {
            limit: MIN_WINDOW,
            next_direction: Direction::Down,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(window.limit(), Direction::Up.candidate(MIN_WINDOW));
        window = DownloadWindow {
            limit: MAX_WINDOW,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(window.limit(), Direction::Down.candidate(MAX_WINDOW));

        for limit in MIN_WINDOW..=MAX_WINDOW {
            for direction in [Direction::Up, Direction::Down] {
                let candidate = direction.candidate(limit);
                assert!((MIN_WINDOW..=MAX_WINDOW).contains(&candidate));
            }
        }
    }

    #[test]
    fn does_not_probe_without_progress_and_reverts_a_stalled_trial() {
        let mut window = DownloadWindow::default();
        assert_eq!(sample(&mut window, 0).action, "idle");
        assert_eq!(window.limit(), DEFAULT_WINDOW);
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(sample(&mut window, 0).action, "revert-down");
        assert_eq!(window.limit(), DEFAULT_WINDOW);
    }

    #[test]
    fn starts_at_one_gib_and_probes_down() {
        let mut window = DownloadWindow::default();
        assert_eq!(window.limit(), DEFAULT_WINDOW);
        assert_eq!(window.limit_bytes(), 1 << 30);
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(window.limit(), Direction::Down.candidate(DEFAULT_WINDOW));
        assert_eq!(sample(&mut window, 100).action, "keep-down");
        assert_eq!(window.limit(), Direction::Down.candidate(DEFAULT_WINDOW));
    }

    #[test]
    fn excludes_settling_bytes_and_waits_longer_for_slow_responses() {
        let start = Instant::now();
        let mut window = DownloadWindow::new(start);
        window.record_bytes(3_000, start + SAMPLE_PERIOD);
        window.update(start + SAMPLE_PERIOD, Duration::from_secs(1), true, 0);
        window.record_bytes(1_000_000, start + Duration::from_secs(34));
        assert_eq!(window.measurement.bytes, 0);
        assert!(
            window
                .update(
                    start + Duration::from_secs(60),
                    Duration::from_secs(1),
                    true,
                    0,
                )
                .is_none()
        );
        let sample = sample(&mut window, 100);
        assert_eq!(
            sample.measured_limit,
            Direction::Down.candidate(DEFAULT_WINDOW)
        );
        assert_eq!(sample.bytes_per_second, 100.0);

        window = DownloadWindow::new(start);
        assert!(
            window
                .update(
                    start + Duration::from_secs(79),
                    Duration::from_secs(40),
                    true,
                    0,
                )
                .is_none()
        );
        window.record_bytes(8_000, start + Duration::from_secs(80));
        assert!(
            window
                .update(
                    start + Duration::from_secs(80),
                    Duration::from_secs(40),
                    true,
                    0,
                )
                .is_some()
        );
        assert_eq!(
            window.measurement.starts_at,
            start + Duration::from_secs(120)
        );
    }

    #[test]
    fn downward_probe_waits_for_drain_without_losing_its_baseline() {
        let start = Instant::now();
        let mut window = DownloadWindow {
            limit: MAX_WINDOW,
            ..DownloadWindow::new(start)
        };
        let now = start + SAMPLE_PERIOD;
        window.record_bytes(3_000, now);
        let result = window
            .update(now, Duration::ZERO, true, MAX_WINDOW)
            .unwrap();
        assert_eq!(result.action, "probe-down");
        assert!(window.waiting_for_drain);
        assert_eq!(window.phase(now), "draining");

        // Even a long drain is neither a zero-throughput trial nor part of the next sample.
        let drained_at = now + MAX_SAMPLE_PERIOD * 3;
        window.record_bytes(1_000_000, drained_at);
        assert!(
            window
                .update(drained_at, Duration::ZERO, true, MAX_WINDOW)
                .is_none()
        );
        assert_eq!(window.measurement.bytes, 0);
        let probe = window.probe.as_ref().unwrap();
        assert_eq!(probe.previous_limit, MAX_WINDOW);
        assert_eq!(probe.baseline_throughput, 100.0);

        let smaller_limit = window.limit();
        assert!(
            window
                .update(drained_at, Duration::ZERO, true, smaller_limit)
                .is_none()
        );
        assert!(!window.waiting_for_drain);
        assert_eq!(window.phase(drained_at), "settling");
        assert_eq!(
            window.measurement.starts_at,
            drained_at + Duration::from_secs(5)
        );
        assert_eq!(window.phase(window.measurement.starts_at), "probe-down");
        window.record_bytes(1_000_000, drained_at + Duration::from_secs(4));
        assert_eq!(window.measurement.bytes, 0);
        let result = sample(&mut window, 98);
        assert_eq!(result.action, "keep-down");
        assert_eq!(result.bytes_per_second, 98.0);
        assert_eq!(result.sample_duration, SAMPLE_PERIOD);
        assert_eq!(result.sample_bytes, 2_940);
        assert_eq!(result.baseline_limit, Some(MAX_WINDOW));
        assert_eq!(result.baseline_bytes_per_second, Some(100.0));
        assert_eq!(window.limit(), smaller_limit);
    }

    #[test]
    fn reverted_upward_probe_drains_before_measuring_a_fresh_baseline() {
        let mut window = DownloadWindow {
            limit: 500,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        let now = window.measurement.starts_at + SAMPLE_PERIOD;
        window.record_bytes(3_000, now);
        assert_eq!(
            window
                .update(now, Duration::ZERO, true, 625)
                .unwrap()
                .action,
            "revert-up"
        );
        assert_eq!(window.limit(), 500);
        assert!(window.waiting_for_drain);
        assert!(window.probe.is_none());

        let drained_at = now + MAX_SAMPLE_PERIOD * 3;
        window.record_bytes(1_000_000, drained_at);
        assert!(
            window
                .update(drained_at, Duration::ZERO, true, 501)
                .is_none()
        );
        assert!(
            window
                .update(drained_at, Duration::ZERO, true, 500)
                .is_none()
        );
        assert_eq!(
            window.measurement.starts_at,
            drained_at + Duration::from_secs(5)
        );
        assert_eq!(sample(&mut window, 80).action, "probe-down");
        assert_eq!(window.probe.as_ref().unwrap().baseline_throughput, 80.0);
    }

    #[test]
    fn entering_the_download_tail_cancels_the_drain_wait() {
        let start = Instant::now();
        let mut window = DownloadWindow {
            limit: MAX_WINDOW,
            ..DownloadWindow::new(start)
        };
        let now = start + SAMPLE_PERIOD;
        window.record_bytes(3_000, now);
        window.update(now, Duration::ZERO, true, MAX_WINDOW);
        assert!(window.waiting_for_drain);

        // Tail logging continues even if the remaining requests exceed the smaller cap.
        assert!(
            window
                .update(now, Duration::ZERO, false, MAX_WINDOW)
                .is_none()
        );
        assert!(!window.waiting_for_drain);
        assert!(window.probe.is_none());
        window.record_bytes(3_000, now + SAMPLE_PERIOD);
        let result = window
            .update(now + SAMPLE_PERIOD, Duration::ZERO, false, MAX_WINDOW)
            .unwrap();
        assert_eq!(result.action, "hold");
        assert_eq!(result.bytes_per_second, 100.0);
    }

    #[test]
    fn tail_samples_do_not_contaminate_recovery() {
        let mut window = DownloadWindow::default();
        sample(&mut window, 100);
        let tail = window.measurement.starts_at;
        assert!(window.update(tail, Duration::ZERO, false, 0).is_none());
        assert!(window.probe.is_none());
        window.record_bytes(300, tail + SAMPLE_PERIOD);
        assert_eq!(
            window
                .update(tail + SAMPLE_PERIOD, Duration::ZERO, false, 0)
                .unwrap()
                .action,
            "hold"
        );
        let resume = tail + SAMPLE_PERIOD * 2;
        assert!(window.update(resume, Duration::ZERO, true, 0).is_none());
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(sample(&mut window, 100).action, "revert-up");
    }

    #[test]
    fn eligibility_changes_discard_comparisons_and_hold_the_current_cap() {
        let mut window = DownloadWindow::default();
        sample(&mut window, 100);

        // An eligibility change starts a fresh observation even during settling.
        let now = window.measurement.starts_at - Duration::from_secs(1);
        window.record_bytes(100, now);
        assert!(window.update(now, Duration::ZERO, false, 0).is_none());
        assert_eq!(window.measurement.starts_at, now);
        assert_eq!(window.measurement.bytes, 0);
        assert!(window.probe.is_none());
        assert_eq!(window.limit(), Direction::Down.candidate(DEFAULT_WINDOW));

        // A full observation while probing is disabled still leaves the cap unchanged.
        let result = window
            .update(now + SAMPLE_PERIOD, Duration::ZERO, false, 0)
            .unwrap();
        assert_eq!(result.action, "hold");
        assert_eq!(window.limit(), Direction::Down.candidate(DEFAULT_WINDOW));
    }

    #[test]
    fn rejected_trials_pause_probes_but_keep_reporting_fresh_samples() {
        let mut window = DownloadWindow {
            limit: 500,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(sample(&mut window, 100).action, "revert-up");
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        let rejected_at = window.measurement.starts_at + SAMPLE_PERIOD;
        assert_eq!(sample(&mut window, 90).action, "revert-down");
        assert_eq!(window.limit(), 500);
        assert_eq!(window.rejected_probes, 0);
        assert_eq!(window.probe_after, rejected_at + STABLE_PROBE_INTERVAL);
        assert_eq!(window.phase(rejected_at), "settling");
        assert_eq!(window.phase(window.measurement.starts_at), "stable");

        // Reporting continues every observation, without changing the accepted cap.
        for speed in [110, 120, 130, 140, 150] {
            let result = sample(&mut window, speed);
            assert_eq!(result.action, "stable");
            assert_eq!(result.measured_limit, 500);
            assert_eq!(result.sample_duration, SAMPLE_PERIOD);
            assert_eq!(result.sample_bytes, (speed * 30) as u64);
            assert_eq!(result.bytes_per_second, speed as f64);
            assert_eq!(result.baseline_limit, None);
            assert_eq!(result.baseline_bytes_per_second, None);
            assert_eq!(window.limit(), 500);
            assert!(window.probe.is_none());
        }

        assert_eq!(
            window.phase(window.probe_after - Duration::from_nanos(1)),
            "stable"
        );
        assert_eq!(window.phase(window.probe_after), "baseline");
        let resumed_at = window.measurement.starts_at + SAMPLE_PERIOD;
        assert!(resumed_at >= window.probe_after);
        let result = sample(&mut window, 160);
        assert_eq!(result.action, "probe-up");
        assert_eq!(result.measured_limit, 500);
        assert_eq!(result.sample_duration, SAMPLE_PERIOD);
        assert_eq!(result.sample_bytes, 4_800);
        assert_eq!(window.limit(), 625);
        assert_eq!(window.probe.as_ref().unwrap().baseline_throughput, 160.0);
    }

    #[test]
    fn accepted_changes_clear_rejections_and_keep_searching_without_a_pause() {
        let mut window = DownloadWindow {
            limit: 500,
            ..Default::default()
        };
        assert_eq!(sample(&mut window, 100).action, "probe-up");
        assert_eq!(sample(&mut window, 100).action, "revert-up");
        assert_eq!(window.rejected_probes, 1);
        assert_eq!(sample(&mut window, 100).action, "probe-down");
        assert_eq!(sample(&mut window, 98).action, "keep-down");
        assert_eq!(window.limit(), 400);
        assert_eq!(window.rejected_probes, 0);
        assert_eq!(sample(&mut window, 98).action, "probe-down");
        assert_eq!(sample(&mut window, 80).action, "revert-down");
        assert_eq!(window.limit(), 400);
        assert_eq!(window.rejected_probes, 1);
        assert_eq!(sample(&mut window, 100).action, "probe-up");
    }

    #[test]
    fn eligibility_changes_clear_rejection_history_and_pending_pauses() {
        for (failures, next_action) in [(1, "probe-down"), (2, "probe-up")] {
            let mut window = DownloadWindow {
                limit: 500,
                ..Default::default()
            };
            assert_eq!(sample(&mut window, 100).action, "probe-up");
            assert_eq!(sample(&mut window, 100).action, "revert-up");
            if failures == 2 {
                assert_eq!(sample(&mut window, 100).action, "probe-down");
                assert_eq!(sample(&mut window, 90).action, "revert-down");
                assert!(window.probe_after > window.measurement.starts_at);
            } else {
                assert_eq!(window.rejected_probes, 1);
            }

            let disabled_at = window.measurement.starts_at - Duration::from_secs(1);
            assert!(
                window
                    .update(disabled_at, Duration::ZERO, false, 0)
                    .is_none()
            );
            assert_eq!(window.rejected_probes, 0);
            assert_eq!(window.probe_after, disabled_at);
            assert_eq!(window.phase(disabled_at), "hold");
            assert_eq!(window.limit(), 500);

            let resumed_at = disabled_at + Duration::from_secs(1);
            assert!(window.update(resumed_at, Duration::ZERO, true, 0).is_none());
            assert_eq!(window.probe_after, resumed_at);
            assert_eq!(window.phase(resumed_at), "baseline");
            assert_eq!(sample(&mut window, 100).action, next_action);
        }
    }

    #[test]
    fn repeated_rejections_at_a_window_bound_also_pause_probing() {
        let mut window = DownloadWindow {
            limit: MAX_WINDOW,
            ..Default::default()
        };
        // At the maximum, reversing direction still leads to another downward trial.
        for _ in 0..2 {
            assert_eq!(sample(&mut window, 100).action, "probe-down");
            assert_eq!(sample(&mut window, 90).action, "revert-down");
            assert_eq!(window.limit(), MAX_WINDOW);
        }
        assert_eq!(sample(&mut window, 100).action, "stable");
        assert_eq!(window.limit(), MAX_WINDOW);
        assert!(window.probe.is_none());
    }
}
