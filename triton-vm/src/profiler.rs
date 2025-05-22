//! Allows profiling [proving] and [verifying] of Triton VM's STARK proofs.
//!
//! The profiler is used to measure the time taken by various parts of the
//! proving and verifying process. It can be used to identify bottlenecks and
//! optimize the performance of the STARK proof system.
//!
//! The profiler is thread-local, meaning that each thread has its own profiler.
//! This allows multiple threads to run in parallel without interfering with
//! each other's profiling data.
//!
//! # Enabling Profiling
//!
//! ## In Dependencies
//!
//! In release builds, profiling is disabled by default to allow for the fastest
//! possible proof generation & verification. To enable profiling, either make
//! sure that `debug_assertions` is set, or add the following to your
//! `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! triton-vm = { version = "x.y.z", default-features = false }
//! ```
//!
//! ## For Benchmarks
//!
//! In order to enable profiling when running a benchmark, pass the flag
//! `--no-default-features` to `cargo bench`. In case this is not working in the
//! workspace directory, navigate to the crate's directory and run the command
//! there.
//!
//! # A note on the `no_profile` feature design decision
//!
//! The feature `no_profile` _disables_ profiling. The feature is enabled by
//! default, _i.e._, profiling is disabled by default. This seems backwards.
//! However, it is an expression of favoring performance over profiling data.
//!
//! Note [how Cargo resolves dependencies][deps]: if some dependency is
//! transitively declared multiple times, the _union_ of all features will be
//! enabled. Imagine some dependency `foo` enables a hypothetical `do_profile`
//! feature. If another dependency `bar` requires the most raw performance, it
//! would be slowed down by `foo`'s desire to generate profiling data and could
//! do nothing about it. Instead, disabling profiling by <i>en</i>abling the
//! feature `no_profile` allows `bar` to dictate. This:
//! 1. makes the profile reports of `foo` disappear, which is sad, but
//! 1. lets `bar` be fast, which is more important.
//!
//! [proving]: crate::stark::Stark::prove
//! [verifying]: crate::stark::Stark::verify
//! [deps]: https://doc.rust-lang.org/cargo/reference/features.html#feature-unification

use std::cell::RefCell;
use std::cmp::Ordering;
use std::cmp::max;
use std::collections::HashMap;
use std::env::var as env_var;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::AddAssign;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use colored::Color;
use colored::ColoredString;
use colored::Colorize;
use indexmap::IndexMap;
use itertools::Itertools;
use unicode_width::UnicodeWidthStr;

const ENV_VAR_PROFILER_LIVE_UPDATE: &str = "TVM_PROFILER_LIVE_UPDATE";

thread_local! {
    pub(crate) static PROFILER: RefCell<Option<VMPerformanceProfiler>> =
        const { RefCell::new(None) };
}

/// Start profiling. If the profiler is already running, this function cancels
/// the current profiling session and starts a new one.
///
/// See the module-level documentation for information on how to enable
/// profiling.
pub fn start(profile_name: impl Into<String>) {
    if cfg!(any(debug_assertions, not(feature = "no_profile"))) {
        PROFILER.replace(Some(VMPerformanceProfiler::new(profile_name)));
    }
}

/// Stop the current profiling session and generate a [`VMPerformanceProfile`].
/// If the profiler is disabled or not running, an empty
/// [`VMPerformanceProfile`] is returned.
///
/// See the module-level documentation for information on how to enable
/// profiling.
pub fn finish() -> VMPerformanceProfile {
    cfg!(any(debug_assertions, not(feature = "no_profile")))
        .then(|| PROFILER.take().map(VMPerformanceProfiler::finish))
        .flatten()
        .unwrap_or_default()
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Task {
    name: String,
    parent_index: Option<usize>,
    depth: usize,

    /// The time at which this task was started last.
    start_time: Instant,

    /// The “resident set size” (RSS), i.e., the amount of memory used by this
    /// process, at the start of this task. In bytes.
    ///
    /// If the task is invoked more than once, the rss of subsequent starts
    /// are accumulated and later averaged.
    ///
    /// [`None`] if the rss cannot be obtained, or if an overflow is encountered
    /// during addition.
    start_rss: Option<u64>,

    /// The “resident set size” (RSS), i.e., the amount of memory used by this
    /// process, at the end of this task. In bytes.
    ///
    /// See also [`Self::start_rss`].
    stop_rss: Option<u64>,

    /// The number of times this task was started.
    num_invocations: u64,

    /// The accumulated time spent in this task, across all invocations.
    total_duration: Duration,

    /// The type of work the task is doing. Helps to track time across specific
    /// tasks. For example, if the task is building a Merkle tree, then the
    /// category could be "hash".
    category: Option<String>,
}

/// Helps detect loops in order to aggregate runtimes of their [`Task`]s.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct CodeLocation {
    file: &'static str,
    line: u32,
    column: u32,
}

#[cfg(any(debug_assertions, not(feature = "no_profile")))]
impl CodeLocation {
    pub fn new(file: &'static str, line: u32, column: u32) -> Self {
        CodeLocation { file, line, column }
    }
}

/// Create a [`CodeLocation`] referencing the location in the source code where
/// this macro is called.
#[cfg(any(debug_assertions, not(feature = "no_profile")))]
macro_rules! here {
    () => {
        crate::profiler::CodeLocation::new(file!(), line!(), column!())
    };
}

#[cfg(any(debug_assertions, not(feature = "no_profile")))]
pub(crate) use here;

impl Display for CodeLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// A path of [`CodeLocation`]s that represent a call stack.
/// Helps detect loops in order to aggregate runtimes of their [`Task`]s.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
pub(crate) struct InvocationPath(Vec<CodeLocation>);

impl Display for InvocationPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0.iter().join("/"))
    }
}

#[cfg(any(debug_assertions, not(feature = "no_profile")))]
impl InvocationPath {
    pub fn extend(&self, location: CodeLocation) -> Self {
        let mut locations = self.0.clone();
        locations.push(location);
        Self(locations)
    }
}

/// In bytes.
#[derive(Debug, Copy, Clone)]
enum RssContribution {
    Addition(u64),
    NoChange,
    Removal(u64),
}

impl RssContribution {
    fn from_diff(start: Option<u64>, end: Option<u64>) -> Option<Self> {
        let start = start?;
        let end = end?;
        let contribution = match start.cmp(&end) {
            Ordering::Less => Self::Addition(end - start),
            Ordering::Equal => Self::NoChange,
            Ordering::Greater => Self::Removal(start - end),
        };

        Some(contribution)
    }

    fn format_colored_aligned(self) -> ColoredString {
        let (sign, fmt_str, color) = self.formatting_parts();

        format!("{sign}{fmt_str}").color(color)
    }

    /// The raw parts for formatting `self`: sign, formatted size, color.
    ///
    /// Example: ('+', "1.0 KiB", Color::White)
    fn formatting_parts(self) -> (char, String, Color) {
        // Use IEC units because this is engineering, not marketing.
        const KIB: u64 = 1024;
        const MIB: u64 = KIB * KIB;
        const GIB: u64 = MIB * KIB;
        const TIB: u64 = GIB * KIB;

        // Use SI units to determine when to print what. This ensures that cases
        // like "1000.0 KiB" are turned into "0.1 MiB" instead.
        const KB: u64 = 1000;
        const MB: u64 = KB * KB;
        const GB: u64 = MB * KB;
        const TB: u64 = GB * KB;

        let fmt = |size, unit, unit_str| {
            let sub_unit = unit / KIB; // eg, GiB -> MiB. Must never be smaller than KiB.
            let remainder = Self::round_decimal_fraction::<1>((size % unit) / sub_unit);
            let size = size / unit;
            format!("{size}.{remainder:<1} {unit_str}")
        };

        let (sign, fmt_str, r, g, b) = match self {
            Self::Addition(v) if v >= TB => ('+', fmt(v, TIB, "TiB"), 255, 0, 0),
            Self::Addition(v) if v >= GB => ('+', fmt(v, GIB, "GiB"), 255, 75, 0),
            Self::Addition(v) if v >= MB => ('+', fmt(v, MIB, "MiB"), 255, 150, 0),
            Self::Addition(v) if v >= KB => ('+', fmt(v, KIB, "KiB"), 255, 255, 120),
            Self::Addition(v) => ('+', format!("{v} B  "), 200, 200, 200),

            Self::Removal(v) if v >= TB => ('-', fmt(v, TIB, "TiB"), 0, 255, 0),
            Self::Removal(v) if v >= GB => ('-', fmt(v, GIB, "GiB"), 50, 255, 50),
            Self::Removal(v) if v >= MB => ('-', fmt(v, MIB, "MiB"), 99, 255, 99),
            Self::Removal(v) if v >= KB => ('-', fmt(v, KIB, "KiB"), 150, 255, 150),
            Self::Removal(v) => ('-', format!("{v} B  "), 200, 200, 200),

            Self::NoChange => ('±', "0 B  ".to_string(), 120, 120, 120),
        };
        let color = Color::TrueColor { r, g, b };

        (sign, fmt_str, color)
    }

    /// Treat the supplied number like the fractional part of a decimal and
    /// round it appropriately, resulting in a number with the requested
    /// number of digits.
    ///
    /// Should only be used with inputs that are sufficiently far from
    /// `u64::MAX`.
    ///
    /// Examples:
    /// - `round_decimal_fraction::<1>(9949) = 1`
    /// - `round_decimal_fraction::<2>(9949) = 99`
    /// - `round_decimal_fraction::<3>(9949) = 995`
    /// - `round_decimal_fraction::<4>(9949) = 9949`
    /// - `round_decimal_fraction::<5>(9949) = 99490`
    ///
    /// For more examples, see the corresponding test below.
    fn round_decimal_fraction<const NUM_DIGITS: u32>(n: u64) -> u64 {
        assert_ne!(0, NUM_DIGITS, "discard the input instead");
        assert!(NUM_DIGITS < 10, "check for potential overflows");

        if n == 0 {
            return 0;
        }

        let n_digits = n.ilog10() + 1;
        match n_digits.cmp(&NUM_DIGITS) {
            Ordering::Equal => return n, // already done
            Ordering::Less => return n * 10_u64.pow(NUM_DIGITS - n_digits), // tack on 0s
            Ordering::Greater => (),     // the interesting case
        };

        let divisor = 10_u64.pow(n_digits - NUM_DIGITS);
        let half_divisor = divisor / 2;
        let rounded_prefix = (n + half_divisor) / divisor;

        // Check if rounded_prefix "spilled over" to 10^NUM_DIGITS, meaning it
        // has NUM_DIGITS+1 digits and its value is the smallest NUM_DIGITS+1
        // digit number.
        //
        // Example: NUM_DIGITS=1, rounded_prefix=10  => 1 (10^(1-1))
        // Example: NUM_DIGITS=2, rounded_prefix=100 => 10 (10^(2-1))
        if rounded_prefix == 10_u64.pow(NUM_DIGITS) {
            10_u64.pow(NUM_DIGITS - 1)
        } else {
            rounded_prefix
        }
    }
}

/// The internal profiler to measure the performance of various, user-specified
/// sub-(sub-(sub-…)tasks
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct VMPerformanceProfiler {
    name: String,
    timer: Instant,

    /// The maximum “resident set size” (RSS), i.e., the maximum amount of RAM
    /// used by this process during execution. In bytes.
    ///
    /// This value cannot be obtained on all systems, and is generally
    /// inaccurate due to system limitations.
    max_rss: Option<u64>,

    /// An index into the `profile`. Keeps track of currently running tasks.
    active_tasks: Vec<usize>,

    /// Tracks all tasks ever started, in the order they were started. Mapping
    /// from [`InvocationPath`] to [`Task`] allows accumulating time spent
    /// in loops.
    profile: IndexMap<InvocationPath, Task>,
}

impl VMPerformanceProfiler {
    pub fn new(name: impl Into<String>) -> Self {
        VMPerformanceProfiler {
            name: name.into(),
            timer: Instant::now(),
            max_rss: resident_set_size(),
            active_tasks: vec![],
            profile: IndexMap::new(),
        }
    }

    fn younger_sibling_indices(&self, index: usize) -> Vec<usize> {
        let parent_index = self.profile[index].parent_index;
        self.profile
            .values()
            .enumerate()
            .filter(|&(idx, _)| idx > index) // younger…
            .filter(|(_, task)| task.parent_index == parent_index) // …sibling
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Terminate the profiling session and generate a profiling report.
    pub fn finish(mut self) -> VMPerformanceProfile {
        let total_time = self.timer.elapsed();
        let max_rss = self.max_rss;

        for &i in &self.active_tasks {
            self.profile[i].name.push_str(" (unfinished)");
        }

        for _ in 0..self.active_tasks.len() {
            self.unconditional_stop();
        }

        // todo: this can count the same category multiple times if it's nested
        let mut category_times = HashMap::new();
        for task in self.profile.values() {
            if let Some(ref category) = task.category {
                category_times
                    .entry(category.to_string())
                    .or_insert(Duration::ZERO)
                    .add_assign(task.total_duration);
            }
        }

        let mut profile: Vec<TaskReport> = vec![];
        for (task_index, task) in self.profile.values().enumerate() {
            let relative_time = task.total_duration.as_secs_f64() / total_time.as_secs_f64();
            let weight =
                Weight::weigh(task.total_duration.as_secs_f64() / total_time.as_secs_f64());

            let num_invocations = std::cmp::max(task.num_invocations, 1);
            let mean_start_rss = task.start_rss.map(|rss| rss / num_invocations);
            let mean_stop_rss = task.stop_rss.map(|rss| rss / num_invocations);
            let rss_contribution = RssContribution::from_diff(mean_start_rss, mean_stop_rss);

            let mut ancestors = vec![];
            let mut current_ancestor_index = task.parent_index;
            while let Some(idx) = current_ancestor_index {
                ancestors.push(idx);
                current_ancestor_index = profile[idx].ancestors.last().copied();
            }
            ancestors.reverse();

            let relative_category_time = task.category.as_ref().map(|category| {
                task.total_duration.as_secs_f64() / category_times[category].as_secs_f64()
            });
            let is_last_sibling = self.younger_sibling_indices(task_index).is_empty();

            profile.push(TaskReport {
                name: task.name.clone(),
                depth: task.depth,
                duration: task.total_duration,
                rss_contribution,
                num_invocations,
                relative_time,
                category: task.category.clone(),
                relative_category_time,
                is_last_sibling,
                ancestors,
                weight,
                younger_max_weight: Weight::LikeNothing,
            });
        }

        for task_index in 0..profile.len() {
            profile[task_index].younger_max_weight = self
                .younger_sibling_indices(task_index)
                .into_iter()
                .map(|sibling_idx| profile[sibling_idx].weight)
                .max()
                .unwrap_or(Weight::LikeNothing);
        }

        VMPerformanceProfile {
            tasks: profile,
            name: self.name.clone(),
            total_time,
            max_rss,
            category_times,
            cycle_count: None,
            padded_height: None,
            fri_domain_len: None,
        }
    }

    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    pub fn start(
        &mut self,
        name: impl Into<String> + Clone,
        location: CodeLocation,
        category: Option<String>,
    ) {
        let rss = resident_set_size();
        self.max_rss = std::cmp::max(self.max_rss, rss);

        if env_var(ENV_VAR_PROFILER_LIVE_UPDATE).is_ok() {
            let name = name.clone().into();
            let rss = rss.map_or_else(|| "unknown".to_string(), |r| r.to_string());
            println!("start: {name} (at {location}) current memory usage: {rss} bytes");
        }

        let parent_index = self.active_tasks.last().copied();
        let path = match parent_index.map(|i| self.profile.get_index(i).unwrap()) {
            Some((path, _)) => path.extend(location),
            None => InvocationPath::default().extend(location),
        };

        let new_task = || Task {
            name: name.into(),
            parent_index,
            depth: self.active_tasks.len(),
            start_time: Instant::now(),
            start_rss: Some(0),
            stop_rss: Some(0),
            num_invocations: 0,
            total_duration: Duration::ZERO,
            category,
        };
        let new_task = self.profile.entry(path.clone()).or_insert_with(new_task);
        new_task.start_time = Instant::now();
        new_task.num_invocations += 1;
        new_task.start_rss = if let (Some(acc), Some(rss)) = (new_task.start_rss, rss) {
            acc.checked_add(rss)
        } else {
            None
        };

        let new_task_index = self.profile.get_index_of(&path).unwrap();
        self.active_tasks.push(new_task_index);
    }

    /// Stops the least recently started task if that is the expected task.
    ///
    /// # Panics
    ///
    /// Panics if the expected task is not the least recently started task.
    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    pub fn stop(&mut self, name: &str) {
        let top_task = self.active_tasks.last();
        let &top_index = top_task.expect("some task should be active in order to be stopped");
        let top_name = &self.profile[top_index].name;
        assert_eq!(top_name, name, "can't stop tasks in disorder");
        self.unconditional_stop();
    }

    fn unconditional_stop(&mut self) {
        let Some(index) = self.active_tasks.pop() else {
            return;
        };
        let task = &mut self.profile[index];
        let duration = task.start_time.elapsed();
        task.total_duration += duration;

        let rss = resident_set_size();
        self.max_rss = std::cmp::max(self.max_rss, rss);
        task.stop_rss =
            if let (Some(_), Some(acc), Some(rss)) = (task.start_rss, task.stop_rss, rss) {
                acc.checked_add(rss)
            } else {
                None
            };

        if env_var(ENV_VAR_PROFILER_LIVE_UPDATE).is_ok() {
            let name = &task.name;
            let rss = rss.map_or_else(|| "unknown".to_string(), |r| r.to_string());
            println!("stop:  {name} – took {duration:.2?} current memory usage: {rss} bytes");
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum Weight {
    LikeNothing,
    VeryLittle,
    Light,
    Noticeable,
    Heavy,
    Massive,
    SuperMassive,
}

impl Weight {
    /// Assign a weight based on a relative cost, which is a number between 0
    /// and 1.
    fn weigh(relative_cost: f64) -> Weight {
        match relative_cost {
            rc if rc >= 0.4 => Weight::SuperMassive,
            rc if rc >= 0.3 => Weight::Massive,
            rc if rc >= 0.2 => Weight::Heavy,
            rc if rc >= 0.1 => Weight::Noticeable,
            rc if rc >= 0.07 => Weight::Light,
            rc if rc >= 0.04 => Weight::VeryLittle,
            _ => Weight::LikeNothing,
        }
    }

    fn color(self) -> Color {
        let [r, g, b] = match self {
            Self::LikeNothing => [120; 3],
            Self::VeryLittle => [200; 3],
            Self::Light => [255; 3],
            Self::Noticeable => [255, 255, 120],
            Self::Heavy => [255, 150, 0],
            Self::Massive => [255, 75, 0],
            Self::SuperMassive => [255, 0, 0],
        };

        Color::TrueColor { r, g, b }
    }
}

#[derive(Debug, Clone)]
struct TaskReport {
    name: String,
    depth: usize,
    duration: Duration,
    rss_contribution: Option<RssContribution>,
    num_invocations: u64,
    relative_time: f64,
    category: Option<String>,
    relative_category_time: Option<f64>,
    is_last_sibling: bool,

    /// The direct parent is the `.last()` ancestor.
    ancestors: Vec<usize>,
    weight: Weight,
    younger_max_weight: Weight,
}

#[derive(Debug, Clone)]
pub struct VMPerformanceProfile {
    name: String,
    tasks: Vec<TaskReport>,
    total_time: Duration,

    /// Largest measured resident set size, in bytes.
    max_rss: Option<u64>,

    category_times: HashMap<String, Duration>,
    cycle_count: Option<usize>,
    padded_height: Option<usize>,
    fri_domain_len: Option<usize>,
}

impl VMPerformanceProfile {
    #[must_use]
    pub fn with_cycle_count(mut self, cycle_count: usize) -> Self {
        self.cycle_count = Some(cycle_count);
        self
    }

    #[must_use]
    pub fn with_padded_height(mut self, padded_height: usize) -> Self {
        self.padded_height = Some(padded_height);
        self
    }

    #[must_use]
    pub fn with_fri_domain_len(mut self, fri_domain_len: usize) -> Self {
        self.fri_domain_len = Some(fri_domain_len);
        self
    }

    fn display_time_aligned(time: Duration) -> String {
        let unaligned_time = format!("{time:.2?}");
        let time_components: Vec<_> = unaligned_time.split('.').collect();
        if time_components.len() != 2 {
            return unaligned_time;
        }

        format!("{:>3}.{:<4}", time_components[0], time_components[1])
    }
}

impl Default for VMPerformanceProfile {
    fn default() -> Self {
        let name = if cfg!(feature = "no_profile") {
            "The performance profiler is disabled through feature `no_profile`."
        } else {
            "Performance Profile"
        };

        Self {
            name: name.to_string(),
            tasks: vec![],
            total_time: Duration::default(),
            max_rss: None,
            category_times: HashMap::default(),
            cycle_count: None,
            padded_height: None,
            fri_domain_len: None,
        }
    }
}

impl Display for VMPerformanceProfile {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let max_name_width = self
            .tasks
            .iter()
            .map(|t| t.name.width() + 2 * t.depth)
            .max()
            .unwrap_or_default();
        let num_reps_width = self
            .tasks
            .iter()
            .map(|t| t.num_invocations.ilog10() as usize)
            .max()
            .unwrap_or_default()
            .max(4);
        let max_category_width = self
            .category_times
            .keys()
            .map(|k| k.width())
            .max()
            .unwrap_or(0);

        let title = format!("### {}", self.name).bold();
        let max_width = max(max_name_width, title.width());
        let title = format!("{title:<max_width$}");

        let total_time = Self::display_time_aligned(self.total_time).bold();
        let num_reps = format!("{:>num_reps_width$}", "#Reps").bold();
        let share_title = "Share".bold();
        let category_title = "Category".bold();
        let category_title_width = max_category_width + 11; // additional chars: "( - xx.xx%)"
        let rss_title = if let Some(rss) = self.max_rss {
            let (sign, rss_fmt, _) = RssContribution::Addition(rss).formatting_parts();
            let sign = if sign == '+' { ' ' } else { sign };
            format!("{sign}{rss_fmt}").bold()
        } else {
            "? KiB".dimmed()
        };
        let rss_width = 10; // e.g., +456.7 GiB

        write!(f, "{title}")?;
        write!(f, "     {total_time}")?;
        write!(f, "   {num_reps}")?;
        write!(f, "   {share_title}")?;
        write!(f, "  {category_title:<category_title_width$}")?;
        writeln!(f, "  {rss_title:>rss_width$}")?;

        for task in &self.tasks {
            for &ancestor_index in &task.ancestors {
                let ancestor = &self.tasks[ancestor_index];
                let spacer_color = ancestor.younger_max_weight.color();
                let is_last_sibling = ancestor.is_last_sibling;
                let spacer = if is_last_sibling { "  " } else { "│ " }.color(spacer_color);
                write!(f, "{spacer}")?;
            }
            let is_last_sibling = task.is_last_sibling;
            let tracer = if is_last_sibling { "└" } else { "├" }
                .color(max(task.weight, task.younger_max_weight).color());
            let dash = "─".color(task.weight.color());

            let task_name_area = max_width - 2 * task.depth;
            let task_name = task.name.color(task.weight.color());
            let task_name = format!("{task_name:<task_name_area$}");
            let task_time = format!("{:<10}", Self::display_time_aligned(task.duration))
                .color(task.weight.color());
            let num_iterations =
                format!("{:>num_reps_width$}", task.num_invocations).color(task.weight.color());
            let relative_time_string =
                format!("{:5.2}%", 100.0 * task.relative_time).color(task.weight.color());
            let relative_category_time = task
                .relative_category_time
                .map(|t| format!("{:6.2}%", 100.0 * t))
                .unwrap_or_default();

            let relative_category_color = task
                .relative_category_time
                .map_or(Color::White, |t| Weight::weigh(t).color());
            let category_and_relative_time = task
                .category
                .as_ref()
                .map(|cat| format!("({cat:<max_category_width$} –{relative_category_time})"))
                .unwrap_or_default()
                .color(relative_category_color);

            let rss_contribution = task
                .rss_contribution
                .map(|c| c.format_colored_aligned())
                .unwrap_or_default();

            write!(f, "{tracer}{dash}")?;
            write!(f, "{task_name}")?;
            write!(f, "   {task_time}")?;
            write!(f, "  {num_iterations}")?;
            write!(f, "  {relative_time_string}")?;
            write!(f, "  {category_and_relative_time:<category_title_width$}")?;
            writeln!(f, "  {rss_contribution:>rss_width$}")?;
        }

        if !self.category_times.is_empty() {
            writeln!(f, "\n{}", "### Categories".bold())?;
        }
        for (category, &category_time) in self
            .category_times
            .iter()
            .sorted_by_key(|&(_, &time)| time)
            .rev()
        {
            let relative_time = category_time.as_secs_f64() / self.total_time.as_secs_f64();
            let color = Weight::weigh(relative_time).color();
            let relative_time = format!("{:>6}", format!("{:2.2}%", 100.0 * relative_time));
            let category_time = Self::display_time_aligned(category_time);

            let category = format!("{category:<max_category_width$}").color(color);
            let category_time = category_time.color(color);
            let category_relative_time = relative_time.color(color);
            writeln!(f, "{category}   {category_time} {category_relative_time}")?;
        }

        let optionals = [self.cycle_count, self.padded_height, self.fri_domain_len];
        if optionals.iter().all(Option::is_none) {
            return Ok(());
        }
        let Ok(total_time) = usize::try_from(self.total_time.as_millis()) else {
            return writeln!(f, "WARN: Total time too large to compute frequency.");
        };
        if total_time == 0 {
            return writeln!(f, "WARN: Total time too small to compute frequency.");
        }
        let tasks = self.tasks.iter();
        let num_iters = tasks.map(|t| t.num_invocations).min().unwrap_or(1);
        let Ok(num_iters) = usize::try_from(num_iters) else {
            return Ok(());
        };

        writeln!(f)?;
        if let Some(cycles) = self.cycle_count {
            let frequency = 1_000 * num_iters * cycles / total_time;
            write!(f, "Clock frequency is {frequency} Hz ")?;
            write!(f, "({cycles} clock cycles ")?;
            write!(f, "/ ({total_time} ms ")?;
            writeln!(f, "/ {num_iters} iterations))")?;
        }

        if let Some(height) = self.padded_height {
            let frequency = 1_000 * num_iters * height / total_time;
            write!(f, "Optimal clock frequency is {frequency} Hz ")?;
            write!(f, "({height} padded height ")?;
            write!(f, "/ ({total_time} ms ")?;
            writeln!(f, "/ {num_iters} iterations))")?;
        }

        if let Some(fri_domain_length) = self.fri_domain_len {
            let log_2_fri_domain_length = fri_domain_length.checked_ilog2().unwrap_or(0);
            writeln!(f, "FRI domain length is 2^{log_2_fri_domain_length}")?;
        }

        Ok(())
    }
}

/// The “resident set size” (RSS) in bytes.
///
/// The RSS (on windows, “working set”) is the total amount of memory in use by
/// the current process. Generally, this metric is not completely accurate. On
/// some platforms, it cannot even be obtained.
#[cfg(any(debug_assertions, not(feature = "no_profile")))]
fn resident_set_size() -> Option<u64> {
    let stats = memory_stats::memory_stats()?;
    let rss = stats.physical_mem.try_into().ok()?;

    Some(rss)
}

/// The “resident set size” (RSS) in bytes. Disabled in this build, probably
/// because of feature [`no_profile`](crate::profiler).
#[cfg(not(any(debug_assertions, not(feature = "no_profile"))))]
#[inline(always)]
fn resident_set_size() -> Option<u64> {
    None
}

/// Start or stop a profiling task. Does nothing if the profiler is not running;
/// see [`start`].
///
/// For starting, use `start` as the first token. For stopping, use `stop`.
/// When starting, an optional category can be provided in parentheses.
/// When stopping, the task's name needs to be an exact match to prevent the
/// accidental stopping of a different task.
///
/// # Examples
///
/// ```no_compile
/// # // The profiler macros are internal only and cannot be used in doc tests.
/// use crate::profiler::profiler;
/// crate::profiler::start("heavy lifting");
/// profiler!(start "good job" ("compute")); // uses the category “compute”
/// for _ in 0..5 {
///     profiler!(start "loop"); // category is optional
///     /* work hard */
///     profiler!(stop "loop"); // must match exactly
/// }
/// profiler!(stop "good job"); // category is inferred when stopping
/// let profile = crate::profiler::finish();
/// ```
macro_rules! profiler {
    (start $task:literal ($category:literal)) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        $crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                let here = $crate::profiler::here!();
                profiler.start($task, here, Some($category.to_string()));
            }
        })
    }};
    (start $task:literal) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        $crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.start($task, $crate::profiler::here!(), None);
            }
        })
    }};
    (stop $task:literal) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        $crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.stop($task);
            }
        })
    }};
}
pub(crate) use profiler;

#[cfg(all(test, any(debug_assertions, not(feature = "no_profile"))))]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use test_strategy::proptest;

    use super::*;

    #[test]
    fn formatting_parts() {
        let (sign, fmt_str, _) = RssContribution::Addition(1_024).formatting_parts();
        assert_eq!('+', sign);
        assert_eq!("1.0 KiB", fmt_str);

        let (sign, fmt_str, _) = RssContribution::Removal(37_258_841_293).formatting_parts();
        assert_eq!('-', sign);
        assert_eq!("34.7 GiB", fmt_str);

        let (_, fmt_str, _) = RssContribution::Addition(1_000).formatting_parts();
        assert_eq!("0.1 KiB", fmt_str);

        let (_, fmt_str, _) = RssContribution::Addition(1023 * 1_024).formatting_parts();
        assert_eq!("0.1 MiB", fmt_str);
    }

    #[test]
    fn round_decimal_fraction() {
        assert_eq!(9, RssContribution::round_decimal_fraction::<1>(9));
        assert_eq!(9, RssContribution::round_decimal_fraction::<1>(94));
        assert_eq!(1, RssContribution::round_decimal_fraction::<1>(95));

        assert_eq!(90, RssContribution::round_decimal_fraction::<2>(9));
        assert_eq!(94, RssContribution::round_decimal_fraction::<2>(94));
        assert_eq!(95, RssContribution::round_decimal_fraction::<2>(95));
        assert_eq!(99, RssContribution::round_decimal_fraction::<2>(9949));
        assert_eq!(10, RssContribution::round_decimal_fraction::<2>(9950));

        assert_eq!(900, RssContribution::round_decimal_fraction::<3>(9));
        assert_eq!(940, RssContribution::round_decimal_fraction::<3>(94));
        assert_eq!(950, RssContribution::round_decimal_fraction::<3>(95));
    }

    #[test]
    fn sanity() {
        let mut profiler = VMPerformanceProfiler::new("Sanity Test");
        profiler.start("Task 0", here!(), None);
        sleep(Duration::from_millis(1));
        profiler.start("Task 1", here!(), Some("setup".to_string()));
        for _ in 0..5 {
            profiler.start("Task 2", here!(), Some("compute".to_string()));
            sleep(Duration::from_millis(1));
            for _ in 0..3 {
                profiler.start("Task 3", here!(), Some("cleanup".to_string()));
                sleep(Duration::from_millis(1));
                profiler.stop("Task 3");
            }
            profiler.stop("Task 2");
        }
        profiler.stop("Task 1");
        profiler.start("Task 4", here!(), Some("cleanup".to_string()));
        sleep(Duration::from_millis(1));
        profiler.stop("Task 4");
        profiler.stop("Task 0");
        profiler.start("Task 5", here!(), None);
        sleep(Duration::from_millis(1));
        profiler.start("Task 6", here!(), Some("setup".to_string()));
        sleep(Duration::from_millis(1));
        profiler.stop("Task 6");
        profiler.stop("Task 5");
        let profile = profiler.finish();
        println!("{profile}");
    }

    #[derive(Debug, Clone, Eq, PartialEq, Hash, test_strategy::Arbitrary)]
    enum DispatchChoice {
        Function0,
        Function1,
        FunctionWithLoops,
        FunctionWithNestedLoop,
        Dispatch,
    }

    #[proptest]
    fn extensive(mut choices: Vec<DispatchChoice>) {
        fn dispatch(choice: DispatchChoice, remaining_choices: &mut Vec<DispatchChoice>) {
            profiler!(start "dispatcher");
            match choice {
                DispatchChoice::Function0 => function_0(),
                DispatchChoice::Function1 => function_1(),
                DispatchChoice::FunctionWithLoops => function_with_loops(),
                DispatchChoice::FunctionWithNestedLoop => function_with_nested_loop(),
                DispatchChoice::Dispatch => {
                    if let Some(choice) = remaining_choices.pop() {
                        dispatch(choice, remaining_choices)
                    }
                }
            }
            profiler!(stop "dispatcher");
        }

        fn function_0() {
            profiler!(start "function_0");
            sleep(Duration::from_micros(1));
            profiler!(stop "function_0");
        }

        fn function_1() {
            profiler!(start "function_1" ("setup"));
            sleep(Duration::from_micros(1));
            profiler!(stop "function_1");
        }

        fn function_with_loops() {
            for _ in 0..5 {
                profiler!(start "function_with_loops" ("compute"));
                sleep(Duration::from_micros(1));
                profiler!(stop "function_with_loops");
            }
        }

        fn function_with_nested_loop() {
            for _ in 0..5 {
                profiler!(start "function_with_nested_loop" ("outer loop"));
                for _ in 0..3 {
                    profiler!(start "function_with_nested_loop" ("inner loop"));
                    sleep(Duration::from_micros(1));
                    profiler!(stop "function_with_nested_loop");
                }
                profiler!(stop "function_with_nested_loop");
            }
        }

        crate::profiler::start("Extensive Test");
        while let Some(choice) = choices.pop() {
            dispatch(choice, &mut choices);
        }
        let profile = crate::profiler::finish();
        println!("{profile}");
    }

    #[test]
    fn clk_freq() {
        crate::profiler::start("Clock Frequency Test");
        profiler!(start "clk_freq_test");
        sleep(Duration::from_millis(3));
        profiler!(stop "clk_freq_test");
        let profile = crate::profiler::finish();

        let profile_with_no_optionals = profile.clone();
        println!("{profile_with_no_optionals}");

        let profile_with_optionals_set_to_0 = profile
            .clone()
            .with_cycle_count(0)
            .with_padded_height(0)
            .with_fri_domain_len(0);
        println!("{profile_with_optionals_set_to_0}");

        let profile_with_optionals_set = profile
            .clone()
            .with_cycle_count(10)
            .with_padded_height(12)
            .with_fri_domain_len(32);
        println!("{profile_with_optionals_set}");
    }

    #[test]
    fn starting_the_profiler_twice_does_not_cause_panic() {
        crate::profiler::start("Double Start Test 0");
        crate::profiler::start("Double Start Test 1");
        let profile = crate::profiler::finish();
        println!("{profile}");
    }

    #[test]
    fn creating_profile_without_starting_profile_does_not_cause_panic() {
        let profile = crate::profiler::finish();
        println!("{profile}");
    }

    #[test]
    fn profiler_without_any_tasks_can_generate_a_profile_report() {
        crate::profiler::start("Empty Test");
        let profile = crate::profiler::finish();
        println!("{profile}");
    }

    #[test]
    fn invocation_path_can_be_displayed() {
        let path = InvocationPath::default().extend(here!());
        let path = path.extend(here!());
        println!("{path}");
    }

    #[test]
    fn profiler_with_unfinished_tasks_can_generate_profile_report() {
        crate::profiler::start("Unfinished Tasks Test");
        profiler!(start "unfinished task");
        let profile = crate::profiler::finish();
        println!("{profile}");
    }

    #[test]
    fn loops() {
        crate::profiler::start("Loops");
        for i in 0..5 {
            profiler!(start "loop");
            println!("iteration {i}");
            profiler!(stop "loop");
        }
        let profile = crate::profiler::finish();
        println!("{profile}");
    }

    #[test]
    fn nested_loops() {
        crate::profiler::start("Nested Loops");
        for i in 0..5 {
            profiler!(start "outer loop");
            print!("outer loop iteration {i}, inner loop iteration");
            for j in 0..5 {
                profiler!(start "inner loop");
                print!(" {j}");
                profiler!(stop "inner loop");
            }
            println!();
            profiler!(stop "outer loop");
        }
        let profile = crate::profiler::finish();
        println!("{profile}");
    }
}
