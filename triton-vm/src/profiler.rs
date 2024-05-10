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
//! In release builds, profiling is disabled by default to allow for the fastest
//! possible proof generation. To enable profiling, either make sure that
//! `debug_assertions` is set, or add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! triton-vm = { version = "x.y.z", default-features = false }
//! ```
//!
//! ### A note on the `no_profile` feature design decision
//!
//! The feature `no_profile` _disables_ profiling, and is enabled by default.
//! This seems backwards. However, it is an expression of how Triton VM favors
//! performance over profiling. Note [how Cargo resolves dependencies][deps]:
//! if some dependency is transitively declared multiple times, the _union_ of
//! all features will be enabled.
//!
//! Imagine some dependency `foo` enables a hypothetical `do_profile` feature.
//! If another dependency `bar` requires the most efficient proof generation,
//! it would be slowed down by `foo` and could do nothing about it. Instead,
//! Disabling profiling by <i>en</i>abling the feature `no_profile` allows `bar`
//! to dictate. This:
//! 1. makes the profile reports of `foo` disappear, which is sad, but
//! 1. lets `bar` be fast, which is more important for Triton VM.
//!
//! [proving]: crate::stark::Stark::prove
//! [verifying]: crate::stark::Stark::verify
//! [deps]: https://doc.rust-lang.org/cargo/reference/features.html#feature-unification

use std::cell::RefCell;
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

use arbitrary::Arbitrary;
use colored::Color;
use colored::ColoredString;
use colored::Colorize;
use itertools::Itertools;
use unicode_width::UnicodeWidthStr;

const ENV_VAR_PROFILER_LIVE_UPDATE: &str = "TVM_PROFILER_LIVE_UPDATE";

thread_local! {
    pub(crate) static PROFILER: RefCell<Option<TritonProfiler>> = const { RefCell::new(None) };
}

/// Start profiling. If the profiler is already running, this function cancels
/// the current profiling session and starts a new one.
///
/// See the module-level documentation for information on how to enable profiling.
pub fn start(profile_name: impl Into<String>) {
    if cfg!(any(debug_assertions, not(feature = "no_profile"))) {
        PROFILER.replace(Some(TritonProfiler::new(profile_name)));
    }
}

/// Stop the current profiling session and generate a [`Report`]. If the
/// profiler is disabled or not running, an empty [`Report`] is returned.
///
/// See the module-level documentation for information on how to enable profiling.
pub fn finish() -> Report {
    cfg!(any(debug_assertions, not(feature = "no_profile")))
        .then(|| PROFILER.take().map(TritonProfiler::finish))
        .flatten()
        .unwrap_or_default()
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Task {
    name: String,
    parent_index: Option<usize>,
    depth: usize,
    time: Duration,
    task_type: TaskType,

    /// The type of work the task is doing. Helps to track time across specific tasks. For
    /// example, if the task is building a Merkle tree, then the category could be "hash".
    category: Option<String>,
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
enum TaskType {
    #[default]
    Generic,
    IterationZero,
    AnyOtherIteration,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct TritonProfiler {
    name: String,
    timer: Instant,

    /// An index into the `profile` vector. Keeps track of currently running tasks.
    active_tasks: Vec<usize>,
    profile: Vec<Task>,
}

impl TritonProfiler {
    pub fn new(name: impl Into<String>) -> Self {
        TritonProfiler {
            name: name.into(),
            timer: Instant::now(),
            active_tasks: vec![],
            profile: vec![],
        }
    }

    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    fn ignoring(&self) -> bool {
        self.active_tasks
            .last()
            .is_some_and(|&idx| self.profile[idx].task_type == TaskType::AnyOtherIteration)
    }

    fn younger_sibling_indices(&self, index: usize) -> Vec<usize> {
        self.profile
            .iter()
            .enumerate()
            .filter(|&(idx, _)| idx > index)
            .filter(|&(_, task)| task.parent_index == self.profile[index].parent_index)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Terminate the profiling session and generate a profiling report.
    pub fn finish(mut self) -> Report {
        for &i in &self.active_tasks {
            let task_name = &mut self.profile[i].name;
            task_name.push_str(" (unfinished)");
        }

        for _ in 0..self.active_tasks.len() {
            self.plain_stop();
        }

        let total_time = self.timer.elapsed();
        let mut report: Vec<TaskReport> = vec![];

        // todo: this can count the same category multiple times if it's nested
        let mut category_times = HashMap::new();
        for task in &self.profile {
            if let Some(ref category) = task.category {
                category_times
                    .entry(category.to_string())
                    .or_insert(Duration::ZERO)
                    .add_assign(task.time);
            }
        }

        for (task_index, task) in self.profile.iter().enumerate() {
            let relative_time = task.time.as_secs_f64() / total_time.as_secs_f64();
            let weight = match task.task_type {
                TaskType::AnyOtherIteration => Weight::LikeNothing,
                _ => Weight::weigh(task.time.as_secs_f64() / total_time.as_secs_f64()),
            };

            let mut ancestors = vec![];
            let mut current_ancestor_index = task.parent_index;
            while let Some(idx) = current_ancestor_index {
                ancestors.push(idx);
                current_ancestor_index = report[idx].parent_index;
            }
            ancestors.reverse();

            let relative_category_time = task
                .category
                .as_ref()
                .map(|category| task.time.as_secs_f64() / category_times[category].as_secs_f64());
            let is_last_sibling = self.younger_sibling_indices(task_index).is_empty();

            report.push(TaskReport {
                name: task.name.clone(),
                parent_index: task.parent_index,
                depth: task.depth,
                time: task.time,
                relative_time,
                category: task.category.clone(),
                relative_category_time,
                is_last_sibling,
                ancestors,
                weight,
                younger_max_weight: Weight::LikeNothing,
            });
        }

        for task_index in 0..report.len() {
            report[task_index].younger_max_weight = self
                .younger_sibling_indices(task_index)
                .into_iter()
                .map(|sibling_idx| report[sibling_idx].weight)
                .max()
                .unwrap_or(Weight::LikeNothing);
        }

        // “Other iterations” are not currently tracked
        let all_tasks = self.profile.iter();
        let other_iterations = all_tasks.filter(|t| t.task_type == TaskType::AnyOtherIteration);
        let untracked_time = other_iterations.map(|t| t.time).sum();

        Report {
            tasks: report,
            name: self.name.clone(),
            total_time,
            untracked_time,
            category_times,
            cycle_count: None,
            padded_height: None,
            fri_domain_len: None,
        }
    }

    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    pub fn start(&mut self, name: &str, category: Option<String>) {
        if !self.ignoring() {
            self.plain_start(name, TaskType::Generic, category);
        }
    }

    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    fn plain_start(
        &mut self,
        name: impl Into<String>,
        task_type: TaskType,
        category: Option<String>,
    ) {
        let name = name.into();
        let parent_index = self.active_tasks.last().copied();
        self.active_tasks.push(self.profile.len());

        if env_var(ENV_VAR_PROFILER_LIVE_UPDATE).is_ok() {
            println!("start: {name}");
        }

        self.profile.push(Task {
            name,
            parent_index,
            depth: self.active_tasks.len(),
            time: self.timer.elapsed(),
            task_type,
            category,
        });
    }

    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    pub fn iteration_zero(&mut self, name: &str, category: Option<String>) {
        if self.ignoring() {
            return;
        }

        let Some(&top_index) = self.active_tasks.last() else {
            panic!("Profiler stack is empty; can't iterate.");
        };
        let top_task = &self.profile[top_index];

        if top_task.task_type == TaskType::Generic {
            self.plain_start("iteration 0", TaskType::IterationZero, category);
            return;
        }

        let num_active_tasks = self.active_tasks.len();
        assert!(
            num_active_tasks >= 2,
            "To profile zeroth iteration, stack must be at least 2-high, \
            but got height of {num_active_tasks}"
        );

        let runner_up_index = self.active_tasks[num_active_tasks - 2];
        let runner_up_name = &self.profile[runner_up_index].name;
        dbg!(runner_up_index);

        assert_eq!(
            runner_up_name, name,
            "To profile zeroth iteration, name must match with top of stack."
        );

        if top_task.task_type == TaskType::IterationZero {
            // switch – stop iteration zero, start “all other iterations”
            self.plain_stop();
            self.plain_start(
                "all other iterations",
                TaskType::AnyOtherIteration,
                category,
            );
        }

        // top == *"all other iterations"
        // in this case we do nothing
    }

    fn plain_stop(&mut self) {
        let Some(index) = self.active_tasks.pop() else {
            return;
        };
        let duration = self.timer.elapsed() - self.profile[index].time;
        self.profile[index].time = duration;
        let name = &self.profile[index].name;

        if env_var(ENV_VAR_PROFILER_LIVE_UPDATE).is_ok() {
            println!("stop:  {name} – took {duration:.2?}");
        }
    }

    #[cfg(any(debug_assertions, not(feature = "no_profile")))]
    pub fn stop(&mut self, name: &str) {
        let top_task = self.active_tasks.last();
        let &top_index = top_task.expect("Cannot stop any tasks when stack is empty.");
        let top_name = &self.profile[top_index].name;

        if top_name == "iteration 0" || top_name == "all other iterations" {
            let num_active_tasks = self.active_tasks.len();
            let runner_up_index = self.active_tasks[num_active_tasks - 2];
            let runner_up_name = &self.profile[runner_up_index].name;
            if runner_up_name == name {
                self.plain_stop();
                self.plain_stop();
            }
        } else {
            assert_eq!(top_name, name, "can't stop tasks in disorder.");
            self.plain_stop();
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
    /// Assign a weight based on a relative cost, which is a number between 0 and 1.
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
    parent_index: Option<usize>,
    depth: usize,
    time: Duration,
    relative_time: f64,
    category: Option<String>,
    relative_category_time: Option<f64>,
    is_last_sibling: bool,
    ancestors: Vec<usize>,
    weight: Weight,
    younger_max_weight: Weight,
}

#[derive(Debug, Clone)]
pub struct Report {
    name: String,
    tasks: Vec<TaskReport>,
    total_time: Duration,
    untracked_time: Duration,
    category_times: HashMap<String, Duration>,
    cycle_count: Option<usize>,
    padded_height: Option<usize>,
    fri_domain_len: Option<usize>,
}

impl Report {
    pub fn with_cycle_count(mut self, cycle_count: usize) -> Self {
        self.cycle_count = Some(cycle_count);
        self
    }

    pub fn with_padded_height(mut self, padded_height: usize) -> Self {
        self.padded_height = Some(padded_height);
        self
    }

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

impl Default for Report {
    fn default() -> Self {
        let name = if cfg!(feature = "no_profile") {
            "Triton VM's profiler is disabled through feature `no_profile`.".to_string()
        } else {
            "__empty__".to_string()
        };

        Self {
            name,
            tasks: vec![],
            total_time: Duration::default(),
            untracked_time: Duration::default(),
            category_times: HashMap::default(),
            cycle_count: None,
            padded_height: None,
            fri_domain_len: None,
        }
    }
}

impl Display for Report {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let max_name_width = self
            .tasks
            .iter()
            .map(|t| t.name.width() + 2 * t.depth)
            .max()
            .unwrap_or_default();
        let max_category_width = self
            .category_times
            .keys()
            .map(|k| k.width())
            .max()
            .unwrap_or(0);

        let title = format!("### {}", self.name).bold();
        let max_width = max(max_name_width, title.width());
        let title = format!("{title:<max_width$}");

        let tracked_time = self.total_time.saturating_sub(self.untracked_time);
        let tracked = Self::display_time_aligned(tracked_time);
        let untracked = Self::display_time_aligned(self.untracked_time);
        writeln!(f, "tracked {tracked}, leaving {untracked} untracked")?;

        let total_time = Self::display_time_aligned(self.total_time).bold();
        let share_title = "Share".to_string().bold();
        let category_title = if self.category_times.is_empty() {
            ColoredString::default()
        } else {
            "Category".bold()
        };
        writeln!(
            f,
            "{title}   {total_time}   {share_title}  {category_title}"
        )?;

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
            write!(f, "{tracer}{dash}")?;

            let task_name_area = max_width - 2 * task.depth;
            let task_name_colored = task.name.color(task.weight.color());
            let task_name_colored = format!("{task_name_colored:<task_name_area$}");
            let task_time = format!("{:<10}", Self::display_time_aligned(task.time));
            let task_time_colored = task_time.color(task.weight.color());
            let relative_time_string =
                format!("{:>6}", format!("{:2.2}%", 100.0 * task.relative_time));
            let relative_time_string_colored = relative_time_string.color(task.weight.color());
            let relative_category_time = task
                .relative_category_time
                .map(|t| format!("{:>6}", format!("{:2.2}%", 100.0 * t)))
                .unwrap_or_default();
            let maybe_category = task.category.as_ref();
            let category_and_relative_time = maybe_category
                .map(|cat| format!("({cat:<max_category_width$} – {relative_category_time})"))
                .unwrap_or_default();

            let relative_category_color = task
                .relative_category_time
                .map_or(Color::White, |t| Weight::weigh(t).color());
            let category_and_relative_time_colored =
                category_and_relative_time.color(relative_category_color);

            writeln!(
                f,
                "{task_name_colored}   {task_time_colored}\
                 {relative_time_string_colored} {category_and_relative_time_colored}"
            )?;
        }

        if !self.category_times.is_empty() {
            writeln!(f)?;
            let category_title = "### Categories".bold();
            writeln!(f, "{category_title}")?;
            let category_times_and_names_sorted_by_time = self
                .category_times
                .iter()
                .sorted_by_key(|(_, &time)| time)
                .rev();
            for (category, &category_time) in category_times_and_names_sorted_by_time {
                let category_relative_time =
                    category_time.as_secs_f64() / self.total_time.as_secs_f64();
                let category_color = Weight::weigh(category_relative_time).color();
                let category_relative_time =
                    format!("{:>6}", format!("{:2.2}%", 100.0 * category_relative_time));
                let category_time = Self::display_time_aligned(category_time);

                let category = format!("{category:<max_category_width$}").color(category_color);
                let category_time = category_time.color(category_color);
                let category_relative_time = category_relative_time.color(category_color);
                writeln!(f, "{category}   {category_time} {category_relative_time}")?;
            }
        }

        let Ok(total_time) = usize::try_from(self.total_time.as_millis()) else {
            return writeln!(f, "WARN: Total time too large. Cannot compute frequency.");
        };
        if total_time == 0 {
            return writeln!(f, "WARN: Total time is zero. Cannot compute frequency.");
        }

        if self.cycle_count.is_some()
            || self.padded_height.is_some()
            || self.fri_domain_len.is_some()
        {
            writeln!(f)?;
        }

        if let Some(cycle_count) = self.cycle_count {
            let frequency = 1_000 * cycle_count / total_time;
            write!(f, "Clock frequency is {frequency} Hz ")?;
            writeln!(f, "({cycle_count} clock cycles / {total_time} ms)",)?;
        }

        if let Some(padded_height) = self.padded_height {
            let frequency = 1_000 * padded_height / total_time;
            write!(f, "Optimal clock frequency is {frequency} Hz ")?;
            writeln!(f, "({padded_height} padded height / {total_time} ms)")?;
        }

        if let Some(fri_domain_length) = self.fri_domain_len {
            let log_2_fri_domain_length = fri_domain_length.checked_ilog2().unwrap_or(0);
            writeln!(f, "FRI domain length is 2^{log_2_fri_domain_length}")?;
        }

        Ok(())
    }
}

/// Start a profiling task. Does nothing if the profiler is not running;
/// see [`start`].
///
/// The first argument is the name of the task.
/// The second, optional argument is a task category.
macro_rules! prof_start {
    ($s:expr, $c:expr) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.start($s, Some($c.to_string()));
            }
        })
    }};
    ($s:expr) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.start($s, None);
            }
        })
    }};
}
pub(crate) use prof_start;

/// Stop a profiling task.
///
/// Requires the same arguments as [`prof_start`], except that the task's
/// category (if any) is inferred. Notably, the task's name needs to be an exact
/// match to prevent the accidental stopping of a different task.
macro_rules! prof_stop {
    ($s:expr) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.stop($s);
            }
        })
    }};
}
pub(crate) use prof_stop;

/// Profile one iteration of a loop. Requires the same arguments as
/// [`prof_start`].
///
/// This macro should be invoked inside the loop in question. The profiling of
/// the loop has to be stopped with [`prof_stop`] after the loop.
macro_rules! prof_itr0 {
    ($s:expr, $c:expr) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.iteration_zero($s, Some($c.to_string()));
            }
        })
    }};
    ($s:expr) => {{
        #[cfg(any(debug_assertions, not(feature = "no_profile")))]
        crate::profiler::PROFILER.with_borrow_mut(|profiler| {
            if let Some(profiler) = profiler.as_mut() {
                profiler.iteration_zero($s, None);
            }
        })
    }};
}
pub(crate) use prof_itr0;

#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use rand::rngs::ThreadRng;
    use rand::Rng;
    use rand::RngCore;

    use super::*;

    fn random_task_name(rng: &mut ThreadRng) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz_";
        let length = rng.gen_range(2..12);

        (0..length)
            .map(|_| (rng.next_u32() as usize) % alphabet.len())
            .map(|i| alphabet.get(i..=i).unwrap())
            .collect()
    }

    fn random_category(rng: &mut ThreadRng) -> String {
        let categories = ["setup", "compute", "drop", "cleanup"];
        let num_categories = categories.len();
        let category_index = rng.gen_range(0..num_categories);
        let category = categories[category_index];
        category.to_string()
    }

    #[test]
    fn sanity() {
        let mut rng = rand::thread_rng();
        let mut stack = vec![];
        let steps = 100;

        let mut profiler = TritonProfiler::new("Sanity Test");

        for step in 0..steps {
            let steps_left = steps - step;
            assert!((1..=steps).contains(&steps_left));
            assert!(
                stack.len() <= steps_left,
                "stack len {} > steps left {}",
                stack.len(),
                steps_left
            );

            let pushable = rng.next_u32() % 2 == 1 && stack.len() + 1 < steps_left;
            let poppable = ((!pushable && rng.next_u32() % 2 == 1)
                || (stack.len() == steps_left && steps_left > 0))
                && !stack.is_empty();

            if pushable {
                let name = random_task_name(&mut rng);
                let category = match rng.gen() {
                    true => Some(random_category(&mut rng)),
                    false => None,
                };
                stack.push(name.clone());
                profiler.start(&name, category);
            }

            sleep(Duration::from_micros(
                (rng.next_u64() % 10) * (rng.next_u64() % 10) * (rng.next_u64() % 10),
            ));

            if poppable {
                let name = stack.pop().unwrap();
                profiler.stop(&name);
            }
        }

        println!("{}", profiler.finish());
    }

    #[test]
    fn clk_freq() {
        crate::profiler::start("Clock Frequency Test");
        prof_start!("clk_freq_test");
        sleep(Duration::from_millis(3));
        prof_stop!("clk_freq_test");
        let report = crate::profiler::finish();

        let report_with_no_optionals = report.clone();
        println!("{report_with_no_optionals}");

        let report_with_optionals_set_to_0 = report
            .clone()
            .with_cycle_count(0)
            .with_padded_height(0)
            .with_fri_domain_len(0);
        println!("{report_with_optionals_set_to_0}");

        let report_with_optionals_set = report
            .clone()
            .with_cycle_count(10)
            .with_padded_height(12)
            .with_fri_domain_len(32);
        println!("{report_with_optionals_set}");
    }

    #[test]
    fn macros() {
        crate::profiler::start("Macro Test");
        let mut rng = rand::thread_rng();
        let mut stack = vec![];
        let steps = 100;

        for steps_left in (0..steps).rev() {
            let pushable = rng.gen() && stack.len() + 1 < steps_left;
            let poppable = ((!pushable && rng.gen())
                || (stack.len() == steps_left && steps_left > 0))
                && !stack.is_empty();

            if pushable {
                let name = random_task_name(&mut rng);
                let category = random_category(&mut rng);
                stack.push(name.clone());
                if rng.gen() {
                    prof_start!(&name, category)
                } else {
                    prof_start!(&name)
                }
            }

            let num_micros = rng.gen_range(1..=10) * rng.gen_range(1..=10) * rng.gen_range(1..=10);
            sleep(Duration::from_micros(num_micros));

            if poppable {
                let name = stack.pop().unwrap();
                prof_stop!(&name);
            }
        }

        println!("{}", crate::profiler::finish());
    }

    #[test]
    fn starting_the_profiler_twice_does_not_cause_panic() {
        crate::profiler::start("Double Start Test 0");
        crate::profiler::start("Double Start Test 1");
        let report = crate::profiler::finish();
        println!("{report}");
    }

    #[test]
    fn creating_report_without_starting_profile_does_not_cause_panic() {
        let report = crate::profiler::finish();
        println!("{report}");
    }

    #[test]
    fn profiler_without_any_tasks_can_generate_a_report() {
        crate::profiler::start("Empty Test");
        let report = crate::profiler::finish();
        println!("{report}");
    }

    #[test]
    fn profiler_with_unfinished_tasks_can_generate_report() {
        crate::profiler::start("Unfinished Tasks Test");
        prof_start!("unfinished task");
        let report = crate::profiler::finish();
        println!("{report}");
    }
}
