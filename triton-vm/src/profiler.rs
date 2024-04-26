use std::cmp::max;
use std::collections::HashMap;
use std::env::var as env_var;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::AddAssign;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use arbitrary::Arbitrary;
use colored::Color;
use colored::ColoredString;
use colored::Colorize;
use criterion::profiler::Profiler;
use itertools::Itertools;
use unicode_width::UnicodeWidthStr;

const GET_PROFILE_OUTPUT_AS_YOU_GO_ENV_VAR_NAME: &str = "PROFILE_AS_YOU_GO";

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
pub struct TritonProfiler {
    name: String,
    timer: Instant,
    stack: Vec<(usize, String)>,
    profile: Vec<Task>,
    total_time: Option<Duration>,
}

impl TritonProfiler {
    pub fn new(name: &str) -> Self {
        TritonProfiler {
            name: name.to_owned(),
            timer: Instant::now(),
            stack: vec![],
            profile: vec![],
            total_time: None,
        }
    }

    fn ignoring(&self) -> bool {
        if let Some(top) = self.stack.last() {
            if self.profile[top.0].task_type == TaskType::AnyOtherIteration {
                return true;
            }
        }
        false
    }

    fn has_younger_sibling(&self, index: usize) -> bool {
        for (idx, task) in self.profile.iter().enumerate() {
            if idx == index {
                continue;
            }
            if task.parent_index == self.profile[index].parent_index && idx > index {
                return true;
            }
        }
        false
    }

    pub fn finish(&mut self) {
        let open_task_positions = self.stack.iter().map(|&(i, _)| i);
        for open_task_position in open_task_positions {
            let task_name = &mut self.profile[open_task_position].name;
            task_name.push_str(" (unfinished)");
        }

        let num_open_tasks = self.stack.len();
        for _ in 0..num_open_tasks {
            self.plain_stop();
        }

        self.total_time = Some(self.timer.elapsed());
    }

    /// [Finishes](TritonProfiler::finish) the profiling and generates a profiling report.
    pub fn report(&mut self) -> Report {
        if self.total_time.is_none() {
            self.finish();
        }
        let total_time = self.total_time.expect("finishing should set a total time");

        let mut report: Vec<TaskReport> = vec![];

        // collect all categories and their total times
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
            // compute this task's time relative to total duration
            let relative_time = task.time.as_secs_f64() / total_time.as_secs_f64();
            let weight = match task.task_type {
                TaskType::AnyOtherIteration => Weight::LikeNothing,
                _ => Weight::weigh(task.time.as_secs_f64() / total_time.as_secs_f64()),
            };

            let is_last_sibling = !self.has_younger_sibling(task_index);

            // compute this task's ancestors
            let mut ancestors: Vec<usize> = vec![];
            let mut current_ancestor_index = task.parent_index;
            while let Some(cai) = current_ancestor_index {
                ancestors.push(cai);
                current_ancestor_index = report[cai].parent_index;
            }
            ancestors.reverse();

            let relative_category_time = task.category.clone().map(|category| {
                let category_time = category_times.get(&category).unwrap();
                task.time.as_secs_f64() / category_time.as_secs_f64()
            });

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

        // pass over report again to fix forward references
        for task_index in 0..report.len() {
            let task = &self.profile[task_index];
            let mut younger_siblings: Vec<usize> = vec![];
            for (tsk_idx, tsk) in self.profile.iter().enumerate() {
                if tsk.parent_index == task.parent_index && tsk_idx > task_index {
                    younger_siblings.push(tsk_idx);
                }
            }
            let mut younger_max_weight: Weight = Weight::LikeNothing;
            for &sibling in &younger_siblings {
                younger_max_weight = max(younger_max_weight, report[sibling].weight);
            }

            report[task_index].younger_max_weight = younger_max_weight;
        }

        // “Other iterations” are not currently tracked
        let is_other_iteration = |t: &&Task| t.task_type == TaskType::AnyOtherIteration;
        let all_tasks = self.profile.iter();
        let other_iterations = all_tasks.filter(is_other_iteration);
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

    pub fn start(&mut self, name: &str, category: Option<String>) {
        if !self.ignoring() {
            self.plain_start(name, TaskType::Generic, category);
        }
    }

    fn plain_start(&mut self, name: &str, task_type: TaskType, category: Option<String>) {
        let parent_index = self.stack.last().map(|(u, _)| *u);
        let now = self.timer.elapsed();

        self.stack.push((self.profile.len(), name.to_owned()));

        self.profile.push(Task {
            name: name.to_owned(),
            parent_index,
            depth: self.stack.len(),
            time: now,
            task_type,
            category,
        });

        if env_var(GET_PROFILE_OUTPUT_AS_YOU_GO_ENV_VAR_NAME).is_ok() {
            println!("start: {name}");
        }
    }

    pub fn iteration_zero(&mut self, name: &str, category: Option<String>) {
        if self.ignoring() {
            return;
        }

        assert!(
            !self.stack.is_empty(),
            "Profiler stack is empty; can't iterate."
        );

        let top_index = self.stack[self.stack.len() - 1].0;
        let top_type = self.profile[top_index].task_type;

        if top_type != TaskType::IterationZero && top_type != TaskType::AnyOtherIteration {
            // start
            self.plain_start("iteration 0", TaskType::IterationZero, category);
            return;
        }

        assert!(
            self.stack.len() >= 2,
            "To profile zeroth iteration, stack must be at least 2-high, but got height of {}",
            self.stack.len()
        );

        let runner_up = self.stack[self.stack.len() - 2].1.clone();

        assert_eq!(
            runner_up, name,
            "To profile zeroth iteration, name must match with top of stack."
        );

        if top_type == TaskType::IterationZero {
            // switch
            // stop iteration zero
            self.plain_stop();

            // start all other iterations
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
        let Some((index, name)) = self.stack.pop() else {
            return;
        };
        let now = self.timer.elapsed();
        let duration = now - self.profile[index].time;
        self.profile[index].time = duration;

        if env_var(GET_PROFILE_OUTPUT_AS_YOU_GO_ENV_VAR_NAME).is_ok() {
            println!("stop:  {name} – took {duration:.2?}");
        }
    }

    pub fn stop(&mut self, name: &str) {
        assert!(
            !self.stack.is_empty(),
            "Cannot stop any tasks when stack is empty.",
        );

        let top = self.stack.last().unwrap().1.clone();
        if top == *"iteration 0" || top == *"all other iterations" {
            assert!(
                self.stack.len() >= 2,
                "To close profiling of zeroth iteration, stack must be at least 2-high, \
                but got stack of height {}.",
                self.stack.len(),
            );
            if self.stack[self.stack.len() - 2].1 == *name {
                self.plain_stop();
                self.plain_stop();
            }
        } else {
            assert_eq!(
                top,
                name.to_owned(),
                "Profiler stack is LIFO; can't pop in disorder."
            );

            self.plain_stop();
        }
    }
}

impl Profiler for TritonProfiler {
    fn start_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        let dir = benchmark_dir
            .to_str()
            .expect("Directory must be valid unicode");
        let name = format!("{dir}{benchmark_id}");
        let category = None;
        self.start(&name, category);
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        let dir = benchmark_dir
            .to_str()
            .expect("Directory must be valid unicode");
        let name = format!("{dir}{benchmark_id}");
        self.stop(&name)
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

#[derive(Debug)]
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

#[derive(Debug)]
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

            f.write_fmt(format_args!(
                "{task_name_colored}   \
                 {task_time_colored}{relative_time_string_colored} \
                 {category_and_relative_time_colored}\n"
            ))?;
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

        if self.cycle_count.is_some()
            || self.padded_height.is_some()
            || self.fri_domain_len.is_some()
        {
            writeln!(f)?;
        }

        let total_time = self.total_time.as_millis() as usize;
        if let Some(cycle_count) = self.cycle_count {
            if total_time != 0 {
                let freq = 1_000 * cycle_count / total_time;
                writeln!(
                    f,
                    "Clock frequency is {freq} Hz ({cycle_count} clock cycles / {total_time} ms)",
                )?;
            }
        }

        if let Some(padded_height) = self.padded_height {
            if total_time != 0 {
                let optimal_freq = 1_000 * padded_height / total_time;
                writeln!(
                    f,
                    "Optimal clock frequency is {optimal_freq} Hz \
                    ({padded_height} padded height / {total_time} ms)",
                )?;
            }
        }

        if let Some(fri_domain_length) = self.fri_domain_len {
            if fri_domain_length != 0 {
                let log_2_fri_domain_length = fri_domain_length.ilog2();
                writeln!(f, "FRI domain length is 2^{log_2_fri_domain_length}")?;
            }
        }

        Ok(())
    }
}

/// Start a profiling task.
/// Requires an `Option<Profiler>` as first argument. Does nothing if this is `None`.
/// The second argument is the name of the task.
/// The third argument is an optional task category.
macro_rules! prof_start {
    ($p: ident, $s : expr, $c : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.start($s, Some($c.to_string()));
        }
    };
    ($p: ident, $s : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.start($s, None);
        }
    };
}
pub(crate) use prof_start;

/// Stop a profiling task. Requires the same arguments as [`prof_start`], except that the task's
/// category (if any) is inferred. Notably, the task's name needs to be an exact match to prevent
/// the accidental stopping of a different task.
macro_rules! prof_stop {
    ($p: ident, $s : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.stop($s);
        }
    };
}
pub(crate) use prof_stop;

/// Profile one iteration of a loop. Requires the same arguments as [`prof_start`].
/// This macro should be invoked inside the loop in question.
/// The profiling of the loop has to be stopped with [`prof_stop`] after the loop.
macro_rules! prof_itr0 {
    ($p : ident, $s : expr, $c : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.iteration_zero($s, Some($c.to_string()));
        }
    };
    ($p : ident, $s : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.iteration_zero($s, None);
        }
    };
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

        println!("{}", profiler.report());
    }

    #[test]
    fn clk_freq() {
        let mut profiler = Some(TritonProfiler::new("Clock Frequency Test"));
        prof_start!(profiler, "clk_freq_test");
        sleep(Duration::from_millis(3));
        prof_stop!(profiler, "clk_freq_test");
        let mut profiler = profiler.unwrap();

        let report_with_no_optionals = profiler.report();
        println!("{report_with_no_optionals}");

        let report_with_optionals_set_to_0 = profiler
            .report()
            .with_cycle_count(0)
            .with_padded_height(0)
            .with_fri_domain_len(0);
        println!("{report_with_optionals_set_to_0}");

        let report_with_optionals_set = profiler
            .report()
            .with_cycle_count(10)
            .with_padded_height(12)
            .with_fri_domain_len(32);
        println!("{report_with_optionals_set}");
    }

    #[test]
    fn macros() {
        let mut profiler = TritonProfiler::new("Macro Test");
        let mut profiler_ref = Some(&mut profiler);
        let mut rng = rand::thread_rng();
        let mut stack = vec![];
        let steps = 100;

        for step in 0..steps {
            let steps_left = steps - step;
            let pushable = rng.gen() && stack.len() + 1 < steps_left;
            let poppable = ((!pushable && rng.gen())
                || (stack.len() == steps_left && steps_left > 0))
                && !stack.is_empty();

            if pushable {
                let name = random_task_name(&mut rng);
                let category = random_category(&mut rng);
                stack.push(name.clone());
                match rng.gen() {
                    true => prof_start!(profiler_ref, &name, category),
                    false => prof_start!(profiler_ref, &name),
                }
            }

            sleep(Duration::from_micros(
                (rng.next_u64() % 10) * (rng.next_u64() % 10) * (rng.next_u64() % 10),
            ));

            if poppable {
                let name = stack.pop().unwrap();
                prof_stop!(profiler_ref, &name);
            }
        }

        println!("{}", profiler.report());
    }

    #[test]
    fn profiler_without_any_tasks_can_generate_a_report() {
        let mut profiler = TritonProfiler::new("Empty Test");
        let report = profiler.report();
        println!("{report}");
    }

    #[test]
    fn profiler_can_be_finished_before_generating_a_report() {
        let mut profiler = TritonProfiler::new("Finish Before Report Test");
        profiler.start("some task", None);
        profiler.finish();
        let report = profiler.report();
        println!("{report}");
    }

    #[test]
    fn profiler_with_unfinished_tasks_can_generate_report() {
        let mut profiler = TritonProfiler::new("Unfinished Tasks Test");
        profiler.start("unfinished task", None);
        let report = profiler.report();
        println!("{report}");
    }

    #[test]
    fn profiler_can_generate_multiple_reports() {
        let mut profiler = TritonProfiler::new("Multiple Reports Test");
        profiler.start("task 1", None);
        let report = profiler.report();
        println!("{report}");

        profiler.start("task 2", None);
        let report = profiler.report();
        println!("{report}");
    }
}
