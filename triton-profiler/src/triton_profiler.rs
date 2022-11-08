use std::{
    fmt::Display,
    time::{Duration, Instant},
    vec,
};

use colored::{Color, ColoredString, Colorize};
use criterion::profiler::Profiler;
use unicode_width::UnicodeWidthStr;

#[derive(Clone, Debug)]
struct Task {
    name: String,
    parent_index: Option<usize>,
    depth: usize,
    time: Duration,
}

pub struct TritonProfiler {
    name: String,
    timer: Instant,
    stack: Vec<(usize, String)>,
    profile: Vec<Task>,
    total_time: Duration,
}

impl TritonProfiler {
    #[allow(clippy::new_without_default)]
    pub fn new(name: &str) -> Self {
        TritonProfiler {
            name: name.to_owned(),
            timer: Instant::now(),
            stack: vec![],
            profile: vec![],
            total_time: Duration::ZERO,
        }
    }

    fn ignoring(&self) -> bool {
        if let Some(top) = self.stack.last() {
            if top.1 == *"all other iterations" {
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

    pub fn report(&self) -> Report {
        assert!(!self.profile.is_empty(), "Nothing to report on.");
        assert!(
            self.stack.is_empty(),
            "Cannot generate report before stack is empty."
        );
        assert!(
            self.total_time != Duration::ZERO,
            "Cannot generate report before profiler has finished. Call `finish()` first."
        );

        let mut report: Vec<TaskReport> = vec![];
        let total_tracked_time = (self.total_time.as_nanos()
            - self
                .profile
                .iter()
                .filter(|t| t.name == *"all other iterations")
                .map(|t| t.time.as_nanos())
                .sum::<u128>()) as f64
            / 1_000_000_000f64;
        println!(
            "total time: {}s and tracked: {}s",
            self.total_time.as_secs_f64(),
            total_tracked_time,
        );
        for (task_index, task) in self.profile.iter().enumerate() {
            // compute this task's time relative to total duration
            let relative_time = task
                .parent_index
                .map(|_| task.time.as_secs_f64() / self.total_time.as_secs_f64());

            let weight = if task.name == *"all other iterations" {
                Weight::Light
            } else {
                Weight::weigh(task.time.as_secs_f64() / total_tracked_time)
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

            report.push(TaskReport {
                name: task.name.clone(),
                parent_index: task.parent_index,
                depth: task.depth,
                time: task.time,
                relative_time,
                is_last_sibling,
                ancestors,
                weight,
                younger_max_weight: Weight::Light,
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
            let mut younger_max_weight: Weight = Weight::Light;
            for sibling in younger_siblings.iter() {
                younger_max_weight = Weight::max(&younger_max_weight, &report[*sibling].weight);
            }

            report[task_index].younger_max_weight = younger_max_weight;
        }

        Report {
            tasks: report,
            name: self.name.clone(),
            total_time: self.total_time,
        }
    }

    pub fn start(&mut self, name: &str) {
        if !self.ignoring() {
            self.plain_start(name);
        }
    }

    fn plain_start(&mut self, name: &str) {
        let parent_index = self.stack.last().map(|(u, _)| *u);
        let now = self.timer.elapsed();

        self.stack.push((self.profile.len(), name.to_owned()));

        self.profile.push(Task {
            name: name.to_owned(),
            parent_index,
            depth: self.stack.len(),
            time: now,
        });
    }

    pub fn iteration_zero(&mut self, name: &str) {
        if self.ignoring() {
            return;
        }

        assert!(
            !self.stack.is_empty(),
            "Profiler stack is empty; can't iterate."
        );

        let top = self.stack[self.stack.len() - 1].1.clone();

        if top != *"iteration 0" && top != *"all other iterations" {
            // start
            self.plain_start("iteration 0");
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

        if top == *"iteration 0" {
            // switch
            // stop iteration zero
            self.plain_stop();

            // start all other iterations
            self.plain_start("all other iterations");
        }

        // top == *"all other iterations"
        // in this case we do nothing
    }

    fn plain_stop(&mut self) {
        let index = self.stack.pop().unwrap().0;
        let now = self.timer.elapsed();
        let duration = now - self.profile[index].time;
        self.profile[index].time = duration;
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
                "To close profiling of zeroth iteration, stack must be at least 2-high, but got stack of height {}.",
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

    pub fn finish(&mut self) {
        assert!(!self.profile.is_empty(), "Nothing to finish.");
        assert!(
            self.stack.is_empty(),
            "Cannot finish before stack is empty."
        );
        self.total_time = self.timer.elapsed();
    }
}

impl Profiler for TritonProfiler {
    fn start_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        let dir = benchmark_dir
            .to_str()
            .expect("Directory must be valid unicode");
        let name = format!("{}{}", dir, benchmark_id);
        self.start(&name);
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        let dir = benchmark_dir
            .to_str()
            .expect("Directory must be valid unicode");
        let name = format!("{}{}", dir, benchmark_id);
        self.stop(&name)
    }
}

#[derive(Debug, Clone)]
enum Weight {
    Light,
    Noticeable,
    Heavy,
    Massive,
}

impl Weight {
    /// Assign a weight based on a relative cost, which is a number
    /// between 0 and 1.
    fn weigh(relative_cost: f64) -> Weight {
        match relative_cost {
            rc if rc >= 0.5 => Weight::Massive,
            rc if rc >= 0.4 => Weight::Heavy,
            rc if rc >= 0.3 => Weight::Noticeable,
            _ => Weight::Light,
        }
    }

    fn color(&self) -> Color {
        match self {
            Weight::Light => Color::White,
            Weight::Noticeable => Color::TrueColor {
                r: 200,
                g: 200,
                b: 100,
            },
            Weight::Heavy => Color::TrueColor {
                r: 255,
                g: 150,
                b: 0,
            },
            Weight::Massive => Color::Red,
        }
    }

    fn max(&self, other: &Self) -> Self {
        match self {
            Weight::Light => match other {
                Weight::Light => Weight::Light,
                Weight::Noticeable => Weight::Noticeable,
                Weight::Heavy => Weight::Heavy,
                Weight::Massive => Weight::Massive,
            },
            Weight::Noticeable => match other {
                Weight::Light => Weight::Noticeable,
                Weight::Noticeable => Weight::Noticeable,
                Weight::Heavy => Weight::Heavy,
                Weight::Massive => Weight::Massive,
            },
            Weight::Heavy => match other {
                Weight::Light => Weight::Heavy,
                Weight::Noticeable => Weight::Heavy,
                Weight::Heavy => Weight::Heavy,
                Weight::Massive => Weight::Massive,
            },
            Weight::Massive => match other {
                Weight::Light => Weight::Massive,
                Weight::Noticeable => Weight::Massive,
                Weight::Heavy => Weight::Massive,
                Weight::Massive => Weight::Massive,
            },
        }
    }
}

#[derive(Debug)]
struct TaskReport {
    name: String,
    parent_index: Option<usize>,
    depth: usize,
    time: Duration,
    relative_time: Option<f64>,
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
}

impl Report {
    pub fn placeholder() -> Self {
        Report {
            name: "".to_string(),
            tasks: vec![],
            total_time: Duration::ZERO,
        }
    }
}

impl Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_name_width = self
            .tasks
            .iter()
            .map(|t| t.name.width() + 2 * t.depth)
            .max()
            .expect("No tasks to generate report from.");

        let title = format!("### {}", self.name).bold();
        let max_width = if max_name_width > title.width() {
            max_name_width
        } else {
            title.width()
        };
        let total_time_string = format!("{:.2?}", self.total_time).bold();
        let separation = String::from_utf8(vec![b' '; max_width - title.width()]).unwrap();
        writeln!(f, "{}{}   {}", title, separation, total_time_string)?;

        for task in self.tasks.iter() {
            for ancestor_index in task.ancestors.iter() {
                let mut spacer: ColoredString = if self.tasks[*ancestor_index].is_last_sibling {
                    "  ".normal()
                } else {
                    "│ ".normal()
                };
                let uncle_weight = &self.tasks[*ancestor_index].younger_max_weight;
                spacer = spacer.color(uncle_weight.color());

                write!(f, "{}", spacer)?;
            }
            let tracer = if task.is_last_sibling {
                "└".normal()
            } else {
                "├".normal()
            }
            .color(
                Weight::max(&task.weight, &task.younger_max_weight)
                    .to_owned()
                    .color(),
            );
            let dash = "─".color(task.weight.color());
            write!(f, "{}{}", tracer, dash)?;

            let padding_length = max_width - task.name.len() - 2 * task.depth;
            assert!(
                padding_length < (1 << 60),
                "max width: {}, width: {}",
                max_name_width,
                task.name.len(),
            );
            let task_name_colored = task.name.color(task.weight.color());
            let mut task_time = format!("{:.2?}", task.time);
            while task_time.width() < 10 {
                task_time.push(b' '.into());
            }
            let task_time_colored = task_time.color(task.weight.color());
            let padding = String::from_utf8(vec![b' '; padding_length]).unwrap();
            let relative_time_string = if let Some(rt) = task.relative_time {
                if rt > 0.3 {
                    format!("{:2.2}%", 100.0 * rt)
                } else {
                    "      ".to_owned()
                }
            } else {
                "      ".to_owned()
            };
            let relative_time_string_colored = relative_time_string.color(task.weight.color());
            f.write_fmt(format_args!(
                "{}{}   {}{}\n",
                task_name_colored, padding, task_time_colored, relative_time_string_colored,
            ))?;
        }

        Ok(())
    }
}

#[macro_export]
macro_rules! prof_start {
    ($p: ident, $s : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.start($s);
        }
    };
}

#[macro_export]
macro_rules! prof_stop {
    ($p: ident, $s : expr) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.stop($s);
        }
    };
}

#[macro_export]
macro_rules! prof_itr0 {
    ($p : ident, $s : expr ) => {
        if let Some(profiler) = $p.as_mut() {
            profiler.iteration_zero($s);
        }
    };
}

#[cfg(test)]
pub mod triton_profiler_tests {
    use super::*;
    use std::{thread::sleep, time::Duration};

    use rand::{rngs::ThreadRng, RngCore};

    fn random_task_name(rng: &mut ThreadRng) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz_";
        let length = (rng.next_u32() as usize % 10) + 2;

        (0..length)
            .map(|_| (rng.next_u32() as usize) % alphabet.len())
            .map(|i| alphabet.get(i..=i).unwrap())
            .collect()
    }

    #[test]
    fn test_sanity() {
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
                stack.push(name.clone());
                profiler.start(&name);
            }

            sleep(Duration::from_micros(
                (rng.next_u64() % 10) * (rng.next_u64() % 10) * (rng.next_u64() % 10),
            ));

            if poppable {
                let name = stack.pop().unwrap();
                profiler.stop(&name);
            }
        }
        profiler.finish();

        let profile = profiler.report();
        println!("{}", profile);
    }
}
