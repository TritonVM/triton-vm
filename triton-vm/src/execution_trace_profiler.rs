use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;

use arbitrary::Arbitrary;
use twenty_first::prelude::*;

use crate::table::hash_table::PERMUTATION_TRACE_LENGTH;
use crate::table::u32_table::U32TableEntry;
use crate::vm::CoProcessorCall;

#[derive(Debug, Default, Clone, Eq, PartialEq, Arbitrary)]
pub(crate) struct ExecutionTraceProfiler {
    call_stack: Vec<usize>,
    profile: Vec<ProfileLine>,
    table_heights: VMTableHeights,
    u32_table_entries: HashSet<U32TableEntry>,
}

/// A single line in a [profile report](ExecutionTraceProfile) for profiling
/// [Triton](crate) programs.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct ProfileLine {
    pub label: String,
    pub call_depth: usize,

    /// Table heights at the start of this span, _i.e._, right before the corresponding
    /// [`call`](isa::instruction::Instruction::Call) instruction was executed.
    pub table_heights_start: VMTableHeights,

    table_heights_stop: VMTableHeights,
}

/// A report for the completed execution of a [Triton](crate) program.
///
/// Offers a human-readable [`Display`] implementation and can be processed
/// programmatically.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct ExecutionTraceProfile {
    pub total: VMTableHeights,
    pub profile: Vec<ProfileLine>,
}

/// The heights of various [tables](crate::aet::AlgebraicExecutionTrace) relevant for
/// proving the correct execution in [Triton VM](crate).
#[non_exhaustive]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct VMTableHeights {
    pub processor: u32,
    pub op_stack: u32,
    pub ram: u32,
    pub hash: u32,
    pub u32: u32,
}

impl ExecutionTraceProfiler {
    pub fn new(num_instructions: usize) -> Self {
        Self {
            call_stack: vec![],
            profile: vec![],
            table_heights: VMTableHeights::new(num_instructions),
            u32_table_entries: HashSet::default(),
        }
    }

    pub fn enter_span(&mut self, label: impl Into<String>) {
        let call_stack_len = self.call_stack.len();
        let line_number = self.profile.len();

        let profile_line = ProfileLine {
            label: label.into(),
            call_depth: call_stack_len,
            table_heights_start: self.table_heights,
            table_heights_stop: VMTableHeights::default(),
        };

        self.profile.push(profile_line);
        self.call_stack.push(line_number);
    }

    pub fn exit_span(&mut self) {
        if let Some(line_number) = self.call_stack.pop() {
            self.profile[line_number].table_heights_stop = self.table_heights;
        };
    }

    pub fn handle_co_processor_calls(&mut self, calls: Vec<CoProcessorCall>) {
        self.table_heights.processor += 1;
        for call in calls {
            match call {
                CoProcessorCall::SpongeStateReset => self.table_heights.hash += 1,
                CoProcessorCall::Tip5Trace(_, trace) => {
                    self.table_heights.hash += u32::try_from(trace.len()).unwrap();
                }
                CoProcessorCall::U32Call(c) => {
                    self.u32_table_entries.insert(c);
                    let contribution = U32TableEntry::table_height_contribution;
                    self.table_heights.u32 = self.u32_table_entries.iter().map(contribution).sum();
                }
                CoProcessorCall::OpStackCall(_) => self.table_heights.op_stack += 1,
                CoProcessorCall::RamCall(_) => self.table_heights.ram += 1,
            }
        }
    }

    pub fn finish(mut self) -> ExecutionTraceProfile {
        for &line_number in &self.call_stack {
            self.profile[line_number].table_heights_stop = self.table_heights;
        }

        ExecutionTraceProfile {
            total: self.table_heights,
            profile: self.profile,
        }
    }
}

impl VMTableHeights {
    fn new(num_instructions: usize) -> Self {
        let padded_program_len = (num_instructions + 1).next_multiple_of(Tip5::RATE);
        let num_absorbs = padded_program_len / Tip5::RATE;
        let initial_hash_table_len = num_absorbs * PERMUTATION_TRACE_LENGTH;

        Self {
            hash: initial_hash_table_len.try_into().unwrap(),
            ..Default::default()
        }
    }
}

impl Sub<Self> for VMTableHeights {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            processor: self.processor.saturating_sub(rhs.processor),
            op_stack: self.op_stack.saturating_sub(rhs.op_stack),
            ram: self.ram.saturating_sub(rhs.ram),
            hash: self.hash.saturating_sub(rhs.hash),
            u32: self.u32.saturating_sub(rhs.u32),
        }
    }
}

impl Add<Self> for VMTableHeights {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            processor: self.processor + rhs.processor,
            op_stack: self.op_stack + rhs.op_stack,
            ram: self.ram + rhs.ram,
            hash: self.hash + rhs.hash,
            u32: self.u32 + rhs.u32,
        }
    }
}

impl AddAssign<Self> for VMTableHeights {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ProfileLine {
    fn table_height_contributions(&self) -> VMTableHeights {
        self.table_heights_stop - self.table_heights_start
    }
}

impl Display for ProfileLine {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let indentation = "  ".repeat(self.call_depth);
        let label = &self.label;
        let cycle_count = self.table_height_contributions().processor;
        write!(f, "{indentation}{label}: {cycle_count}")
    }
}

impl Display for ExecutionTraceProfile {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        struct AggregateLine {
            label: String,
            call_depth: usize,
            table_heights: VMTableHeights,
        }

        const COL_WIDTH: usize = 20;

        let mut aggregated: Vec<AggregateLine> = vec![];
        for line in &self.profile {
            if let Some(agg) = aggregated
                .iter_mut()
                .find(|agg| agg.label == line.label && agg.call_depth == line.call_depth)
            {
                agg.table_heights += line.table_height_contributions();
            } else {
                aggregated.push(AggregateLine {
                    label: line.label.clone(),
                    call_depth: line.call_depth,
                    table_heights: line.table_height_contributions(),
                });
            }
        }
        aggregated.push(AggregateLine {
            label: "Total".to_string(),
            call_depth: 0,
            table_heights: self.total,
        });

        let label = |line: &AggregateLine| "··".repeat(line.call_depth) + &line.label;
        let label_len = |line| label(line).len();

        let max_label_len = aggregated.iter().map(label_len).max();
        let max_label_len = max_label_len.unwrap_or_default().max(COL_WIDTH);

        let [subroutine, processor, op_stack, ram, hash, u32_title] =
            ["Subroutine", "Processor", "Op Stack", "RAM", "Hash", "U32"];

        write!(f, "| {subroutine:<max_label_len$} ")?;
        write!(f, "| {processor:>COL_WIDTH$} ")?;
        write!(f, "| {op_stack:>COL_WIDTH$} ")?;
        write!(f, "| {ram:>COL_WIDTH$} ")?;
        write!(f, "| {hash:>COL_WIDTH$} ")?;
        write!(f, "| {u32_title:>COL_WIDTH$} ")?;
        writeln!(f, "|")?;

        let dash = "-";
        write!(f, "|:{dash:-<max_label_len$}-")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        write!(f, "|-{dash:->COL_WIDTH$}:")?;
        writeln!(f, "|")?;

        for line in &aggregated {
            let rel_precision = 1;
            let rel_width = 3 + 1 + rel_precision; // eg '100.0'
            let abs_width = COL_WIDTH - rel_width - 4; // ' (' and '%)'

            let label = label(line);
            let proc_abs = line.table_heights.processor;
            let proc_rel = 100.0 * f64::from(proc_abs) / f64::from(self.total.processor);
            let proc_rel = format!("{proc_rel:.rel_precision$}");
            let stack_abs = line.table_heights.op_stack;
            let stack_rel = 100.0 * f64::from(stack_abs) / f64::from(self.total.op_stack);
            let stack_rel = format!("{stack_rel:.rel_precision$}");
            let ram_abs = line.table_heights.ram;
            let ram_rel = 100.0 * f64::from(ram_abs) / f64::from(self.total.ram);
            let ram_rel = format!("{ram_rel:.rel_precision$}");
            let hash_abs = line.table_heights.hash;
            let hash_rel = 100.0 * f64::from(hash_abs) / f64::from(self.total.hash);
            let hash_rel = format!("{hash_rel:.rel_precision$}");
            let u32_abs = line.table_heights.u32;
            let u32_rel = 100.0 * f64::from(u32_abs) / f64::from(self.total.u32);
            let u32_rel = format!("{u32_rel:.rel_precision$}");

            write!(f, "| {label:<max_label_len$} ")?;
            write!(f, "| {proc_abs:>abs_width$} ({proc_rel:>rel_width$}%) ")?;
            write!(f, "| {stack_abs:>abs_width$} ({stack_rel:>rel_width$}%) ")?;
            write!(f, "| {ram_abs:>abs_width$} ({ram_rel:>rel_width$}%) ")?;
            write!(f, "| {hash_abs:>abs_width$} ({hash_rel:>rel_width$}%) ")?;
            write!(f, "| {u32_abs:>abs_width$} ({u32_rel:>rel_width$}%) ")?;
            writeln!(f, "|")?;
        }

        Ok(())
    }
}
