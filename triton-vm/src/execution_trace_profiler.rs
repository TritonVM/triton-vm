use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;

use air::table::TableId;
use arbitrary::Arbitrary;
use strum::IntoEnumIterator;

use crate::aet::AlgebraicExecutionTrace;
use crate::table::u32::U32TableEntry;

#[derive(Debug, Default, Clone, Eq, PartialEq, Arbitrary)]
pub(crate) struct ExecutionTraceProfiler {
    call_stack: Vec<usize>,
    profile: Vec<ProfileLine>,
    u32_table_entries: HashSet<U32TableEntry>,
}

/// A single line in a [profile report](ExecutionTraceProfile) for profiling
/// [Triton](crate) programs.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct ProfileLine {
    pub label: String,
    pub call_depth: usize,

    /// Table heights at the start of this span, _i.e._, right after the
    /// corresponding [`call`](isa::instruction::Instruction::Call)
    /// instruction was executed.
    pub table_heights_start: VMTableHeights,

    /// Table heights at the end of this span, _i.e._, right after the
    /// corresponding [`return`](isa::instruction::Instruction::Return)
    /// or [`recurse_or_return`](isa::instruction::Instruction::RecurseOrReturn)
    /// (in “return” mode) was executed
    pub table_heights_stop: VMTableHeights,
}

/// A report for the completed execution of a [Triton](crate) program.
///
/// Offers a human-readable [`Display`] implementation and can be processed
/// programmatically.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct ExecutionTraceProfile {
    /// The total height of all tables in the [AET](AlgebraicExecutionTrace).
    pub total: VMTableHeights,

    /// The profile lines, each representing a span of the program execution.
    pub profile: Vec<ProfileLine>,

    /// The [padded height](AlgebraicExecutionTrace::padded_height) of the
    /// [AET](AlgebraicExecutionTrace).
    pub padded_height: usize,
}

/// The heights of various [tables](AlgebraicExecutionTrace)
/// relevant for proving the correct execution in [Triton VM](crate).
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub struct VMTableHeights {
    pub program: u32,
    pub processor: u32,
    pub op_stack: u32,
    pub ram: u32,
    pub jump_stack: u32,
    pub hash: u32,
    pub cascade: u32,
    pub lookup: u32,
    pub u32: u32,
}

impl ExecutionTraceProfiler {
    pub fn new() -> Self {
        Self {
            call_stack: vec![],
            profile: vec![],
            u32_table_entries: HashSet::default(),
        }
    }

    pub fn enter_span(&mut self, label: impl Into<String>, aet: &AlgebraicExecutionTrace) {
        let profile_line = ProfileLine {
            label: label.into(),
            call_depth: self.call_stack.len(),
            table_heights_start: Self::extract_table_heights(aet),
            table_heights_stop: VMTableHeights::default(),
        };

        let line_number = self.profile.len();
        self.profile.push(profile_line);
        self.call_stack.push(line_number);
    }

    pub fn exit_span(&mut self, aet: &AlgebraicExecutionTrace) {
        if let Some(line_number) = self.call_stack.pop() {
            self.profile[line_number].table_heights_stop = Self::extract_table_heights(aet);
        };
    }

    pub fn finish(mut self, aet: &AlgebraicExecutionTrace) -> ExecutionTraceProfile {
        let table_heights = Self::extract_table_heights(aet);
        for &line_number in &self.call_stack {
            self.profile[line_number].table_heights_stop = table_heights;
        }

        ExecutionTraceProfile {
            total: table_heights,
            profile: self.profile,
            padded_height: aet.padded_height(),
        }
    }

    fn extract_table_heights(aet: &AlgebraicExecutionTrace) -> VMTableHeights {
        // any table in Triton VM can be of length at most u32::MAX
        let height = |id| aet.height_of_table(id).try_into().unwrap_or(u32::MAX);

        VMTableHeights {
            program: height(TableId::Program),
            processor: height(TableId::Processor),
            op_stack: height(TableId::OpStack),
            ram: height(TableId::Ram),
            jump_stack: height(TableId::JumpStack),
            hash: height(TableId::Hash),
            cascade: height(TableId::Cascade),
            lookup: height(TableId::Lookup),
            u32: height(TableId::U32),
        }
    }
}

impl VMTableHeights {
    fn height_of_table(&self, table: TableId) -> u32 {
        match table {
            TableId::Program => self.program,
            TableId::Processor => self.processor,
            TableId::OpStack => self.op_stack,
            TableId::Ram => self.ram,
            TableId::JumpStack => self.jump_stack,
            TableId::Hash => self.hash,
            TableId::Cascade => self.cascade,
            TableId::Lookup => self.lookup,
            TableId::U32 => self.u32,
        }
    }

    fn max(&self) -> u32 {
        TableId::iter()
            .map(|id| self.height_of_table(id))
            .max()
            .unwrap()
    }
}

impl Sub for VMTableHeights {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            program: self.program.saturating_sub(rhs.program),
            processor: self.processor.saturating_sub(rhs.processor),
            op_stack: self.op_stack.saturating_sub(rhs.op_stack),
            ram: self.ram.saturating_sub(rhs.ram),
            jump_stack: self.jump_stack.saturating_sub(rhs.jump_stack),
            hash: self.hash.saturating_sub(rhs.hash),
            cascade: self.cascade.saturating_sub(rhs.cascade),
            lookup: self.lookup.saturating_sub(rhs.lookup),
            u32: self.u32.saturating_sub(rhs.u32),
        }
    }
}

impl Add for VMTableHeights {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            program: self.program + rhs.program,
            processor: self.processor + rhs.processor,
            op_stack: self.op_stack + rhs.op_stack,
            ram: self.ram + rhs.ram,
            jump_stack: self.jump_stack + rhs.jump_stack,
            hash: self.hash + rhs.hash,
            cascade: self.cascade + rhs.cascade,
            lookup: self.lookup + rhs.lookup,
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

        let max_label_len = aggregated.iter().map(|line| label(line).len()).max();
        let max_label_len = max_label_len.unwrap_or_default().max(COL_WIDTH);

        write!(f, "| {: <max_label_len$} ", "Subroutine")?;
        write!(f, "| {: >COL_WIDTH$} ", "Processor")?;
        write!(f, "| {: >COL_WIDTH$} ", "OpStack")?;
        write!(f, "| {: >COL_WIDTH$} ", "Ram")?;
        write!(f, "| {: >COL_WIDTH$} ", "Hash")?;
        write!(f, "| {: >COL_WIDTH$} ", "U32")?;
        writeln!(f, "|")?;

        write!(f, "|:{:-<max_label_len$}-", "")?;
        write!(f, "|-{:->COL_WIDTH$}:", "")?;
        write!(f, "|-{:->COL_WIDTH$}:", "")?;
        write!(f, "|-{:->COL_WIDTH$}:", "")?;
        write!(f, "|-{:->COL_WIDTH$}:", "")?;
        write!(f, "|-{:->COL_WIDTH$}:", "")?;
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

        // print total height of all tables
        let max_height = self.total.max();
        let height_len = std::cmp::max(max_height.to_string().len(), "Height".len());

        writeln!(f)?;
        writeln!(f, "| Table     | {: >height_len$} | Dominates |", "Height")?;
        writeln!(f, "|:----------|-{:->height_len$}:|----------:|", "")?;
        for id in TableId::iter() {
            let height = self.total.height_of_table(id);
            let dominates = if height == max_height { "yes" } else { "no" };
            writeln!(f, "| {id:<9} | {height:>height_len$} | {dominates:>9} |")?;
        }
        writeln!(f)?;
        writeln!(f, "Padded height: 2^{}", self.padded_height.ilog2())?;

        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use assert2::let_assert;

    use crate::prelude::InstructionError;
    use crate::prelude::TableId;
    use crate::prelude::VM;
    use crate::prelude::VMState;
    use crate::prelude::triton_program;

    #[test]
    fn profile_can_be_created_and_agrees_with_regular_vm_run() {
        let program =
            crate::example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone();
        let (profile_output, profile) = VM::profile(program.clone(), [].into(), [].into()).unwrap();
        let mut vm_state = VMState::new(program.clone(), [].into(), [].into());
        let_assert!(Ok(()) = vm_state.run());
        assert!(profile_output == vm_state.public_output);
        assert!(profile.total.processor == vm_state.cycle_count);

        let_assert!(Ok((aet, trace_output)) = VM::trace_execution(program, [].into(), [].into()));
        assert!(profile_output == trace_output);

        let height = |id| u32::try_from(aet.height_of_table(id)).unwrap();
        assert!(height(TableId::Program) == profile.total.program);
        assert!(height(TableId::Processor) == profile.total.processor);
        assert!(height(TableId::OpStack) == profile.total.op_stack);
        assert!(height(TableId::Ram) == profile.total.ram);
        assert!(height(TableId::Hash) == profile.total.hash);
        assert!(height(TableId::Cascade) == profile.total.cascade);
        assert!(height(TableId::Lookup) == profile.total.lookup);
        assert!(height(TableId::U32) == profile.total.u32);

        println!("{profile}");
    }

    #[test]
    fn program_with_too_many_returns_crashes_vm_but_not_profiler() {
        let program = triton_program! {
            call foo return halt
            foo: return
        };
        let_assert!(Err(err) = VM::profile(program, [].into(), [].into()));
        let_assert!(InstructionError::JumpStackIsEmpty = err.source);
    }

    #[test]
    fn call_instruction_does_not_contribute_to_profile_span() {
        let program = triton_program! { call foo halt foo: return };
        let_assert!(Ok((_, profile)) = VM::profile(program, [].into(), [].into()));

        let [foo_span] = &profile.profile[..] else {
            panic!("span `foo` must be present")
        };
        assert!("foo" == foo_span.label);
        assert!(1 == foo_span.table_height_contributions().processor);
    }
}
