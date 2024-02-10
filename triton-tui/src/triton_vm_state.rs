use color_eyre::eyre::anyhow;
use color_eyre::eyre::bail;
use color_eyre::eyre::Result;
use color_eyre::Report;
use fs_err as fs;
use itertools::Itertools;
use tokio::sync::mpsc::UnboundedSender;
use tracing::error;
use tracing::info;
use tracing::warn;
use triton_vm::error::InstructionError;
use triton_vm::instruction::*;
use triton_vm::op_stack::NUM_OP_STACK_REGISTERS;
use triton_vm::prelude::*;

use crate::action::*;
use crate::args::TuiArgs;
use crate::components::Component;
use crate::shadow_memory::ShadowMemory;
use crate::shadow_memory::TopOfStack;

#[derive(Debug)]
pub(crate) struct TritonVMState {
    pub action_tx: Option<UnboundedSender<Action>>,

    pub program: Program,
    pub vm_state: VMState,

    pub type_hints: ShadowMemory,
    pub undo_stack: Vec<UndoInformation>,

    pub warning: Option<Report>,
    pub error: Option<InstructionError>,

    pub num_cycles_since_user_action: u32,
    pub interrupt_cycle: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct UndoInformation {
    vm_state: VMState,
    type_hints: ShadowMemory,
}

impl TritonVMState {
    pub fn new(args: &TuiArgs) -> Result<Self> {
        let program = Self::program_from_args(args)?;
        let vm_state = match args.initial_state.is_some() {
            true => Self::vm_state_from_initial_state(args, &program)?,
            false => Self::vm_state_with_specified_input(args, &program)?,
        };
        let type_hints = match args.initial_state.is_some() {
            true => ShadowMemory::new_for_initial_state(&vm_state),
            false => ShadowMemory::new_for_default_initial_state(),
        };

        let mut state = Self {
            action_tx: None,
            program,
            vm_state,
            type_hints,
            undo_stack: vec![],
            warning: None,
            error: None,
            num_cycles_since_user_action: 0,
            interrupt_cycle: args.interrupt_cycle,
        };
        state.apply_type_hints();
        Ok(state)
    }

    fn program_from_args(args: &TuiArgs) -> Result<Program> {
        let source_code = fs::read_to_string(&args.program)?;
        let program = Program::from_code(&source_code)
            .map_err(|err| anyhow!("program parsing error: {err}"))?;
        Ok(program)
    }

    fn vm_state_from_initial_state(args: &TuiArgs, program: &Program) -> Result<VMState> {
        let Some(ref initial_state_path) = args.initial_state else {
            error!("path to initial state must exist");
            bail!("path to initial state must exist");
        };
        let file = fs::File::open(initial_state_path)?;
        let initial_state: VMState = serde_json::from_reader(file)?;
        if program.instructions != initial_state.program {
            error!("given program must match program of given initial state");
            bail!("given program must match program of given initial state");
        }
        Ok(initial_state)
    }

    fn vm_state_with_specified_input(args: &TuiArgs, program: &Program) -> Result<VMState> {
        let public_input = Self::public_input_from_args(args)?;
        let non_determinism = Self::non_determinism_from_args(args)?;
        let vm_state = VMState::new(program, public_input, non_determinism);
        Ok(vm_state)
    }

    fn public_input_from_args(args: &TuiArgs) -> Result<PublicInput> {
        let Some(ref input_args) = args.input_args else {
            return Ok(PublicInput::default());
        };
        let Some(ref input_path) = input_args.input else {
            return Ok(PublicInput::default());
        };
        let file_content = fs::read_to_string(input_path)?;
        let string_tokens = file_content.split_whitespace();
        let mut elements = vec![];
        for string_token in string_tokens {
            let element = string_token.parse::<u64>()?;
            elements.push(element.into());
        }
        Ok(PublicInput::new(elements))
    }

    fn non_determinism_from_args(args: &TuiArgs) -> Result<NonDeterminism<BFieldElement>> {
        let Some(ref input_args) = args.input_args else {
            return Ok(NonDeterminism::default());
        };
        let Some(ref non_determinism_path) = input_args.non_determinism else {
            return Ok(NonDeterminism::default());
        };
        let file = fs::File::open(non_determinism_path)?;
        let non_determinism: NonDeterminism<u64> = serde_json::from_reader(file)?;
        Ok(NonDeterminism::from(&non_determinism))
    }

    fn top_of_stack(&self) -> TopOfStack {
        let stack_len = self.vm_state.op_stack.stack.len();
        let index_of_lowest_accessible_element = stack_len - NUM_OP_STACK_REGISTERS;
        let top_of_stack = &self.vm_state.op_stack.stack[index_of_lowest_accessible_element..];
        let top_of_stack = top_of_stack.iter().copied();
        top_of_stack.rev().collect_vec().try_into().unwrap()
    }

    fn vm_is_stopped(&self) -> bool {
        self.vm_state.halting || self.error.is_some() || self.interrupted()
    }

    fn vm_is_running(&self) -> bool {
        !self.vm_is_stopped()
    }

    fn at_breakpoint(&self) -> bool {
        let ip = self.vm_state.instruction_pointer as u64;
        self.program.is_breakpoint(ip)
    }

    fn apply_type_hints(&mut self) {
        let ip = self.vm_state.instruction_pointer as u64;
        for type_hint in self.program.type_hints_at(ip) {
            let maybe_error = self.type_hints.apply_type_hint(type_hint);
            if let Err(report) = maybe_error {
                info!("Error applying type hint: {report}");
                self.warning = Some(report);
            };
        }
    }

    fn execute(&mut self, execute: &Execute) {
        self.num_cycles_since_user_action = 0;
        self.record_undo_information();
        match execute {
            Execute::Continue => self.continue_execution(),
            Execute::Step => self.step(),
            Execute::Next => self.next(),
            Execute::Finish => self.finish(),
        }
    }

    /// Handle [`Execute::Continue`].
    fn continue_execution(&mut self) {
        self.step();
        while self.vm_is_running() && !self.at_breakpoint() {
            self.step();
        }
    }

    /// Handle [`Execute::Step`].
    fn step(&mut self) {
        if self.vm_is_stopped() {
            return;
        }

        let instruction = self.vm_state.current_instruction().ok();
        let old_top_of_stack = self.top_of_stack();
        if let Err(err) = self.vm_state.step() {
            warn!("Error stepping: {err}");
            self.error = Some(err);
            return;
        }
        self.warning = None;
        self.num_cycles_since_user_action += 1;
        self.maybe_inform_about_interrupt();

        let instruction = instruction.expect("instruction should exist after successful `step`");
        let new_top_of_stack = self.top_of_stack();
        let executed_instruction =
            ExecutedInstruction::new(instruction, old_top_of_stack, new_top_of_stack);

        self.send_executed_transaction(executed_instruction);
        self.type_hints.mimic_instruction(executed_instruction);
        self.apply_type_hints();
    }

    fn maybe_inform_about_interrupt(&mut self) {
        if self.interrupted() {
            let num_cycles = self.num_cycles_since_user_action;
            self.warning = Some(anyhow!(
                "Infinite loop? VM interrupted after {num_cycles} cycles since last interaction…"
            ));
        }
    }

    fn interrupted(&self) -> bool {
        self.num_cycles_since_user_action >= self.interrupt_cycle
    }

    fn send_executed_transaction(&mut self, executed_instruction: ExecutedInstruction) {
        let Some(ref action_tx) = self.action_tx else {
            error!("action_tx should exist");
            return;
        };
        let _ = action_tx.send(Action::ExecutedInstruction(Box::new(executed_instruction)));
    }

    /// Handle [`Execute::Next`].
    fn next(&mut self) {
        let instruction = self.vm_state.current_instruction();
        let instruction_is_call = matches!(instruction, Ok(Instruction::Call(_)));
        self.step();
        if instruction_is_call {
            self.finish();
        }
    }

    /// Handle [`Execute::Finish`].
    fn finish(&mut self) {
        let current_jump_stack_depth = self.vm_state.jump_stack.len();
        while self.vm_is_running() && self.vm_state.jump_stack.len() >= current_jump_stack_depth {
            self.step();
        }
    }

    fn record_undo_information(&mut self) {
        if self.vm_is_stopped() {
            return;
        }
        let undo_information = UndoInformation {
            vm_state: self.vm_state.clone(),
            type_hints: self.type_hints.clone(),
        };
        self.undo_stack.push(undo_information);

        let Some(ref action_tx) = self.action_tx else {
            error!("action_tx must exist");
            return;
        };
        let _ = action_tx.send(Action::RecordUndoInfo);
    }

    fn program_undo(&mut self) {
        let Some(undo_information) = self.undo_stack.pop() else {
            self.warning = Some(anyhow!("no more undo information available"));
            return;
        };
        self.warning = None;
        self.error = None;
        self.vm_state = undo_information.vm_state;
        self.type_hints = undo_information.type_hints;
    }
}

impl Component for TritonVMState {
    fn register_action_handler(&mut self, tx: UnboundedSender<Action>) -> Result<()> {
        self.action_tx = Some(tx);
        Ok(())
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::Execute(ref execute) => self.execute(execute),
            Action::Undo => self.program_undo(),
            _ => (),
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::args_tests::args_for_test_program_with_initial_state;
    use crate::args_tests::args_for_test_program_with_test_input;

    use super::*;

    #[proptest]
    fn presumed_top_of_stack_is_actually_top_of_stack(
        #[strategy(vec(arb(), NUM_OP_STACK_REGISTERS..100))] stack: Vec<BFieldElement>,
    ) {
        let mut triton_vm_state = TritonVMState::new(&TuiArgs::default()).unwrap();
        triton_vm_state.vm_state.op_stack.stack = stack.clone();
        let top_of_stack = triton_vm_state.top_of_stack();
        prop_assert_eq!(top_of_stack[0], stack[stack.len() - 1]);
        prop_assert_eq!(top_of_stack[1], stack[stack.len() - 2]);
        prop_assert_eq!(top_of_stack[2], stack[stack.len() - 3]);
    }

    #[proptest]
    fn serialize_and_deserialize_non_determinism_to_and_from_json(
        #[strategy(arb())] non_determinism: NonDeterminism<u64>,
    ) {
        let serialized = serde_json::to_string(&non_determinism).unwrap();
        let deserialized: NonDeterminism<u64> = serde_json::from_str(&serialized).unwrap();
        prop_assert_eq!(non_determinism, deserialized);
    }

    #[test]
    fn serialize_example_program_and_input_to_json() {
        let args = args_for_test_program_with_test_input();
        let program = TritonVMState::program_from_args(&args).unwrap();
        let mut state = TritonVMState::vm_state_with_specified_input(&args, &program).unwrap();
        while state.op_stack.len() <= NUM_OP_STACK_REGISTERS + 4 {
            state.step().unwrap();
        }
        let serialized = serde_json::to_string(&state).unwrap();
        println!("{serialized}");
    }

    #[test]
    fn starting_tui_with_initial_state_makes_type_hint_stack_have_correct_length() {
        let args = args_for_test_program_with_initial_state();
        let state = TritonVMState::new(&args).unwrap();
        assert!(state.vm_state.op_stack.len() == state.type_hints.stack.len());
    }
}
