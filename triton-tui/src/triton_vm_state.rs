use color_eyre::eyre::anyhow;
use color_eyre::eyre::Result;
use color_eyre::Report;
use fs_err as fs;
use itertools::Itertools;
use tokio::sync::mpsc::UnboundedSender;
use tracing::error;
use tracing::info;

use triton_vm::error::InstructionError;
use triton_vm::instruction::*;
use triton_vm::op_stack::NUM_OP_STACK_REGISTERS;
use triton_vm::vm::VMState;
use triton_vm::*;

use crate::action::Action;
use crate::action::ExecutedInstruction;
use crate::args::Args;
use crate::components::Component;
use crate::type_hint_stack::TypeHintStack;

#[derive(Debug)]
pub(crate) struct TritonVMState {
    pub action_tx: Option<UnboundedSender<Action>>,

    pub program: Program,
    pub vm_state: VMState,

    pub type_hint_stack: TypeHintStack,
    pub undo_stack: Vec<UndoInformation>,

    pub warning: Option<Report>,
    pub error: Option<InstructionError>,
}

#[derive(Debug, Clone)]
pub(crate) struct UndoInformation {
    vm_state: VMState,
    type_hint_stack: TypeHintStack,
}

impl TritonVMState {
    pub fn new(args: &Args) -> Result<Self> {
        let program = Self::program_from_args(args)?;
        let public_input = Self::public_input_from_args(args)?;
        let non_determinism = Self::non_determinism_from_args(args)?;

        let vm_state = VMState::new(&program, public_input.clone(), non_determinism);

        let mut state = Self {
            action_tx: None,
            program,
            vm_state,
            type_hint_stack: TypeHintStack::default(),
            undo_stack: vec![],
            warning: None,
            error: None,
        };
        state.apply_type_hints();
        Ok(state)
    }

    fn program_from_args(args: &Args) -> Result<Program> {
        let source_code = fs::read_to_string(&args.program)?;
        let program = Program::from_code(&source_code)
            .map_err(|err| anyhow!("program parsing error: {err}"))?;
        Ok(program)
    }

    fn public_input_from_args(args: &Args) -> Result<PublicInput> {
        let Some(input_path) = args.input.clone() else {
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

    fn non_determinism_from_args(_args: &Args) -> Result<NonDeterminism<BFieldElement>> {
        Ok(NonDeterminism::default())
    }

    fn top_of_stack(&self) -> [BFieldElement; NUM_OP_STACK_REGISTERS] {
        let stack_len = self.vm_state.op_stack.stack.len();
        let index_of_lowest_accessible_element = stack_len - NUM_OP_STACK_REGISTERS;
        let top_of_stack = &self.vm_state.op_stack.stack[index_of_lowest_accessible_element..];
        let top_of_stack = top_of_stack.iter().copied();
        top_of_stack.rev().collect_vec().try_into().unwrap()
    }

    fn vm_has_stopped(&self) -> bool {
        self.vm_state.halting || self.error.is_some()
    }

    fn vm_is_running(&self) -> bool {
        !self.vm_has_stopped()
    }

    fn at_breakpoint(&self) -> bool {
        let ip = self.vm_state.instruction_pointer as u64;
        self.program.is_breakpoint(ip)
    }

    fn apply_type_hints(&mut self) {
        let ip = self.vm_state.instruction_pointer as u64;
        for type_hint in self.program.type_hints_at(ip) {
            let maybe_error = self.type_hint_stack.apply_type_hint(type_hint);
            if let Err(report) = maybe_error {
                info!("Error applying type hint: {report}");
                self.warning = Some(report);
            };
        }
    }

    /// Handle [`Action::ProgramContinue`].
    fn continue_execution(&mut self) {
        self.step();
        while self.vm_is_running() && !self.at_breakpoint() {
            self.step();
        }
    }

    /// Handle [`Action::ProgramStep`].
    fn step(&mut self) {
        if self.vm_has_stopped() {
            return;
        }

        let instruction = self.vm_state.current_instruction().ok();
        let old_top_of_stack = self.top_of_stack();
        if let Err(err) = self.vm_state.step() {
            self.error = Some(err);
            return;
        }
        self.warning = None;

        let instruction = instruction.expect("instruction should exist after successful `step`");
        let new_top_of_stack = self.top_of_stack();
        let executed_instruction =
            ExecutedInstruction::new(instruction, old_top_of_stack, new_top_of_stack);

        self.send_executed_transaction(executed_instruction);
        self.type_hint_stack.mimic_instruction(executed_instruction);
        self.apply_type_hints();
    }

    fn send_executed_transaction(&mut self, executed_instruction: ExecutedInstruction) {
        let Some(ref action_tx) = self.action_tx else {
            error!("action_tx should exist");
            return;
        };
        let _ = action_tx.send(Action::ExecutedInstruction(Box::new(executed_instruction)));
    }

    /// Handle [`Action::ProgramNext`].
    fn next(&mut self) {
        let instruction = self.vm_state.current_instruction();
        let instruction_is_call = matches!(instruction, Ok(Instruction::Call(_)));
        self.step();
        if instruction_is_call {
            self.finish();
        }
    }

    /// Handle [`Action::ProgramFinish`].
    fn finish(&mut self) {
        let current_jump_stack_depth = self.vm_state.jump_stack.len();
        while self.vm_is_running() && self.vm_state.jump_stack.len() >= current_jump_stack_depth {
            self.step();
        }
    }

    fn record_undo_information(&mut self) {
        let undo_information = UndoInformation {
            vm_state: self.vm_state.clone(),
            type_hint_stack: self.type_hint_stack.clone(),
        };
        self.undo_stack.push(undo_information);
    }

    fn program_undo(&mut self) {
        let Some(undo_information) = self.undo_stack.pop() else {
            self.warning = Some(anyhow!("no more undo information available"));
            return;
        };
        self.warning = None;
        self.error = None;
        self.vm_state = undo_information.vm_state;
        self.type_hint_stack = undo_information.type_hint_stack;
    }
}
impl Component for TritonVMState {
    fn register_action_handler(&mut self, tx: UnboundedSender<Action>) -> Result<()> {
        self.action_tx = Some(tx);
        Ok(())
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::ProgramContinue => {
                self.record_undo_information();
                self.continue_execution();
            }
            Action::ProgramStep => {
                self.record_undo_information();
                self.step();
            }
            Action::ProgramNext => {
                self.record_undo_information();
                self.next();
            }
            Action::ProgramFinish => {
                self.record_undo_information();
                self.finish();
            }
            Action::ProgramUndo => self.program_undo(),
            _ => {}
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use proptest::collection::vec;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn presumed_top_of_stack_is_actually_top_of_stack(
        #[strategy(vec(arb(), NUM_OP_STACK_REGISTERS..100))] stack: Vec<BFieldElement>,
    ) {
        let mut triton_vm_state = TritonVMState::new(&Default::default()).unwrap();
        triton_vm_state.vm_state.op_stack.stack = stack.clone();
        let top_of_stack = triton_vm_state.top_of_stack();
        assert!(top_of_stack[0] == stack[stack.len() - 1]);
        assert!(top_of_stack[1] == stack[stack.len() - 2]);
        assert!(top_of_stack[2] == stack[stack.len() - 3]);
    }
}
