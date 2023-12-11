use color_eyre::eyre::anyhow;
use color_eyre::eyre::Result;
use color_eyre::Report;
use fs_err as fs;
use tracing::info;

use triton_vm::error::InstructionError;
use triton_vm::instruction::*;
use triton_vm::vm::VMState;
use triton_vm::*;

use crate::action::Action;
use crate::args::Args;
use crate::type_hint_stack::TypeHintStack;

#[derive(Debug)]
pub(crate) struct TritonVMState {
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
        self.warning = None;
        let current_instruction = self.vm_state.current_instruction().ok();
        match self.vm_state.step() {
            Ok(_) => self.type_hint_stack.mimic_instruction(current_instruction),
            Err(err) => self.error = Some(err),
        }
        self.apply_type_hints();
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

    pub fn update(&mut self, action: Action) -> Result<Option<Action>> {
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
