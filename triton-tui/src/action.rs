use std::fmt;

use arbitrary::Arbitrary;
use itertools::Itertools;
use serde::de::*;
use serde::*;

use triton_vm::instruction::Instruction;
use triton_vm::op_stack::NUM_OP_STACK_REGISTERS;
use triton_vm::BFieldElement;

use crate::mode::Mode;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) enum Action {
    Tick,
    Render,
    Resize(u16, u16),
    Suspend,
    Resume,
    Quit,
    Refresh,
    Error(String),

    Execute(Execute),

    /// Undo the last [`Execute`] action.
    Undo,

    RecordUndoInfo,

    /// Reset the program state.
    Reset,

    Toggle(ToggleWidget),

    HideHelpScreen,

    Mode(Mode),

    ExecutedInstruction(Box<ExecutedInstruction>),
}

/// Various ways to advance the program state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) enum Execute {
    /// Continue program execution until next breakpoint.
    Continue,

    /// Execute a single instruction.
    Step,

    /// Execute a single instruction, stepping over `call`s.
    Next,

    /// Execute instructions until the current `call` returns.
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Arbitrary)]
pub(crate) enum ToggleWidget {
    TypeHint,
    CallStack,
    Input,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Arbitrary)]
pub(crate) struct ExecutedInstruction {
    pub instruction: Instruction,
    pub old_top_of_stack: [BFieldElement; NUM_OP_STACK_REGISTERS],
    pub new_top_of_stack: [BFieldElement; NUM_OP_STACK_REGISTERS],
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ActionVisitor;

        impl<'de> Visitor<'de> for ActionVisitor {
            type Value = Action;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a valid string representation of Action")
            }

            fn visit_str<E>(self, value: &str) -> Result<Action, E>
            where
                E: Error,
            {
                match value {
                    "Tick" => Ok(Action::Tick),
                    "Render" => Ok(Action::Render),
                    "Suspend" => Ok(Action::Suspend),
                    "Resume" => Ok(Action::Resume),
                    "Quit" => Ok(Action::Quit),
                    "Refresh" => Ok(Action::Refresh),

                    "Continue" => Ok(Action::Execute(Execute::Continue)),
                    "Step" => Ok(Action::Execute(Execute::Step)),
                    "Next" => Ok(Action::Execute(Execute::Next)),
                    "Finish" => Ok(Action::Execute(Execute::Finish)),

                    "Undo" => Ok(Action::Undo),
                    "Reset" => Ok(Action::Reset),

                    "ToggleTypeHintDisplay" => Ok(Action::Toggle(ToggleWidget::TypeHint)),
                    "ToggleCallStackDisplay" => Ok(Action::Toggle(ToggleWidget::CallStack)),
                    "ToggleInputDisplay" => Ok(Action::Toggle(ToggleWidget::Input)),

                    "HideHelpScreen" => Ok(Action::HideHelpScreen),

                    mode if mode.starts_with("Mode::") => Self::parse_mode(mode),
                    data if data.starts_with("Error(") => Self::parse_error(data),
                    data if data.starts_with("Resize(") => Self::parse_resize(data),
                    _ => Err(E::custom(format!("Unknown Action variant: {value}"))),
                }
            }
        }

        impl ActionVisitor {
            fn parse_mode<E>(mode: &str) -> Result<Action, E>
            where
                E: Error,
            {
                let maybe_mode_and_variant = mode.split("::").collect_vec();
                let maybe_variant = maybe_mode_and_variant.get(1).copied();
                let mode_variant =
                    maybe_variant.ok_or(E::custom(format!("Missing Mode variant: {mode}")))?;
                let mode = Mode::deserialize(mode_variant.into_deserializer())?;
                Ok(Action::Mode(mode))
            }

            fn parse_error<E>(data: &str) -> Result<Action, E>
            where
                E: Error,
            {
                let error_msg = data.trim_start_matches("Error(").trim_end_matches(')');
                Ok(Action::Error(error_msg.to_string()))
            }

            fn parse_resize<E>(data: &str) -> Result<Action, E>
            where
                E: Error,
            {
                let parts: Vec<&str> = data
                    .trim_start_matches("Resize(")
                    .trim_end_matches(')')
                    .split(',')
                    .collect();
                if parts.len() == 2 {
                    let width: u16 = parts[0].trim().parse().map_err(E::custom)?;
                    let height: u16 = parts[1].trim().parse().map_err(E::custom)?;
                    Ok(Action::Resize(width, height))
                } else {
                    Err(E::custom(format!("Invalid Resize format: {data}")))
                }
            }
        }

        deserializer.deserialize_str(ActionVisitor)
    }
}

impl ExecutedInstruction {
    pub fn new(
        instruction: Instruction,
        old_top_of_stack: [BFieldElement; NUM_OP_STACK_REGISTERS],
        new_top_of_stack: [BFieldElement; NUM_OP_STACK_REGISTERS],
    ) -> Self {
        Self {
            instruction,
            old_top_of_stack,
            new_top_of_stack,
        }
    }
}
