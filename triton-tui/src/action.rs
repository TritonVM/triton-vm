use std::fmt;

use arbitrary::Arbitrary;
use itertools::Itertools;
use serde::de::*;
use serde::*;
use triton_vm::instruction::Instruction;

use crate::mode::Mode;
use crate::shadow_memory::TopOfStack;

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
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

    Toggle(Toggle),

    HideHelpScreen,

    Mode(Mode),

    ExecutedInstruction(Box<ExecutedInstruction>),
}

/// Various ways to advance the program state.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Arbitrary)]
pub(crate) enum Toggle {
    All,
    TypeHint,
    CallStack,
    SpongeState,
    Input,
    BlockAddress,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Arbitrary)]
pub(crate) struct ExecutedInstruction {
    pub instruction: Instruction,
    pub old_top_of_stack: TopOfStack,
    pub new_top_of_stack: TopOfStack,
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

                    "ToggleAll" => Ok(Action::Toggle(Toggle::All)),
                    "ToggleTypeHintDisplay" => Ok(Action::Toggle(Toggle::TypeHint)),
                    "ToggleCallStackDisplay" => Ok(Action::Toggle(Toggle::CallStack)),
                    "ToggleSpongeStateDisplay" => Ok(Action::Toggle(Toggle::SpongeState)),
                    "ToggleInputDisplay" => Ok(Action::Toggle(Toggle::Input)),
                    "ToggleBlockAddressDisplay" => Ok(Action::Toggle(Toggle::BlockAddress)),

                    "HideHelpScreen" => Ok(Action::HideHelpScreen),

                    mode if mode.starts_with("Mode::") => Self::parse_mode(mode),
                    data if data.starts_with("Error(") => Ok(Self::parse_error(data)),
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

            fn parse_error(data: &str) -> Action {
                let error_msg = data.trim_start_matches("Error(").trim_end_matches(')');
                Action::Error(error_msg.to_string())
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
                let [width, height] = parts[..] else {
                    return Err(E::custom(format!("Invalid Resize format: {data}")));
                };

                let width: u16 = width.trim().parse().map_err(E::custom)?;
                let height: u16 = height.trim().parse().map_err(E::custom)?;
                Ok(Action::Resize(width, height))
            }
        }

        deserializer.deserialize_str(ActionVisitor)
    }
}

impl ExecutedInstruction {
    pub fn new(
        instruction: Instruction,
        old_top_of_stack: TopOfStack,
        new_top_of_stack: TopOfStack,
    ) -> Self {
        Self {
            instruction,
            old_top_of_stack,
            new_top_of_stack,
        }
    }
}
