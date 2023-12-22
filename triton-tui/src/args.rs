use clap::value_parser;
use clap::Args;
use clap::Parser;

use crate::utils::version;

pub(crate) const DEFAULT_INTERRUPT_CYCLE: u32 = 100_000;
pub(crate) const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
pub(crate) const EXAMPLE_PROGRAM_PATH: &str = "examples/program.tasm";

#[derive(Debug, Clone, PartialEq, Parser)]
#[command(author, version = version(), about)]
pub(crate) struct TuiArgs {
    /// File containing the program to run
    pub program: String,

    #[command(flatten)]
    pub input_args: Option<InputArgs>,

    /// JSON file containing entire initial state
    #[arg(long, value_name = "PATH", group = "state")]
    pub initial_state: Option<String>,

    /// The maximum number of cycles to run after any interaction, preventing a frozen TUI in infinite loops
    #[arg(
        long,
        value_name = "u32",
        default_value = DEFAULT_INTERRUPT_CYCLE.to_string(),
        value_parser = value_parser!(u32).range(1..)
    )]
    pub interrupt_cycle: u32,
}

#[derive(Debug, Clone, PartialEq, Args)]
#[group(required = false, multiple = true, conflicts_with = "state")]
pub(crate) struct InputArgs {
    /// File containing public input
    #[arg(short, long, value_name = "PATH")]
    pub input: Option<String>,

    /// JSON file containing all non-determinism
    #[arg(short, long, value_name = "PATH")]
    pub non_determinism: Option<String>,
}

impl Default for TuiArgs {
    fn default() -> Self {
        let program = format!("{MANIFEST_DIR}/{EXAMPLE_PROGRAM_PATH}");
        Self {
            program,
            input_args: None,
            initial_state: None,
            interrupt_cycle: DEFAULT_INTERRUPT_CYCLE,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use assert2::let_assert;

    use super::*;

    pub(crate) const EXAMPLE_INPUT_PATH: &str = "examples/public_input.txt";
    pub(crate) const EXAMPLE_NON_DETERMINISM_PATH: &str = "examples/non_determinism.json";

    fn binary_name() -> Vec<String> {
        vec!["triton-tui".into()]
    }

    fn tui_arg_help() -> Vec<String> {
        vec!["--help".into()]
    }

    fn tui_arg_program() -> Vec<String> {
        vec!["program.tasm".into()]
    }

    fn tui_arg_public_input() -> Vec<String> {
        vec!["--input".into(), "my_input.txt".into()]
    }

    fn tui_arg_non_determinism() -> Vec<String> {
        vec!["--non-determinism".into(), "my_non_determinism.json".into()]
    }

    fn tui_arg_initial_state() -> Vec<String> {
        vec!["--initial-state".into(), "my_state.json".into()]
    }

    pub(crate) fn args_for_test_program_with_test_input() -> TuiArgs {
        let program_path = format!("{MANIFEST_DIR}/{EXAMPLE_PROGRAM_PATH}");
        let input_path = format!("{MANIFEST_DIR}/{EXAMPLE_INPUT_PATH}");
        let non_determinism_path = format!("{MANIFEST_DIR}/{EXAMPLE_NON_DETERMINISM_PATH}");

        let args = [
            binary_name(),
            vec![program_path],
            vec!["-i".into(), input_path],
            vec!["-n".into(), non_determinism_path],
        ]
        .concat();
        TuiArgs::parse_from(args)
    }

    #[test]
    fn tui_requires_some_arguments() {
        let args = binary_name();
        let_assert!(Err(_) = TuiArgs::try_parse_from(args));
    }

    #[test]
    fn argument_help_is_valid() {
        let args = [binary_name(), tui_arg_help()].concat();
        TuiArgs::parse_from(args);
    }

    #[test]
    fn argument_just_program_is_valid() {
        let args = [binary_name(), tui_arg_program()].concat();
        TuiArgs::parse_from(args);
    }

    #[test]
    fn argument_program_and_public_input_is_valid() {
        let args = [binary_name(), tui_arg_program(), tui_arg_public_input()].concat();
        TuiArgs::parse_from(args);
    }

    #[test]
    fn argument_program_and_secret_input_is_valid() {
        let args = [binary_name(), tui_arg_program(), tui_arg_non_determinism()].concat();
        TuiArgs::parse_from(args);
    }

    #[test]
    fn argument_program_and_public_input_and_secret_input_is_valid() {
        let args = [
            binary_name(),
            tui_arg_program(),
            tui_arg_public_input(),
            tui_arg_non_determinism(),
        ]
        .concat();
        TuiArgs::parse_from(args);
    }

    #[test]
    fn argument_program_and_initial_state_is_valid() {
        let args = [binary_name(), tui_arg_program(), tui_arg_initial_state()].concat();
        TuiArgs::parse_from(args);
    }

    #[test]
    fn argument_initial_state_conflicts_with_public_input() {
        let args = [
            binary_name(),
            tui_arg_program(),
            tui_arg_public_input(),
            tui_arg_initial_state(),
        ]
        .concat();
        let_assert!(Err(_) = TuiArgs::try_parse_from(args));
    }

    #[test]
    fn argument_initial_state_conflicts_with_non_determinism() {
        let args = [
            binary_name(),
            tui_arg_program(),
            tui_arg_public_input(),
            tui_arg_initial_state(),
        ]
        .concat();
        let_assert!(Err(_) = TuiArgs::try_parse_from(args));
    }
}
