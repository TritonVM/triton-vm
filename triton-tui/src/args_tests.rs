//! Tests for the command line arguments of the TUI.
//! Lives in a dedicated file because `src/args.rs` is `include!`d in `build.rs`.

use assert2::let_assert;
use clap::Parser;

use crate::args::*;

pub const EXAMPLE_INPUT_PATH: &str = "examples/public_input.txt";
pub const EXAMPLE_NON_DETERMINISM_PATH: &str = "examples/non_determinism.json";
pub const EXAMPLE_INITIAL_STATE_PATH: &str = "examples/initial_state.json";

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

pub fn args_for_test_program_with_test_input() -> TuiArgs {
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

pub fn args_for_test_program_with_initial_state() -> TuiArgs {
    let program_path = format!("{MANIFEST_DIR}/{EXAMPLE_PROGRAM_PATH}");
    let initial_state_path = format!("{MANIFEST_DIR}/{EXAMPLE_INITIAL_STATE_PATH}");

    let args = [
        binary_name(),
        vec![program_path],
        vec!["--initial-state".into(), initial_state_path],
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
