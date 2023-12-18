use std::io::Write;
use std::path::PathBuf;

use assert2::assert;
use assert2::let_assert;
use pty_process::blocking::Command;
use pty_process::blocking::Pty;
use pty_process::Size;

/// The directory containing the Cargo.toml file.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn execute_then_terminate_triton_tui_with_trivial_program() {
    let path_to_trivial_program = manifest_dir().join("tests/trivial_program.tasm");
    let_assert!(Some(path_to_trivial_program) = path_to_trivial_program.to_str());
    println!("{path_to_trivial_program}");

    let mut pty = Pty::new().unwrap();
    let_assert!(Ok(()) = pty.resize(Size::new(24, 80)));
    let_assert!(Ok(pts) = pty.pts());

    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--bin", "triton-tui", "--"]);
    cmd.args(["-p", path_to_trivial_program]);
    let_assert!(Ok(mut child) = cmd.spawn(&pts));

    let_assert!(Ok(()) = write!(pty, "q"));

    let_assert!(Ok(exit_status) = child.wait());
    assert!(Some(0) == exit_status.code());
}
