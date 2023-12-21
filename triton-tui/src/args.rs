use clap::Parser;

use crate::utils::version;

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
const DEFAULT_PROGRAM_PATH: &str = "examples/program.tasm";

#[derive(Debug, Clone, PartialEq, Parser)]
#[command(author, version = version(), about)]
pub(crate) struct Args {
    #[arg(short, long, value_name = "PATH")]
    /// Path to program to run
    pub program: String,

    #[arg(short, long, value_name = "PATH")]
    /// Path to file containing public input
    pub input: Option<String>,

    #[arg(short, long, value_name = "PATH")]
    /// Path to JSON file containing all non-determinism
    pub non_determinism: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        let program = format!("{MANIFEST_DIR}/{DEFAULT_PROGRAM_PATH}");
        Self {
            program,
            input: None,
            non_determinism: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use assert2::let_assert;

    use super::*;

    #[test]
    fn tui_requires_some_arguments() {
        let cli_args: Vec<String> = vec![];
        let_assert!(Err(_) = Args::try_parse_from(cli_args));
    }
}
