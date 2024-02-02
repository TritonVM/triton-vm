use std::path::PathBuf;

use clap::value_parser;
use clap::Args;
use clap::Parser;
use directories::ProjectDirs;
use lazy_static::lazy_static;

lazy_static! {
    pub(crate) static ref PROJECT_NAME: String =
        env!("CARGO_CRATE_NAME").to_uppercase().to_string();
    pub(crate) static ref DATA_FOLDER: Option<PathBuf> =
        std::env::var(format!("{}_DATA", PROJECT_NAME.clone()))
            .ok()
            .map(PathBuf::from);
    pub(crate) static ref CONFIG_FOLDER: Option<PathBuf> =
        std::env::var(format!("{}_CONFIG", PROJECT_NAME.clone()))
            .ok()
            .map(PathBuf::from);
    pub(crate) static ref GIT_COMMIT_HASH: String =
        std::env::var(format!("{}_GIT_INFO", PROJECT_NAME.clone()))
            .unwrap_or_else(|_| String::from("UNKNOWN"));
    pub(crate) static ref LOG_ENV: String = format!("{}_LOGLEVEL", PROJECT_NAME.clone());
    pub(crate) static ref LOG_FILE: String = format!("{}.log", env!("CARGO_PKG_NAME"));
}

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

    /// The maximum number of cycles to run after any interaction,
    /// preventing a frozen TUI in infinite loops
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

fn project_directory() -> Option<ProjectDirs> {
    ProjectDirs::from(
        "org.triton-vm.triton-tui",
        "Triton VM",
        env!("CARGO_PKG_NAME"),
    )
}

pub(crate) fn get_data_dir() -> PathBuf {
    DATA_FOLDER
        .clone()
        .or_else(|| project_directory().map(|dirs| dirs.data_local_dir().to_path_buf()))
        .unwrap_or_else(|| PathBuf::from(".").join(".data"))
}

pub(crate) fn get_config_dir() -> PathBuf {
    CONFIG_FOLDER
        .clone()
        .or_else(|| project_directory().map(|dirs| dirs.config_local_dir().to_path_buf()))
        .unwrap_or_else(|| PathBuf::from(".").join(".config"))
}

pub(crate) fn version() -> String {
    let commit_hash = GIT_COMMIT_HASH.clone();
    let author = clap::crate_authors!();
    let config_dir_path = get_config_dir().display().to_string();
    let data_dir_path = get_data_dir().display().to_string();

    format!(
        "{commit_hash}\n\n\
        Authors: {author}\n\n\
        Config directory: {config_dir_path}\n\
        Data directory:   {data_dir_path}"
    )
}
