use clap::Parser;
use color_eyre::eyre::Result;
use tracing::error;

use args::TuiArgs;

use crate::triton_tui::TritonTUI;
use crate::utils::initialize_logging;
use crate::utils::initialize_panic_handler;

pub(crate) mod action;
pub(crate) mod args;
pub(crate) mod components;
pub(crate) mod config;
pub(crate) mod element_type_hint;
pub(crate) mod mode;
pub(crate) mod shadow_memory;
pub(crate) mod triton_tui;
pub(crate) mod triton_vm_state;
pub(crate) mod tui;
pub(crate) mod utils;

#[tokio::main]
async fn main() -> Result<()> {
    initialize_logging()?;
    initialize_panic_handler()?;

    let args = TuiArgs::parse();
    let mut triton_tui = TritonTUI::new(args)?;
    let execution_result = triton_tui.run().await;
    if let Err(ref err) = execution_result {
        error!("{err}");
        triton_tui.terminate()?;
    };
    execution_result
}
