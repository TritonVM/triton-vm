use clap::Parser;
use color_eyre::eyre::Result;
use tracing::error;

use args::Args;

use crate::triton_tui::TritonTUI;
use crate::utils::initialize_logging;
use crate::utils::initialize_panic_handler;

pub(crate) mod action;
pub(crate) mod args;
pub(crate) mod components;
pub(crate) mod config;
pub(crate) mod element_type_hint;
pub(crate) mod mode;
pub(crate) mod triton_tui;
pub(crate) mod triton_vm_state;
pub(crate) mod tui;
pub(crate) mod type_hint_stack;
pub(crate) mod utils;

#[tokio::main]
async fn main() -> Result<()> {
    initialize_logging()?;
    initialize_panic_handler()?;

    let args = Args::parse();
    let mut triton_tui = TritonTUI::new(args)?;
    if let Err(e) = triton_tui.run().await {
        let error = format!("{e}");
        error!(error);
        triton_tui.terminate()?;
    };
    Ok(())
}
