pub(crate) mod action;
pub(crate) mod args;
pub(crate) mod components;
pub(crate) mod config;
pub(crate) mod mode;
pub(crate) mod triton_tui;
pub(crate) mod tui;
pub(crate) mod utils;

use args::Args;
use clap::Parser;
use color_eyre::eyre::Result;

use crate::triton_tui::TritonTUI;
use crate::utils::initialize_logging;
use crate::utils::initialize_panic_handler;

#[tokio::main]
async fn main() -> Result<()> {
    initialize_logging()?;
    initialize_panic_handler()?;

    let args = Args::parse();
    let mut tui = TritonTUI::new(args)?;
    tui.run().await
}
