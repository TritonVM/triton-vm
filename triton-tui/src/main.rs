//! Triton TUI is a terminal user interface for Triton VM. It allows executing and debugging
//! programs in Triton assembly. It is intended to be used as a standalone application;
//! see the [README](https://crates.io/crates/triton-tui) for more information.

use clap::Parser;
use color_eyre::eyre::Result;
use tracing::error;
use tracing_error::ErrorLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

use args::get_data_dir;
use args::TuiArgs;
use args::LOG_ENV;
use args::LOG_FILE;
use triton_tui::TritonTUI;

pub(crate) mod action;
pub(crate) mod args;
#[cfg(test)]
pub(crate) mod args_tests;
pub(crate) mod components;
pub(crate) mod config;
pub(crate) mod element_type_hint;
pub(crate) mod mode;
pub(crate) mod shadow_memory;
pub(crate) mod triton_tui;
pub(crate) mod triton_vm_state;
pub(crate) mod tui;

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

fn initialize_logging() -> Result<()> {
    let directory = get_data_dir();
    std::fs::create_dir_all(directory.clone())?;
    let log_path = directory.join(LOG_FILE.clone());
    let log_file = std::fs::File::create(log_path)?;
    std::env::set_var(
        "RUST_LOG",
        std::env::var("RUST_LOG")
            .or_else(|_| std::env::var(LOG_ENV.clone()))
            .unwrap_or_else(|_| format!("{}=info", env!("CARGO_CRATE_NAME"))),
    );
    let file_subscriber = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .with_writer(log_file)
        .with_target(false)
        .with_ansi(false)
        .with_filter(tracing_subscriber::filter::EnvFilter::from_default_env());
    tracing_subscriber::registry()
        .with(file_subscriber)
        .with(ErrorLayer::default())
        .init();
    Ok(())
}

fn initialize_panic_handler() -> Result<()> {
    let (panic_hook, eyre_hook) = color_eyre::config::HookBuilder::default()
        .panic_section(format!(
            "This is a bug. Consider reporting it at {}",
            env!("CARGO_PKG_REPOSITORY")
        ))
        .capture_span_trace_by_default(false)
        .display_location_section(false)
        .display_env_section(false)
        .into_hooks();
    eyre_hook.install()?;
    std::panic::set_hook(Box::new(move |panic_info| {
        if let Ok(mut t) = crate::tui::Tui::new() {
            if let Err(r) = t.exit() {
                error!("Unable to exit Terminal: {:?}", r);
            }
        }

        #[cfg(not(debug_assertions))]
        {
            use human_panic::handle_dump;
            use human_panic::print_msg;
            use human_panic::Metadata;
            let meta = Metadata {
                version: env!("CARGO_PKG_VERSION").into(),
                name: env!("CARGO_PKG_NAME").into(),
                authors: env!("CARGO_PKG_AUTHORS").replace(':', ", ").into(),
                homepage: env!("CARGO_PKG_HOMEPAGE").into(),
            };

            let file_path = handle_dump(&meta, panic_info);
            // prints human-panic message
            print_msg(file_path, &meta)
                .expect("human-panic: printing error message to console failed");
            // prints color-eyre stack trace to stderr
            eprintln!("{}", panic_hook.panic_report(panic_info));
        }
        let msg = format!("{}", panic_hook.panic_report(panic_info));
        error!("Error: {}", strip_ansi_escapes::strip_str(msg));

        #[cfg(debug_assertions)]
        {
            // Better Panic stacktrace that is only enabled when debugging.
            better_panic::Settings::auto()
                .most_recent_first(false)
                .lineno_suffix(true)
                .verbosity(better_panic::Verbosity::Full)
                .create_panic_handler()(panic_info);
        }

        std::process::exit(libc::EXIT_FAILURE);
    }));
    Ok(())
}
