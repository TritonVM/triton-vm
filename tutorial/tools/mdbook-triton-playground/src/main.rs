use std::io;
use std::io::Read;

use anyhow::Result;
use clap::Parser;
use mdbook_preprocessor::Preprocessor;
use mdbook_triton_playground::TritonPlayPreprocessor;

#[derive(Parser, Debug)]
#[command(name = "mdbook-triton-playground")]
struct Args {
    /// mdBook compatibility probe: `supports <renderer>`
    #[arg()]
    command: Option<String>,

    #[arg()]
    renderer: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let preprocessor = TritonPlayPreprocessor::new();

    if args.command.as_deref() == Some("supports") {
        let ok = preprocessor.supports_renderer(args.renderer.as_deref().unwrap_or(""))?;
        std::process::exit(if ok { 0 } else { 1 });
    }

    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let (context, book) = mdbook_preprocessor::parse_input(input.as_bytes())?;
    let processed = preprocessor.run(&context, book)?;
    serde_json::to_writer(io::stdout(), &processed)?;

    Ok(())
}
