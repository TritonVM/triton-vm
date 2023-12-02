use clap::Parser;

use crate::utils::version;

const DEFAULT_PROGRAM_PATH: &str = "./program.tasm";
const DEFAULT_TICK_RATE: f64 = 1.0;
const DEFAULT_FRAME_RATE: f64 = 32.0;

#[derive(Parser, Debug, Clone)]
#[command(author, version = version(), about)]
pub(crate) struct Args {
    #[arg(
        short,
        long,
        value_name = "PATH",
        default_value_t = String::from(DEFAULT_PROGRAM_PATH),
    )]
    /// path to program to run
    pub program_path: String,

    #[arg(short, long, value_name = "PATH")]
    /// path to public input file
    pub input_path: Option<String>,

    #[arg(
        short,
        long,
        value_name = "FLOAT",
        default_value_t = DEFAULT_TICK_RATE
    )]
    /// tick rate, i.e. number of ticks per second
    pub tick_rate: f64,

    #[arg(
        short,
        long,
        value_name = "FLOAT",
        default_value_t = DEFAULT_FRAME_RATE
    )]
    /// frame rate, i.e. number of frames per second
    pub frame_rate: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            program_path: DEFAULT_PROGRAM_PATH.into(),
            input_path: None,
            tick_rate: DEFAULT_TICK_RATE,
            frame_rate: DEFAULT_FRAME_RATE,
        }
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;

    use super::*;

    #[test]
    fn default_cli_and_clap_default_parsing_produce_same_values() {
        let cli_args: Vec<String> = vec![];
        let args = Args::parse_from(cli_args);
        assert!(DEFAULT_PROGRAM_PATH == args.program_path);
        assert!(DEFAULT_TICK_RATE == args.tick_rate);
        assert!(DEFAULT_FRAME_RATE == args.frame_rate);
    }
}
