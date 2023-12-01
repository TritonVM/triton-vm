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
        help = "path to program to run",
        default_value_t = String::from("./program.tasm"),
    )]
    pub program_path: String,

    #[arg(
        short,
        long,
        value_name = "FLOAT",
        help = "tick rate, i.e. number of ticks per second",
        default_value_t = DEFAULT_TICK_RATE
    )]
    pub tick_rate: f64,

    #[arg(
        short,
        long,
        value_name = "FLOAT",
        help = "frame rate, i.e. number of frames per second",
        default_value_t = DEFAULT_FRAME_RATE
    )]
    pub frame_rate: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            program_path: DEFAULT_PROGRAM_PATH.into(),
            tick_rate: DEFAULT_TICK_RATE,
            frame_rate: DEFAULT_FRAME_RATE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cli_and_clap_default_parsing_produce_same_values() {
        let cli_args: Vec<String> = vec![];
        let args = Args::parse_from(cli_args);
        assert_eq!(DEFAULT_PROGRAM_PATH, args.program_path);
        assert_eq!(DEFAULT_TICK_RATE, args.tick_rate);
        assert_eq!(DEFAULT_FRAME_RATE, args.frame_rate);
    }
}
