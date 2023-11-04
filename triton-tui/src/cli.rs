use clap::Parser;

use crate::utils::version;

const DEFAULT_TICK_RATE: f64 = 1.0;
const DEFAULT_FRAME_RATE: f64 = 16.0;

#[derive(Parser, Debug)]
#[command(author, version = version(), about)]
pub(crate) struct Cli {
    #[arg(
        short,
        long,
        value_name = "FLOAT",
        help = "Tick rate, i.e. number of ticks per second",
        default_value_t = DEFAULT_TICK_RATE
    )]
    pub tick_rate: f64,

    #[arg(
        short,
        long,
        value_name = "FLOAT",
        help = "Frame rate, i.e. number of frames per second",
        default_value_t = DEFAULT_FRAME_RATE
    )]
    pub frame_rate: f64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
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
        let args = Cli::parse_from(cli_args);
        assert_eq!(DEFAULT_TICK_RATE, args.tick_rate);
        assert_eq!(DEFAULT_FRAME_RATE, args.frame_rate);
    }
}
