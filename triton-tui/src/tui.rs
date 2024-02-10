use std::ops::Deref;
use std::ops::DerefMut;
use std::time::Duration;

use color_eyre::eyre::bail;
use color_eyre::eyre::Result;
use crossterm::event::Event as CrosstermEvent;
use crossterm::event::*;
use crossterm::terminal::*;
use crossterm::tty::IsTty;
use crossterm::*;
use futures::*;
use ratatui::backend::CrosstermBackend as Backend;
use ratatui::Terminal;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::mpsc::*;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;
use tracing::error;

use crate::args::TuiArgs;

pub(crate) type IO = std::io::Stdout;

pub(crate) fn io() -> IO {
    std::io::stdout()
}

const DEFAULT_TICK_RATE: f64 = 1.0;
const DEFAULT_FRAME_RATE: f64 = 32.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Event {
    Init,
    Quit,
    Error,
    Closed,
    Tick,
    Render,
    FocusGained,
    FocusLost,
    Paste(String),
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
}

#[derive(Debug)]
pub(crate) struct Tui {
    pub terminal: Terminal<Backend<IO>>,
    pub task: JoinHandle<()>,
    pub cancellation_token: CancellationToken,
    pub event_rx: UnboundedReceiver<Event>,
    pub event_tx: UnboundedSender<Event>,
    pub frame_rate: f64,
    pub tick_rate: f64,
    pub mouse: bool,
    pub paste: bool,
}

impl Tui {
    pub fn new() -> Result<Self> {
        if !io().is_tty() {
            error!("not a TTY");
            bail!("not a TTY");
        }

        let tick_rate = DEFAULT_TICK_RATE;
        let frame_rate = DEFAULT_FRAME_RATE;
        let terminal = Terminal::new(Backend::new(io()))?;
        let (event_tx, event_rx) = unbounded_channel();
        let cancellation_token = CancellationToken::new();
        let task = tokio::spawn(async {});
        Ok(Self {
            terminal,
            task,
            cancellation_token,
            event_rx,
            event_tx,
            frame_rate,
            tick_rate,
            mouse: true,
            paste: true,
        })
    }

    pub fn apply_args(&mut self, _: &TuiArgs) -> &mut Self {
        self.frame_rate(DEFAULT_FRAME_RATE);
        self.mouse(true);
        self.paste(true);
        self
    }

    pub fn frame_rate(&mut self, frame_rate: f64) -> &mut Self {
        self.frame_rate = frame_rate;
        self
    }

    pub fn mouse(&mut self, mouse: bool) -> &mut Self {
        self.mouse = mouse;
        self
    }

    pub fn paste(&mut self, paste: bool) -> &mut Self {
        self.paste = paste;
        self
    }

    pub fn start(&mut self) {
        let tick_delay = Duration::from_secs_f64(1.0 / self.tick_rate);
        let render_delay = Duration::from_secs_f64(1.0 / self.frame_rate);

        self.cancel();
        self.cancellation_token = CancellationToken::new();
        let cancellation_token = self.cancellation_token.clone();

        let event_tx = self.event_tx.clone();
        self.task = tokio::spawn(async move {
            let mut reader = EventStream::new();
            let mut tick_interval = interval(tick_delay);
            let mut render_interval = interval(render_delay);
            event_tx.send(Event::Init).unwrap();
            loop {
                let tick_delay = tick_interval.tick();
                let render_delay = render_interval.tick();
                let crossterm_event = reader.next().fuse();
                tokio::select! {
                    event = crossterm_event => Self::handle_crossterm_event(&event_tx, event),
                    _ = cancellation_token.cancelled() => return,
                    _ = tick_delay => event_tx.send(Event::Tick).unwrap(),
                    _ = render_delay => event_tx.send(Event::Render).unwrap(),
                }
            }
        });
    }

    fn handle_crossterm_event(
        event_tx: &UnboundedSender<Event>,
        maybe_event: Option<io::Result<CrosstermEvent>>,
    ) {
        let Some(event_result) = maybe_event else {
            return;
        };
        let Ok(event) = event_result else {
            return event_tx.send(Event::Error).unwrap();
        };

        match event {
            CrosstermEvent::Key(key) => {
                if key.kind == KeyEventKind::Press {
                    event_tx.send(Event::Key(key)).unwrap()
                }
            }
            CrosstermEvent::Mouse(mouse) => event_tx.send(Event::Mouse(mouse)).unwrap(),
            CrosstermEvent::Resize(x, y) => event_tx.send(Event::Resize(x, y)).unwrap(),
            CrosstermEvent::FocusLost => event_tx.send(Event::FocusLost).unwrap(),
            CrosstermEvent::FocusGained => event_tx.send(Event::FocusGained).unwrap(),
            CrosstermEvent::Paste(s) => event_tx.send(Event::Paste(s)).unwrap(),
        }
    }

    pub fn enter(&mut self) -> Result<()> {
        enable_raw_mode()?;
        execute!(io(), EnterAlternateScreen, cursor::Hide)?;
        if self.mouse {
            execute!(io(), EnableMouseCapture)?;
        }
        if self.paste {
            execute!(io(), EnableBracketedPaste)?;
        }
        self.start();
        Ok(())
    }

    pub fn exit(&mut self) -> Result<()> {
        self.stop();
        if is_raw_mode_enabled()? {
            self.flush()?;
            if self.paste {
                execute!(io(), DisableBracketedPaste)?;
            }
            if self.mouse {
                execute!(io(), DisableMouseCapture)?;
            }
            execute!(io(), LeaveAlternateScreen, cursor::Show)?;
            disable_raw_mode()?;
        }
        Ok(())
    }

    pub fn stop(&self) {
        self.cancel();
        let mut counter = 0;
        while !self.task.is_finished() {
            std::thread::sleep(Duration::from_millis(1));
            counter += 1;
            if counter > 50 {
                self.task.abort();
            }
            if counter > 100 {
                error!("Failed to abort task in 100 milliseconds for unknown reason");
                break;
            }
        }
    }

    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    pub fn suspend(&mut self) -> Result<()> {
        self.exit()?;
        #[cfg(not(windows))]
        signal_hook::low_level::raise(signal_hook::consts::signal::SIGTSTP)?;
        Ok(())
    }

    pub fn resume(&mut self) -> Result<()> {
        self.enter()
    }

    pub async fn next(&mut self) -> Option<Event> {
        self.event_rx.recv().await
    }
}

impl Deref for Tui {
    type Target = Terminal<Backend<IO>>;

    fn deref(&self) -> &Self::Target {
        &self.terminal
    }
}

impl DerefMut for Tui {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.terminal
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        self.exit().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::let_assert;

    #[test]
    fn creating_tui_outside_of_tty_gives_error() {
        let_assert!(Err(err) = Tui::new());
        assert!(err.to_string().contains("TTY"));
    }
}
