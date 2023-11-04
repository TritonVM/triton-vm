use color_eyre::eyre::Result;
use crossterm::event::KeyEvent;
use ratatui::prelude::Rect;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use tracing::debug;
use tracing::info;

use crate::action::Action;
use crate::components::fps::FpsCounter;
use crate::components::home::Home;
use crate::components::Component;
use crate::config::Config;
use crate::config::KeyEvents;
use crate::mode::Mode;
use crate::tui::*;
use crate::utils::trace_dbg;

pub(crate) struct TritonTUI {
    pub config: Config,
    pub tick_rate: f64,
    pub frame_rate: f64,
    pub components: Vec<Box<dyn Component>>,
    pub should_quit: bool,
    pub should_suspend: bool,
    pub mode: Mode,
    pub recent_key_events: KeyEvents,
}

impl TritonTUI {
    pub fn new(tick_rate: f64, frame_rate: f64) -> Result<Self> {
        let home = Home::new();
        let fps = FpsCounter::default();
        let config = Config::new()?;
        let mode = Mode::Home;
        Ok(Self {
            tick_rate,
            frame_rate,
            components: vec![Box::new(home), Box::new(fps)],
            should_quit: false,
            should_suspend: false,
            config,
            mode,
            recent_key_events: Vec::new(),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        let (action_tx, mut action_rx) = mpsc::unbounded_channel();

        let mut tui = self.tui()?;
        tui.enter()?;

        trace_dbg!("Tui entered");

        for component in self.components.iter_mut() {
            component.register_action_handler(action_tx.clone())?;
        }

        for component in self.components.iter_mut() {
            component.register_config_handler(self.config.clone())?;
        }

        for component in self.components.iter_mut() {
            component.init(tui.size()?)?;
        }

        while !self.should_quit {
            if let Some(e) = tui.next().await {
                match e {
                    Event::Quit => action_tx.send(Action::Quit)?,
                    Event::Tick => action_tx.send(Action::Tick)?,
                    Event::Render => action_tx.send(Action::Render)?,
                    Event::Key(key) => self.handle_key_sequence(&action_tx, key)?,
                    Event::Resize(x, y) => action_tx.send(Action::Resize(x, y))?,
                    _ => {}
                }
                for component in self.components.iter_mut() {
                    if let Some(action) = component.handle_events(Some(e.clone()))? {
                        action_tx.send(action)?;
                    }
                }
            }

            while let Ok(action) = action_rx.try_recv() {
                if action != Action::Tick && action != Action::Render {
                    debug!("{action:?}");
                }
                match action {
                    Action::Tick => self.recent_key_events.clear(),
                    Action::Render => self.render(&mut tui)?,
                    Action::Resize(w, h) => {
                        tui.resize(Rect::new(0, 0, w, h))?;
                        self.render(&mut tui)?;
                    }
                    Action::Suspend => self.should_suspend = true,
                    Action::Resume => self.should_suspend = false,
                    Action::Quit => self.should_quit = true,
                    _ => {}
                }
                for component in self.components.iter_mut() {
                    if let Some(action) = component.update(action.clone())? {
                        action_tx.send(action)?
                    };
                }
            }
            if self.should_suspend {
                tui.suspend()?;
                action_tx.send(Action::Resume)?;
                tui = self.tui()?;
                tui.resume()?;
            }
        }

        tui.exit()
    }

    fn tui(&self) -> Result<Tui> {
        let mut tui = Tui::new()?;
        tui.tick_rate(self.tick_rate);
        tui.frame_rate(self.frame_rate);
        tui.mouse(true);
        tui.paste(true);
        Ok(tui)
    }

    fn render(&mut self, tui: &mut Tui) -> Result<()> {
        let mut draw_result = Ok(());
        tui.draw(|f| {
            for component in self.components.iter_mut() {
                let maybe_err = component.draw(f, f.size());
                if maybe_err.is_err() {
                    draw_result = maybe_err;
                }
            }
        })?;
        draw_result
    }

    fn handle_key_sequence(
        &mut self,
        action_tx: &UnboundedSender<Action>,
        key: KeyEvent,
    ) -> Result<()> {
        let Some(keymap) = self.config.keybindings.get(&self.mode) else {
            return Ok(());
        };
        self.recent_key_events.push(key);
        if let Some(action) = keymap.get(&self.recent_key_events) {
            info!("In mode {mode:?}, got action: {action:?}", mode = self.mode);
            action_tx.send(action.clone())?;
            self.recent_key_events.clear();
        }
        Ok(())
    }
}
