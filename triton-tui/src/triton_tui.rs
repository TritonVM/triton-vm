use color_eyre::eyre::Result;
use crossterm::event::KeyEvent;
use ratatui::prelude::Rect;
use strum::EnumCount;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use tracing::info;

use crate::action::Action;
use crate::args::Args;
use crate::components::help::Help;
use crate::components::home::Home;
use crate::components::Component;
use crate::config::Config;
use crate::config::KeyEvents;
use crate::mode::Mode;
use crate::triton_vm_state::TritonVMState;
use crate::tui::*;
use crate::utils::trace_dbg;

pub(crate) struct TritonTUI {
    pub args: Args,
    pub config: Config,

    pub tui: Tui,
    pub mode: Mode,
    pub components: [Box<dyn Component>; Mode::COUNT],

    pub should_quit: bool,
    pub should_suspend: bool,
    pub recent_key_events: KeyEvents,

    pub vm_state: TritonVMState,
}

impl TritonTUI {
    pub fn new(args: Args) -> Result<Self> {
        let config = Config::new()?;
        let tui = Self::tui(&args)?;
        let mode = Mode::default();

        let components: [Box<dyn Component>; Mode::COUNT] =
            [Box::<Home>::default(), Box::<Help>::default()];

        let vm_state = TritonVMState::new(&args)?;

        Ok(Self {
            args,
            config,
            tui,
            mode,
            components,
            should_quit: false,
            should_suspend: false,
            recent_key_events: vec![],
            vm_state,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        let (action_tx, mut action_rx) = mpsc::unbounded_channel();
        self.tui.enter()?;

        trace_dbg!("Tui entered");

        for component in self.components.iter_mut() {
            component.register_action_handler(action_tx.clone())?;
            component.register_config_handler(self.config.clone())?;
            component.init(self.tui.size()?)?;
        }

        while !self.should_quit {
            if let Some(e) = self.tui.next().await {
                match e {
                    Event::Quit => action_tx.send(Action::Quit)?,
                    Event::Tick => action_tx.send(Action::Tick)?,
                    Event::Render => action_tx.send(Action::Render)?,
                    Event::Key(key) => self.handle_key_sequence(&action_tx, key)?,
                    Event::Resize(x, y) => action_tx.send(Action::Resize(x, y))?,
                    _ => {}
                }
                for component in self.components.iter_mut() {
                    if let Some(action) = component.handle_event(Some(e.clone()))? {
                        action_tx.send(action)?;
                    }
                }
            }

            while let Ok(action) = action_rx.try_recv() {
                match action {
                    Action::Tick => self.recent_key_events.clear(),
                    Action::Render => self.render()?,
                    Action::Resize(w, h) => {
                        self.tui.resize(Rect::new(0, 0, w, h))?;
                        self.render()?;
                    }
                    Action::Mode(mode) => {
                        self.mode = mode;
                        self.render()?;
                    }
                    Action::ProgramReset => {
                        self.vm_state = TritonVMState::new(&self.args)?;
                        self.render()?;
                    }
                    Action::Suspend => self.should_suspend = true,
                    Action::Resume => self.should_suspend = false,
                    Action::Quit => self.should_quit = true,
                    _ => {}
                }
                self.vm_state.update(action.clone())?;
                for component in self.components.iter_mut() {
                    if let Some(action) = component.update(action.clone())? {
                        action_tx.send(action)?
                    };
                }
            }
            if self.should_suspend {
                self.tui.suspend()?;
                action_tx.send(Action::Resume)?;
                self.tui = Self::tui(&self.args)?;
                self.tui.resume()?;
            }
        }

        self.tui.exit()
    }

    fn tui(args: &Args) -> Result<Tui> {
        let mut tui = Tui::new()?;
        tui.apply_args(args);
        Ok(tui)
    }

    fn render(&mut self) -> Result<()> {
        let mode_id = self.mode.id();
        let vm_state = &self.vm_state;
        let mut draw_result = Ok(());
        self.tui.draw(|frame| {
            let maybe_err = self.components[mode_id].draw(frame, vm_state);
            if maybe_err.is_err() {
                draw_result = maybe_err;
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

    pub fn terminate(&mut self) -> Result<()> {
        self.tui.exit()?;
        self.should_quit = true;
        Ok(())
    }
}
