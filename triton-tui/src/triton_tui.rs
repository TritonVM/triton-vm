use color_eyre::eyre::Result;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyEventKind;
use ratatui::prelude::Rect;
use strum::EnumCount;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;

use crate::action::*;
use crate::args::Args;
use crate::components::help::Help;
use crate::components::home::Home;
use crate::components::memory::Memory;
use crate::components::Component;
use crate::config::Config;
use crate::config::KeyEvents;
use crate::mode::Mode;
use crate::triton_vm_state::TritonVMState;
use crate::tui::*;

const RECENT_KEY_EVENTS_RESET_DELAY: u32 = 1;

pub(crate) struct TritonTUI {
    pub args: Args,
    pub config: Config,

    pub tui: Tui,
    pub mode: Mode,
    pub components: [Box<dyn Component>; Mode::COUNT],

    pub should_quit: bool,
    pub should_suspend: bool,

    pub recent_key_events_reset_delay: u32,
    pub recent_key_events: KeyEvents,

    pub vm_state: TritonVMState,
}

impl TritonTUI {
    pub fn new(args: Args) -> Result<Self> {
        let tui = Self::tui(&args)?;
        let config = Config::new()?;

        let mode = Mode::default();
        let components: [Box<dyn Component>; Mode::COUNT] = [
            Box::<Home>::default(),
            Box::<Memory>::default(),
            Box::<Help>::default(),
        ];

        let vm_state = TritonVMState::new(&args)?;

        Ok(Self {
            args,
            config,
            tui,
            mode,
            components,
            should_quit: false,
            should_suspend: false,
            recent_key_events_reset_delay: 0,
            recent_key_events: vec![],
            vm_state,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        let (action_tx, mut action_rx) = mpsc::unbounded_channel();
        self.tui.enter()?;

        self.vm_state.register_action_handler(action_tx.clone())?;
        for component in self.components.iter_mut() {
            component.register_action_handler(action_tx.clone())?;
        }

        // in case of infinite loop (or similar) before first breakpoint, provide visual feedback
        self.render()?;
        action_tx.send(Action::Execute(Execute::Continue))?;

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
                    Action::Tick => self.maybe_clear_recent_key_events(),
                    Action::Render => self.render()?,
                    Action::Resize(w, h) => {
                        self.tui.resize(Rect::new(0, 0, w, h))?;
                        self.render()?;
                    }
                    Action::Mode(mode) => {
                        self.mode = mode;
                        self.render()?;
                    }
                    Action::Reset => self.reset_state(action_tx.clone())?,
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

    fn maybe_clear_recent_key_events(&mut self) {
        if self.recent_key_events_reset_delay > 0 {
            self.recent_key_events_reset_delay -= 1;
        } else {
            self.recent_key_events.clear();
        }
    }

    fn reset_state(&mut self, action_tx: UnboundedSender<Action>) -> Result<()> {
        let vm_state = match TritonVMState::new(&self.args) {
            Ok(vm_state) => vm_state,
            Err(report) => {
                self.vm_state.warning = Some(report);
                return Ok(());
            }
        };
        self.vm_state = vm_state;
        self.vm_state.register_action_handler(action_tx.clone())?;
        self.render()?;
        action_tx.send(Action::Execute(Execute::Continue))?;
        Ok(())
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
        if self.components[self.mode.id()].request_exclusive_key_event_handling() {
            return Ok(());
        }
        let Some(keymap) = self.config.keybindings.get(&self.mode) else {
            return Ok(());
        };
        self.recent_key_events.push(key);
        self.recent_key_events_reset_delay = RECENT_KEY_EVENTS_RESET_DELAY;
        if let Some(action) = keymap.get(&self.recent_key_events) {
            action_tx.send(action.clone())?;
            self.recent_key_events.clear();
        }
        if key.code == KeyCode::Esc && key.kind != KeyEventKind::Release {
            self.recent_key_events_reset_delay = 0;
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
