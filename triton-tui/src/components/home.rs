use color_eyre::eyre::Result;
use crossterm::event::*;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::*;
use tokio::sync::mpsc::UnboundedSender;
use tracing::info;
use triton_vm::triton_program;

use crate::action::Action;
use crate::components::centered_rect;
use crate::config::Config;

use super::Component;
use super::Frame;

#[derive(Debug)]
pub(crate) struct Home {
    command_tx: Option<UnboundedSender<Action>>,
    config: Config,
    _program: triton_vm::Program,
    vm_state: triton_vm::vm::VMState,
}

impl Home {
    pub fn new() -> Self {
        let program = triton_program! { push 5 push 7 break add halt };

        let public_input = [].into();
        let non_determinism = [].into();
        let vm_state = triton_vm::vm::VMState::new(&program, public_input, non_determinism);

        Self {
            command_tx: None,
            config: Config::default(),
            _program: program,
            vm_state,
        }
    }

    fn run_program(&mut self) -> Result<()> {
        if !self.vm_state.halting {
            self.vm_state.step()?;
        }
        Ok(())
    }
}

impl Component for Home {
    fn register_action_handler(&mut self, tx: UnboundedSender<Action>) -> Result<()> {
        self.command_tx = Some(tx);
        Ok(())
    }

    fn register_config_handler(&mut self, config: Config) -> Result<()> {
        self.config = config;
        Ok(())
    }

    fn handle_mouse_event(&mut self, event: MouseEvent) -> Result<Option<Action>> {
        let MouseEvent { kind, .. } = event;
        match kind {
            MouseEventKind::Down(MouseButton::Left) => Ok(Some(Action::RunProgram)),
            _ => Ok(None),
        }
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::RunProgram => self.run_program()?,
            Action::Tick => info!("tick"),
            _ => {}
        }
        Ok(None)
    }

    fn draw(&mut self, f: &mut Frame<'_>, area: Rect) -> Result<()> {
        let title = Title::from(" Triton TUI ").alignment(Alignment::Left);
        let state = self.vm_state.to_string();

        let block = Block::default().title(title).padding(Padding::uniform(1));

        let paragraph = Paragraph::new(state)
            .block(block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true });

        let area = centered_rect(area, 90, 90);
        f.render_widget(paragraph, area);
        Ok(())
    }
}
