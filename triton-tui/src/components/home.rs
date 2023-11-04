use color_eyre::eyre::Result;
use crossterm::event::*;
use ratatui::prelude::*;
use ratatui::widgets::*;
use tokio::sync::mpsc::UnboundedSender;

use crate::action::Action;
use crate::config::Config;

use super::Component;
use super::Frame;

#[derive(Default)]
pub(crate) struct Home {
    command_tx: Option<UnboundedSender<Action>>,
    config: Config,
    counter: i64,
}

impl Home {
    pub fn new() -> Self {
        Self::default()
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

    fn handle_mouse_events(&mut self, event: MouseEvent) -> Result<Option<Action>> {
        let MouseEvent { kind, .. } = event;
        match kind {
            MouseEventKind::Down(MouseButton::Left) => Ok(Some(Action::IncrementCounter)),
            MouseEventKind::Down(MouseButton::Right) => Ok(Some(Action::DecrementCounter)),
            MouseEventKind::ScrollUp => Ok(Some(Action::IncrementCounter)),
            MouseEventKind::ScrollDown => Ok(Some(Action::DecrementCounter)),
            _ => Ok(None),
        }
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::IncrementCounter => self.counter += 1,
            Action::DecrementCounter => self.counter -= 1,
            _ => {}
        }
        Ok(None)
    }

    fn draw(&mut self, f: &mut Frame<'_>, area: Rect) -> Result<()> {
        let counter = self.counter;
        let message = format!("Press e to increment, n to decrement.\n\n\t{counter}");
        f.render_widget(Paragraph::new(message), area);
        Ok(())
    }
}
