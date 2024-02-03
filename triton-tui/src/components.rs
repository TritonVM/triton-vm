use std::fmt::Debug;

use color_eyre::eyre::Result;
use crossterm::event::KeyEvent;
use crossterm::event::MouseEvent;
use ratatui::prelude::*;
use tokio::sync::mpsc::UnboundedSender;

use crate::action::Action;
use crate::triton_vm_state::TritonVMState;
use crate::tui::Event;

pub(crate) mod help;
pub(crate) mod home;
pub(crate) mod memory;

/// `Component` is a trait that represents a visual and interactive element of the user interface.
/// Implementors of this trait can be registered with the main application loop and will be able to
/// receive events, update state, and be rendered on the screen.
pub(crate) trait Component: Debug {
    fn register_action_handler(&mut self, _tx: UnboundedSender<Action>) -> Result<()> {
        Ok(())
    }

    fn request_exclusive_key_event_handling(&self) -> bool {
        false
    }

    fn handle_event(&mut self, event: Option<Event>) -> Result<Option<Action>> {
        let r = match event {
            Some(Event::Key(key_event)) => self.handle_key_event(key_event)?,
            Some(Event::Mouse(mouse_event)) => self.handle_mouse_event(mouse_event)?,
            _ => None,
        };
        Ok(r)
    }

    fn handle_key_event(&mut self, _: KeyEvent) -> Result<Option<Action>> {
        Ok(None)
    }

    fn handle_mouse_event(&mut self, _: MouseEvent) -> Result<Option<Action>> {
        Ok(None)
    }

    /// Update the state of the component based on a received action.
    fn update(&mut self, _: Action) -> Result<Option<Action>> {
        Ok(None)
    }

    /// Render the component on the screen.
    fn draw(&mut self, _frame: &mut Frame<'_>, _state: &TritonVMState) -> Result<()> {
        Ok(())
    }
}
