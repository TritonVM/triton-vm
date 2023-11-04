use color_eyre::eyre::Result;
use crossterm::event::KeyEvent;
use crossterm::event::MouseEvent;
use ratatui::prelude::*;
use tokio::sync::mpsc::UnboundedSender;

use crate::action::Action;
use crate::config::Config;
use crate::tui::Event;

pub(crate) mod fps;
pub(crate) mod home;

/// `Component` is a trait that represents a visual and interactive element of the user interface.
/// Implementors of this trait can be registered with the main application loop and will be able to
/// receive events, update state, and be rendered on the screen.
pub(crate) trait Component {
    /// Register an action handler that can send actions for processing if necessary.
    ///
    /// # Arguments
    ///
    /// * `tx` - An unbounded sender that can send actions.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An Ok result or an error.
    fn register_action_handler(&mut self, _tx: UnboundedSender<Action>) -> Result<()> {
        Ok(())
    }

    /// Register a configuration handler that provides configuration settings if necessary.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration settings.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An Ok result or an error.
    fn register_config_handler(&mut self, _config: Config) -> Result<()> {
        Ok(())
    }

    /// Initialize the component with a specified area if necessary.
    ///
    /// # Arguments
    ///
    /// * `area` - Rectangular area to initialize the component within.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An Ok result or an error.
    fn init(&mut self, _area: Rect) -> Result<()> {
        Ok(())
    }

    /// Handle incoming events and produce actions if necessary.
    ///
    /// # Arguments
    ///
    /// * `event` - An optional event to be processed.
    ///
    /// # Returns
    ///
    /// * `Result<Option<Action>>` - An action to be processed or none.
    fn handle_events(&mut self, event: Option<Event>) -> Result<Option<Action>> {
        let r = match event {
            Some(Event::Key(key_event)) => self.handle_key_events(key_event)?,
            Some(Event::Mouse(mouse_event)) => self.handle_mouse_events(mouse_event)?,
            _ => None,
        };
        Ok(r)
    }

    /// Handle key events and produce actions if necessary.
    ///
    /// # Arguments
    ///
    /// * `key` - A key event to be processed.
    ///
    /// # Returns
    ///
    /// * `Result<Option<Action>>` - An action to be processed or none.
    fn handle_key_events(&mut self, _key: KeyEvent) -> Result<Option<Action>> {
        Ok(None)
    }

    /// Handle mouse events and produce actions if necessary.
    ///
    /// # Arguments
    ///
    /// * `mouse` - A mouse event to be processed.
    ///
    /// # Returns
    ///
    /// * `Result<Option<Action>>` - An action to be processed or none.
    fn handle_mouse_events(&mut self, _mouse: MouseEvent) -> Result<Option<Action>> {
        Ok(None)
    }

    /// Update the state of the component based on a received action.
    ///
    /// # Arguments
    ///
    /// * `action` - An action that may modify the state of the component.
    ///
    /// # Returns
    ///
    /// * `Result<Option<Action>>` - An action to be processed or none.
    fn update(&mut self, _action: Action) -> Result<Option<Action>> {
        Ok(None)
    }

    /// Render the component on the screen.
    ///
    /// # Arguments
    ///
    /// * `f` - A frame used for rendering.
    /// * `area` - The area in which the component should be drawn.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An Ok result or an error.
    fn draw(&mut self, f: &mut Frame<'_>, area: Rect) -> Result<()>;
}

/// helper function to create a centered rect using up certain percentage of the available rect `r`
fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let area = centered_rect_in_direction(area, percent_y, Direction::Vertical);
    centered_rect_in_direction(area, percent_x, Direction::Horizontal)
}

fn centered_rect_in_direction(area: Rect, percentage: u16, direction: Direction) -> Rect {
    Layout::default()
        .direction(direction)
        .constraints([
            Constraint::Percentage((100 - percentage) / 2),
            Constraint::Percentage(percentage),
            Constraint::Percentage((100 - percentage) / 2),
        ])
        .split(area)[1]
}
