use std::fmt::Display;

use arbitrary::Arbitrary;
use color_eyre::eyre::Result;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::action::Action;
use crate::components::Component;
use crate::mode::Mode;
use crate::triton_vm_state::TritonVMState;

#[derive(Debug, Default, Copy, Clone, Arbitrary)]
pub(crate) struct Help {
    pub previous_mode: Mode,
}

impl Component for Help {
    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::HideHelpScreen => Ok(Some(Action::Mode(self.previous_mode))),
            Action::Mode(mode) if mode != Mode::Help => {
                self.previous_mode = mode;
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn draw(&mut self, frame: &mut Frame<'_>, _: &TritonVMState) -> Result<()> {
        let title = Title::from("Triton TUI — Help");
        let text = [
            Help::mode_line("Home"),
            Help::help_line("c", "continue – execute to next breakpoint"),
            Help::help_line("s", "step     – execute one instruction"),
            Help::help_line("n", "next     – like “step” but steps over “call”"),
            Help::help_line("f", "finish   – step out of current “call”"),
            Help::help_line("u", "undo last command that advanced execution"),
            Help::help_line("r", "reload files and restart Triton VM"),
            String::new(),
            Help::help_line("t,a", "toggle all widgets"),
            Help::help_line("t,t", "toggle type annotations"),
            Help::help_line("t,c", "toggle call stack"),
            Help::help_line("t,i", "toggle displaying input (if any)"),
            String::new(),
            Help::mode_line("Memory"),
            Help::help_line("Enter", "focus text area"),
            Help::help_line("Shift+PgUp", "go to previous block"),
            Help::help_line("Shift+PgDn", "go to next block"),
            String::new(),
            Help::help_line("t,b", "toggle block address display"),
            String::new(),
            Help::mode_line("General"),
            Help::help_line("Esc", "show Home screen"),
            Help::help_line("m", "toggle Memory screen"),
            Help::help_line("h", "toggle Help"),
            Help::help_line("q", "quit"),
        ];

        let centered_rect = Self::centered_rect(frame.size(), &text);
        let block = Block::default().title(title).padding(Padding::top(1));
        let paragraph = Paragraph::new(text.map(Line::from).to_vec()).block(block);

        frame.render_widget(paragraph, centered_rect);
        Ok(())
    }
}

impl Help {
    fn mode_line(mode: impl Display) -> String {
        format!("{mode}:")
    }

    fn help_line(keys: impl Display, help: impl Display) -> String {
        format!("  {keys: <10}  {help}")
    }

    fn centered_rect<const N: usize>(area: Rect, text: &[String; N]) -> Rect {
        let max_line_length = text.iter().map(String::len).max().unwrap_or(0) as u16;
        let layout = Layout::horizontal([max_line_length]);
        let [horizontally_centered] = layout.flex(Flex::Center).areas(area);

        let padded_title_height = 2;
        let layout = Layout::vertical([N as u16 + padded_title_height]);
        let [centered] = layout.flex(Flex::Center).areas(horizontally_centered);

        centered
    }
}

#[cfg(test)]
mod tests {
    use proptest_arbitrary_interop::arb;
    use ratatui::backend::TestBackend;
    use test_strategy::proptest;

    use crate::args::TuiArgs;

    use super::*;

    #[proptest]
    fn render(#[strategy(arb())] mut help: Help) {
        let state = TritonVMState::new(&TuiArgs::default()).unwrap();

        let backend = TestBackend::new(150, 50);
        let mut terminal = Terminal::new(backend)?;
        terminal.draw(|f| help.draw(f, &state).unwrap()).unwrap();
    }
}
