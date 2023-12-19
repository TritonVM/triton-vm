use std::fmt::Display;

use arbitrary::Arbitrary;
use color_eyre::eyre::Result;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::action::Action;
use crate::components::centered_rect;
use crate::components::Component;
use crate::mode::Mode;
use crate::triton_vm_state::TritonVMState;

#[derive(Default, Debug, Clone, Copy, Arbitrary)]
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
        let title = Title::from(" Triton TUI — Help").alignment(Alignment::Left);
        let text = [
            Help::mode_line("Home"),
            Help::help_line("c", "continue – execute to next breakpoint"),
            Help::help_line("s", "step     – execute one instruction"),
            Help::help_line("n", "next     – like “step” but steps over “call”"),
            Help::help_line("f", "finish   – step out of current “call”"),
            Help::help_line("u", "undo last command that advanced execution"),
            Help::help_line("r", "reload files and restart Triton VM"),
            String::new(),
            Help::help_line("t", "toggle type annotations"),
            Help::help_line("i", "toggle displaying input (if any)"),
            String::new(),
            "General:".to_string(),
            Help::help_line("esc", "show Home screen"),
            Help::help_line("m", "toggle Memory screen"),
            Help::help_line("h", "toggle Help"),
            String::new(),
            Help::help_line("q", "quit"),
        ]
        .map(Line::from)
        .to_vec();

        let block = Block::default().title(title).padding(Padding::uniform(1));
        let paragraph = Paragraph::new(text).block(block);

        let area = centered_rect(frame.size(), 50, 80);
        frame.render_widget(paragraph, area);
        Ok(())
    }
}

impl Help {
    fn mode_line(mode: impl Display) -> String {
        format!("{mode}:")
    }

    fn help_line(keys: impl Display, help: impl Display) -> String {
        format!("  {keys: <4} {help}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest_arbitrary_interop::arb;
    use ratatui::backend::TestBackend;
    use test_strategy::proptest;

    #[proptest]
    fn render(#[strategy(arb())] mut help: Help) {
        let state = TritonVMState::new(&Default::default()).unwrap();

        let backend = TestBackend::new(150, 50);
        let mut terminal = Terminal::new(backend)?;
        terminal.draw(|f| help.draw(f, &state).unwrap()).unwrap();
    }
}
