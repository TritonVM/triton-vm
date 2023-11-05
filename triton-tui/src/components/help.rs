use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Wrap;
use ratatui::Frame;
use std::fmt::Display;

use crate::components::centered_rect;
use crate::components::Component;

#[derive(Default)]
pub(crate) struct Help;

impl Component for Help {
    fn draw(&mut self, f: &mut Frame<'_>, area: Rect) -> color_eyre::Result<()> {
        let title = Title::from(" Triton TUI — Help").alignment(Alignment::Left);
        let text = [
            Help::help_line("l", "load program"),
            Help::help_line("r", "run program"),
            Help::help_line("b", "run program to breakpoint"),
            Help::help_line("i", "step into call"),
            Help::help_line("o", "step over call"),
            String::new(),
            Help::help_line("?", "show help"),
            Help::help_line("q", "quit"),
        ]
        .map(Line::from)
        .to_vec();

        let block = Block::default().title(title).padding(Padding::uniform(1));
        let paragraph = Paragraph::new(text)
            .block(block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true });
        let area = centered_rect(area, 50, 50);

        f.render_widget(paragraph, area);
        Ok(())
    }
}

impl Help {
    fn help_line(keys: impl Display, help: impl Display) -> String {
        format!("{keys: <10} {help}")
    }
}
