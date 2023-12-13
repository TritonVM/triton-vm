use color_eyre::eyre::Result;
use ratatui::prelude::*;
use ratatui::widgets::block::Title;
use ratatui::widgets::*;
use ratatui::Frame;

use crate::components::Component;
use crate::triton_vm_state::TritonVMState;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Memory {
    /// The address touched last by any `read_mem` or `write_mem` instruction.
    pub most_recent_address: u64,

    /// The address to show. Can be manually set (and unset) by the user.
    pub user_address: Option<u64>,

    pub undo_stack: Vec<UndoInformation>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UndoInformation {
    pub most_recent_address: u64,
    pub user_address: Option<u64>,
}

impl Component for Memory {
    fn draw(&mut self, frame: &mut Frame<'_>, state: &TritonVMState) -> Result<()> {
        let title = Title::from(" Random Access Memory ").alignment(Alignment::Left);
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(title);
        let draw_area = frame.size();

        let num_lines = block.inner(draw_area).height;
        let requested_address = self.user_address.unwrap_or(self.most_recent_address);
        let address_range_start = requested_address.saturating_sub(num_lines as u64 / 2);
        let address_range_end = address_range_start + num_lines as u64;

        let mut text = vec![];
        for address in address_range_start..address_range_end {
            let address_style = match address == requested_address {
                true => Style::new().bold(),
                false => Style::new().dim(),
            };

            let address = address.into();
            let maybe_value = state.vm_state.ram.get(&address);
            let value = maybe_value.copied().unwrap_or(0_u64.into());

            // additional `.to_string()` to circumvent padding bug (?) in `format`
            let address = Span::from(format!("{address: >20}", address = address.to_string()));
            let address = address.set_style(address_style);
            let separator = Span::from("  ");
            let value = Span::from(value.to_string());
            text.push(Line::from([address, separator, value].to_vec()));
        }

        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, draw_area);
        Ok(())
    }
}
