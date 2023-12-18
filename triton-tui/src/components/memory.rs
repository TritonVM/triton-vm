use color_eyre::eyre::Result;
use crossterm::event::KeyEventKind::Release;
use crossterm::event::*;
use num_traits::One;
use ratatui::prelude::*;
use ratatui::widgets::*;
use ratatui::Frame;
use tui_textarea::TextArea;

use triton_vm::instruction::Instruction;
use triton_vm::BFieldElement;

use crate::action::Action;
use crate::action::ExecutedInstruction;
use crate::components::Component;
use crate::element_type_hint::ElementTypeHint;
use crate::mode::Mode;
use crate::triton_vm_state::TritonVMState;

#[derive(Debug, Clone)]
pub(crate) struct Memory<'a> {
    /// The address touched last by any `read_mem` or `write_mem` instruction.
    pub most_recent_address: u64,

    /// The address to show. Can be manually set (and unset) by the user.
    pub user_address: Option<u64>,

    pub text_area: TextArea<'a>,
    pub text_area_in_focus: bool,

    pub undo_stack: Vec<UndoInformation>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UndoInformation {
    pub most_recent_address: u64,
}

#[derive(Debug, Clone, Copy)]
struct RenderInfo<'s> {
    state: &'s TritonVMState,
    areas: WidgetAreas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WidgetAreas {
    memory: Rect,
    text_input: Rect,
}

impl<'a> Default for Memory<'a> {
    fn default() -> Self {
        Self {
            most_recent_address: 0,
            user_address: None,
            text_area: Self::initial_text_area(),
            undo_stack: vec![],
            text_area_in_focus: false,
        }
    }
}

impl<'a> Memory<'a> {
    fn initial_text_area() -> TextArea<'a> {
        let mut text_area = TextArea::default();
        text_area.set_cursor_line_style(Style::default());
        text_area
    }

    pub fn undo(&mut self) {
        let Some(undo_information) = self.undo_stack.pop() else {
            return;
        };

        self.most_recent_address = undo_information.most_recent_address;
    }

    pub fn record_undo_information(&mut self) {
        let undo_information = UndoInformation {
            most_recent_address: self.most_recent_address,
        };
        self.undo_stack.push(undo_information);
    }

    pub fn reset(&mut self) {
        self.most_recent_address = 0;
        self.user_address = None;
        self.undo_stack.clear();
    }

    pub fn handle_instruction(&mut self, executed_instruction: ExecutedInstruction) {
        let presumed_ram_pointer = executed_instruction.new_top_of_stack[0];
        let overshoot_adjustment = match executed_instruction.instruction {
            Instruction::ReadMem(_) => BFieldElement::one(),
            Instruction::WriteMem(_) => -BFieldElement::one(),
            _ => return,
        };
        let last_ram_pointer = presumed_ram_pointer + overshoot_adjustment;
        self.most_recent_address = last_ram_pointer.value();
    }

    fn submit_address(&mut self) {
        let user_input = self.text_area.lines()[0].trim();
        let maybe_address = user_input.parse::<u64>().ok();
        self.user_address = maybe_address;
    }

    fn requested_address(&self) -> u64 {
        self.user_address.unwrap_or(self.most_recent_address)
    }

    fn distribute_area_for_widgets(&self, area: Rect) -> WidgetAreas {
        let text_area_height = Constraint::Min(2);
        let constraints = [Constraint::Percentage(100), text_area_height];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        WidgetAreas {
            memory: layout[0],
            text_input: layout[1],
        }
    }

    fn render_memory_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let block = Self::memory_widget_block();
        let draw_area = render_info.areas.memory;

        let num_lines = block.inner(draw_area).height;
        let requested_address = self.requested_address();
        let address_range_start = requested_address.saturating_sub(num_lines as u64 / 2);
        let address_range_end = address_range_start.saturating_add(num_lines as u64);

        let mut text = vec![];
        for address in address_range_start..address_range_end {
            let address = address.into();

            let memory_cell = self.render_memory_cell_at_address(render_info, address);
            let separator = vec![Span::from("  ")];
            let type_hint = Self::render_type_hint_at_address(render_info, address);

            text.push(Line::from([memory_cell, separator, type_hint].concat()));
        }

        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, draw_area);
    }

    fn render_memory_cell_at_address(
        &self,
        render_info: RenderInfo,
        address: BFieldElement,
    ) -> Vec<Span> {
        let address_style = match address.value() == self.requested_address() {
            true => Style::new().bold(),
            false => Style::new().dim(),
        };

        let maybe_value = render_info.state.vm_state.ram.get(&address);
        let value = maybe_value.copied().unwrap_or(0_u64.into());

        // additional `.to_string()` to circumvent padding bug (?) in `format`
        let address = Span::from(format!("{address: >20}", address = address.to_string()));
        let address = address.set_style(address_style);
        let separator = Span::from("  ");
        let value = Span::from(format!("{value: <20}", value = value.to_string()));

        vec![address, separator, value]
    }

    fn render_type_hint_at_address(render_info: RenderInfo, address: BFieldElement) -> Vec<Span> {
        let maybe_type_hint = render_info.state.type_hints.ram.get(&address);
        let type_hint = maybe_type_hint.unwrap_or(&None);
        ElementTypeHint::render(type_hint)
    }

    fn render_text_input_widget(&mut self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let placeholder_text = match self.text_area_in_focus {
            true => "",
            false => "Go to address. Empty for most recent read / write.",
        };
        self.text_area.set_placeholder_text(placeholder_text);

        let cursor_style = match self.text_area_in_focus {
            true => Style::default().add_modifier(Modifier::REVERSED),
            false => Style::default(),
        };
        self.text_area.set_cursor_style(cursor_style);

        let text_style = match self.text_area_in_focus {
            true => Style::default(),
            false => Style::default().dim(),
        };
        self.text_area.set_style(text_style);

        let block = Self::text_input_block();
        self.text_area.set_block(block);
        frame.render_widget(self.text_area.widget(), render_info.areas.text_input);
    }

    fn memory_widget_block() -> Block<'a> {
        let border_set = symbols::border::Set {
            bottom_left: symbols::line::ROUNDED.vertical_right,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .borders(Borders::ALL)
            .border_set(border_set)
            .title(" Random Access Memory ")
    }

    fn text_input_block() -> Block<'a> {
        Block::default()
            .padding(Padding::horizontal(1))
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_type(BorderType::Rounded)
    }
}

impl<'a> Component for Memory<'a> {
    fn request_exclusive_key_event_handling(&self) -> bool {
        self.text_area_in_focus
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) -> Result<Option<Action>> {
        if key_event.kind == Release {
            return Ok(None);
        }
        if key_event.code == KeyCode::Esc {
            self.text_area_in_focus = false;
            return Ok(None);
        }
        if key_event.code == KeyCode::Enter {
            if self.text_area_in_focus {
                self.submit_address();
            }
            self.text_area_in_focus = !self.text_area_in_focus;
            return Ok(None);
        }
        if self.text_area_in_focus {
            self.text_area.input(key_event);
        }
        Ok(None)
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::Mode(Mode::Memory) => (),
            Action::Mode(_) => self.text_area_in_focus = false,
            Action::Undo => self.undo(),
            Action::RecordUndoInfo => self.record_undo_information(),
            Action::Reset => self.reset(),
            Action::ExecutedInstruction(instruction) => self.handle_instruction(*instruction),
            _ => (),
        }
        Ok(None)
    }

    fn draw(&mut self, frame: &mut Frame<'_>, state: &TritonVMState) -> Result<()> {
        let widget_areas = self.distribute_area_for_widgets(frame.size());
        let render_info = RenderInfo {
            state,
            areas: widget_areas,
        };

        self.render_memory_widget(frame, render_info);
        self.render_text_input_widget(frame, render_info);
        Ok(())
    }
}
