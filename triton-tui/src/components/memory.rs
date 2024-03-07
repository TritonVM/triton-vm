use arbitrary::Arbitrary;
use color_eyre::eyre::Result;
use crossterm::event::KeyEventKind::Release;
use crossterm::event::*;
use ratatui::prelude::*;
use ratatui::widgets::*;
use ratatui::Frame;
use triton_vm::instruction::Instruction;
use triton_vm::prelude::*;
use tui_textarea::TextArea;

use crate::action::Action;
use crate::action::ExecutedInstruction;
use crate::action::Toggle;
use crate::components::Component;
use crate::element_type_hint::ElementTypeHint;
use crate::triton_vm_state::TritonVMState;
use crate::tui::Event;

#[derive(Debug, Clone)]
pub(crate) struct Memory<'a> {
    /// The address touched last by any `read_mem` or `write_mem` instruction.
    pub most_recent_address: BFieldElement,

    /// The address to show. Can be manually set (and unset) by the user.
    pub user_address: Option<BFieldElement>,

    pub text_area: TextArea<'a>,
    pub text_area_in_focus: bool,
    pub show_block_addresses: bool,

    pub undo_stack: Vec<UndoInformation>,
}

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Arbitrary)]
pub(crate) struct UndoInformation {
    pub most_recent_address: BFieldElement,
}

#[derive(Debug, Copy, Clone)]
struct RenderInfo<'s> {
    state: &'s TritonVMState,
    areas: WidgetAreas,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct WidgetAreas {
    memory: Rect,
    text_input: Rect,
}

impl<'a> Default for Memory<'a> {
    fn default() -> Self {
        Self {
            most_recent_address: 0_u64.into(),
            user_address: None,
            text_area: Self::initial_text_area(),
            text_area_in_focus: false,
            show_block_addresses: false,
            undo_stack: vec![],
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
        self.most_recent_address = 0_u64.into();
        self.user_address = None;
        self.undo_stack.clear();
    }

    pub fn handle_instruction(&mut self, executed_instruction: ExecutedInstruction) {
        let presumed_ram_pointer = executed_instruction.new_top_of_stack[0];
        let overshoot_adjustment = match executed_instruction.instruction {
            Instruction::ReadMem(_) => bfe!(1),
            Instruction::WriteMem(_) => bfe!(-1),
            _ => return,
        };
        let last_ram_pointer = presumed_ram_pointer + overshoot_adjustment;
        self.most_recent_address = last_ram_pointer;
    }

    pub fn toggle_address_display(&mut self) {
        self.show_block_addresses = !self.show_block_addresses;
    }

    fn submit_address(&mut self) {
        let user_input = self.text_area.lines()[0].trim();
        let Ok(address) = user_input.parse::<i128>() else {
            self.user_address = None;
            return;
        };

        let modulus = i128::from(BFieldElement::P);
        if address < -modulus || modulus <= address {
            self.user_address = None;
            return;
        }
        let address = ((address + modulus) % modulus) as u64;
        self.user_address = Some(bfe!(address));
    }

    fn requested_address(&self) -> BFieldElement {
        self.user_address.unwrap_or(self.most_recent_address)
    }

    fn paste(&mut self, s: &str) {
        self.text_area_in_focus = true;
        let s = s.replace(['\r', '\n'], "");
        self.text_area.insert_str(s);
    }

    fn scroll_content(&mut self, key: KeyEvent) {
        let page_size = bfe!(20);
        let new_address = match (key.modifiers, key.code) {
            (KeyModifiers::NONE, KeyCode::Up) => self.requested_address() - bfe!(1),
            (KeyModifiers::NONE, KeyCode::Down) => self.requested_address() + bfe!(1),
            (KeyModifiers::NONE, KeyCode::Home) => bfe!(0),
            (KeyModifiers::NONE, KeyCode::PageUp) => self.requested_address() - page_size,
            (KeyModifiers::NONE, KeyCode::PageDown) => self.requested_address() + page_size,
            (KeyModifiers::SHIFT, KeyCode::PageUp) => self.previous_memory_block(),
            (KeyModifiers::SHIFT, KeyCode::PageDown) => self.next_memory_block(),
            _ => return,
        };
        self.user_address = Some(new_address);
    }

    fn previous_memory_block(&self) -> BFieldElement {
        let current_block = Self::block_of_address(self.requested_address());
        let (previous_block, _) = current_block.overflowing_sub(1);
        Self::first_address_in_block(previous_block)
    }

    fn next_memory_block(&self) -> BFieldElement {
        let current_block = Self::block_of_address(self.requested_address());
        let (next_block, _) = current_block.overflowing_add(1);
        Self::first_address_in_block(next_block)
    }

    fn block_of_address(address: BFieldElement) -> u32 {
        (address.value() >> 32) as u32
    }

    fn address_in_block(address: BFieldElement) -> u32 {
        (address.value() & 0xFFFF_FFFF) as u32
    }

    fn first_address_in_block(block: u32) -> BFieldElement {
        (u64::from(block) << 32).into()
    }

    fn distribute_area_for_widgets(&self, area: Rect) -> WidgetAreas {
        let [memory, text_input] = Layout::vertical([Constraint::Fill(1), 2.into()]).areas(area);
        WidgetAreas { memory, text_input }
    }

    fn render_memory_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let block = Self::memory_widget_block();
        let draw_area = render_info.areas.memory;

        let num_lines = block.inner(draw_area).height;
        let address_range_start = self.requested_address() - bfe!(num_lines / 2);
        let address_range_end = address_range_start + bfe!(num_lines);

        let mut text = vec![];
        let mut address = address_range_start;
        while address != address_range_end {
            let memory_cell = self.render_memory_cell_at_address(render_info, address);
            let separator = vec![Span::from("  ")];
            let type_hint = Self::render_type_hint_at_address(render_info, address);

            text.push(Line::from([memory_cell, separator, type_hint].concat()));
            address.increment();
        }

        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, draw_area);
    }

    fn render_memory_cell_at_address(
        &self,
        render_info: RenderInfo,
        address: BFieldElement,
    ) -> Vec<Span> {
        let address_style = match address == self.requested_address() {
            true => Style::new().bold(),
            false => Style::new().dim(),
        };

        let maybe_value = render_info.state.vm_state.ram.get(&address);
        let value = maybe_value.copied().unwrap_or(0_u64.into());

        let address = Span::from(format!("{addr: >21}", addr = self.render_address(address)));
        let address = address.set_style(address_style);
        let separator = Span::from("  ");
        let value = Span::from(format!("{value: <20}", value = value.to_string()));

        vec![address, separator, value]
    }

    fn render_address(&self, address: BFieldElement) -> String {
        match self.show_block_addresses {
            true => Self::render_block_address(address),
            false => address.to_string(),
        }
    }

    fn render_block_address(address: BFieldElement) -> String {
        let block = Self::block_of_address(address);
        let address = Self::address_in_block(address);
        format!("{block: >10}╎{address: >10}")
    }

    fn render_type_hint_at_address(render_info: RenderInfo, address: BFieldElement) -> Vec<Span> {
        let prev_address = address - bfe!(1);
        let next_address = address + bfe!(1);

        let shadow_ram = &render_info.state.type_hints.ram;
        let prev_hint = shadow_ram.get(&prev_address).unwrap_or(&None);
        let curr_hint = shadow_ram.get(&address).unwrap_or(&None);
        let next_hint = shadow_ram.get(&next_address).unwrap_or(&None);

        if ElementTypeHint::is_continuous_sequence(&[prev_hint, curr_hint, next_hint]) {
            vec!["⋅".dim()]
        } else {
            ElementTypeHint::render(curr_hint)
        }
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
        Block::bordered()
            .padding(Padding::new(1, 1, 1, 0))
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

    fn handle_event(&mut self, event: Option<Event>) -> Result<Option<Action>> {
        let Some(event) = event else {
            return Ok(None);
        };

        if let Event::Paste(ref s) = event {
            self.paste(s);
        }

        let response = match event {
            Event::Key(key_event) => self.handle_key_event(key_event)?,
            Event::Mouse(mouse_event) => self.handle_mouse_event(mouse_event)?,
            _ => None,
        };
        Ok(response)
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
        match self.text_area_in_focus {
            true => _ = self.text_area.input(key_event),
            false => self.scroll_content(key_event),
        }
        Ok(None)
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::Mode(_) => self.text_area_in_focus = false,
            Action::Undo => self.undo(),
            Action::RecordUndoInfo => self.record_undo_information(),
            Action::Reset => self.reset(),
            Action::ExecutedInstruction(instruction) => self.handle_instruction(*instruction),
            Action::Toggle(Toggle::BlockAddress) => self.toggle_address_display(),
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

#[cfg(test)]
mod tests {
    use proptest_arbitrary_interop::arb;
    use ratatui::backend::TestBackend;
    use test_strategy::proptest;
    use triton_vm::prelude::*;

    use crate::args::TuiArgs;

    use super::*;

    /// Since `TextArea` is not `Arbitrary`, implement `Arbitrary` for `Memory` manually.
    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    struct ArbitraryMemory {
        #[strategy(arb())]
        most_recent_address: BFieldElement,

        #[strategy(arb())]
        user_address: Option<BFieldElement>,

        text_area_input: String,
        text_area_in_focus: bool,
        show_block_addresses: bool,

        #[strategy(arb())]
        undo_stack: Vec<UndoInformation>,
    }

    #[proptest]
    fn render_arbitrary_vm_state(
        arb_memory: ArbitraryMemory,
        #[strategy(arb())] program: Program,
        #[strategy(arb())] mut vm_state: VMState,
    ) {
        let mut memory = Memory {
            most_recent_address: arb_memory.most_recent_address,
            user_address: arb_memory.user_address,
            text_area: TextArea::new(vec![arb_memory.text_area_input]),
            text_area_in_focus: arb_memory.text_area_in_focus,
            show_block_addresses: arb_memory.show_block_addresses,
            undo_stack: arb_memory.undo_stack,
        };

        vm_state.program = program.instructions.clone();
        let mut complete_state = TritonVMState::new(&TuiArgs::default()).unwrap();
        complete_state.vm_state = vm_state;
        complete_state.program = program;

        let backend = TestBackend::new(150, 50);
        let mut terminal = Terminal::new(backend)?;
        terminal
            .draw(|f| memory.draw(f, &complete_state).unwrap())
            .unwrap();
    }
}
