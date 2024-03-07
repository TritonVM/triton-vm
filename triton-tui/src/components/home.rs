use arbitrary::Arbitrary;
use color_eyre::eyre::Result;
use itertools::Itertools;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::*;
use triton_vm::instruction::*;
use triton_vm::op_stack::NUM_OP_STACK_REGISTERS;
use triton_vm::prelude::Tip5;

use crate::action::*;
use crate::element_type_hint::ElementTypeHint;
use crate::triton_vm_state::TritonVMState;

use super::Component;
use super::Frame;

#[derive(Debug, Copy, Clone, Arbitrary)]
pub(crate) struct Home {
    type_hints: bool,
    call_stack: bool,
    sponge: bool,
    inputs: bool,
}

impl Default for Home {
    fn default() -> Self {
        Self {
            type_hints: true,
            call_stack: true,
            sponge: false,
            inputs: true,
        }
    }
}

impl Home {
    fn address_render_width(state: &TritonVMState) -> usize {
        let max_address = state.program.len_bwords();
        max_address.to_string().len()
    }

    fn toggle_widget(&mut self, toggle: Toggle) {
        match toggle {
            Toggle::All => self.toggle_all_widgets(),
            Toggle::TypeHint => self.type_hints = !self.type_hints,
            Toggle::CallStack => self.call_stack = !self.call_stack,
            Toggle::SpongeState => self.sponge = !self.sponge,
            Toggle::Input => self.inputs = !self.inputs,
            Toggle::BlockAddress => (),
        };
    }

    fn toggle_all_widgets(&mut self) {
        let any_widget_is_shown = self.all_widget_visibilities().into_iter().any(|v| v);
        self.set_all_widgets_visibility_to(!any_widget_is_shown);
    }

    fn all_widget_visibilities(self) -> [bool; 4] {
        [self.type_hints, self.call_stack, self.sponge, self.inputs]
    }

    fn set_all_widgets_visibility_to(&mut self, visibility: bool) {
        self.type_hints = visibility;
        self.call_stack = visibility;
        self.sponge = visibility;
        self.inputs = visibility;
    }

    fn distribute_area_for_widgets(self, state: &TritonVMState, area: Rect) -> WidgetAreas {
        let public_input_height = match self.maybe_render_public_input(state).is_some() {
            true => Constraint::Length(2),
            false => Constraint::Length(0),
        };
        let secret_input_height = match self.maybe_render_secret_input(state).is_some() {
            true => Constraint::Length(2),
            false => Constraint::Length(0),
        };
        let message_box_height = Constraint::Length(2);
        let constraints = [
            Constraint::Fill(1),
            public_input_height,
            secret_input_height,
            message_box_height,
        ];
        let [state_area, public_input_area, secret_input_area, message_box_area] =
            Layout::vertical(constraints).areas(area);

        let op_stack_widget_width = Constraint::Length(30);
        let remaining_width = Constraint::Fill(1);
        let sponge_state_width = match self.sponge {
            true => Constraint::Length(32),
            false => Constraint::Length(1),
        };
        let [op_stack_area, remaining_area, sponge_state_area] =
            Layout::horizontal([op_stack_widget_width, remaining_width, sponge_state_width])
                .areas(state_area);

        let show = Constraint::Fill(1);
        let hide = Constraint::Length(0);
        let hints_program_calls_constraints = match (self.type_hints, self.call_stack) {
            (true, true) => [show, show, show],
            (true, false) => [show, show, hide],
            (false, true) => [hide, show, show],
            (false, false) => [hide, show, hide],
        };
        let [type_hint_area, program_area, call_stack_area] =
            Layout::horizontal(hints_program_calls_constraints).areas(remaining_area);

        WidgetAreas {
            op_stack: op_stack_area,
            type_hint: type_hint_area,
            program: program_area,
            call_stack: call_stack_area,
            sponge: sponge_state_area,
            public_input: public_input_area,
            secret_input: secret_input_area,
            message_box: message_box_area,
        }
    }

    fn render_op_stack_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let op_stack = &render_info.state.vm_state.op_stack.stack;
        let render_area = render_info.areas.op_stack;

        let stack_size = op_stack.len();
        let title = format!(" Stack (size: {stack_size:>4}) ");
        let title = Title::from(title).alignment(Alignment::Left);

        let border_set = symbols::border::Set {
            bottom_left: symbols::line::ROUNDED.vertical_right,
            ..symbols::border::ROUNDED
        };
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set)
            .title(title);

        let num_available_lines = block.inner(render_area).height as usize;
        let num_padding_lines = num_available_lines.saturating_sub(stack_size);
        let mut text = vec![Line::from(""); num_padding_lines];
        for (i, st) in op_stack.iter().rev().enumerate() {
            let stack_index_style = match i {
                i if i < NUM_OP_STACK_REGISTERS => Style::new().bold(),
                _ => Style::new().dim(),
            };
            let stack_index = Span::from(format!("{i:>3}")).set_style(stack_index_style);
            let separator = Span::from("  ");
            let stack_element = Span::from(format!("{st}"));
            let line = Line::from(vec![stack_index, separator, stack_element]);
            text.push(line);
        }
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_type_hint_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        if !self.type_hints {
            return;
        }
        let block = Block::default()
            .padding(Padding::new(0, 1, 1, 0))
            .borders(Borders::TOP | Borders::BOTTOM);
        let render_area = render_info.areas.type_hint;
        let type_hints = &render_info.state.type_hints.stack;

        let num_available_lines = block.inner(render_area).height as usize;
        let num_padding_lines = num_available_lines.saturating_sub(type_hints.len());
        let mut text = vec![Line::from(""); num_padding_lines];

        let highest_hint = type_hints.last().cloned().flatten();
        let lowest_hint = type_hints.first().cloned().flatten();

        text.push(ElementTypeHint::render(&highest_hint).into());
        for (hint_0, hint_1, hint_2) in type_hints.iter().rev().tuple_windows() {
            if ElementTypeHint::is_continuous_sequence(&[hint_0, hint_1, hint_2]) {
                text.push("⋅".dim().into());
            } else {
                text.push(ElementTypeHint::render(hint_1).into());
            }
        }
        text.push(ElementTypeHint::render(&lowest_hint).into());

        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_program_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let state = &render_info.state;
        let render_area = render_info.areas.program;

        let cycle_count = state.vm_state.cycle_count;
        let title = format!(" Program (cycle: {cycle_count:>5}) ");
        let title = Title::from(title).alignment(Alignment::Left);

        let address_width = Self::address_render_width(state).max(2);
        let mut address = 0;
        let mut text = vec![];
        let instruction_pointer = state.vm_state.instruction_pointer;
        let mut line_number_of_ip = 0;
        let mut is_breakpoint = false;
        for labelled_instruction in state.program.labelled_instructions() {
            if labelled_instruction == LabelledInstruction::Breakpoint {
                is_breakpoint = true;
                continue;
            }
            if let LabelledInstruction::TypeHint(_) = labelled_instruction {
                continue;
            }
            let ip_points_here = instruction_pointer == address
                && matches!(labelled_instruction, LabelledInstruction::Instruction(_));
            if ip_points_here {
                line_number_of_ip = text.len();
            }
            let ip = match ip_points_here {
                true => Span::from("→").bold(),
                false => Span::from(" "),
            };
            let mut gutter_item = match is_breakpoint {
                true => format!("{:>address_width$}  ", "🔴").into(),
                false => format!(" {address:>address_width$}  ").dim(),
            };
            if let LabelledInstruction::Label(_) = labelled_instruction {
                gutter_item = " ".into();
            }
            let instruction = Span::from(format!("{labelled_instruction}"));
            let line = Line::from(vec![ip, gutter_item, instruction]);
            text.push(line);
            if let LabelledInstruction::Instruction(instruction) = labelled_instruction {
                address += instruction.size();
            }
            is_breakpoint = false;
        }

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            ..symbols::border::ROUNDED
        };

        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(title)
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set);
        let render_area_for_lines = block.inner(render_area).height;
        let num_total_lines = text.len() as u16;
        let num_lines_to_show_at_top = render_area_for_lines / 2;
        let maximum_scroll_amount = num_total_lines.saturating_sub(render_area_for_lines);
        let num_lines_to_scroll = (line_number_of_ip as u16)
            .saturating_sub(num_lines_to_show_at_top)
            .min(maximum_scroll_amount);

        let paragraph = Paragraph::new(text)
            .block(block)
            .scroll((num_lines_to_scroll, 0));
        frame.render_widget(paragraph, render_area);
    }

    fn render_call_stack_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        if !self.call_stack {
            return;
        }

        let state = &render_info.state;
        let jump_stack = &state.vm_state.jump_stack;

        let jump_stack_depth = jump_stack.len();
        let title = format!(" Calls (depth: {jump_stack_depth:>3}) ");
        let title = Title::from(title).alignment(Alignment::Left);

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            ..symbols::border::ROUNDED
        };
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(title)
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set);
        let render_area = render_info.areas.call_stack;

        let num_available_lines = block.inner(render_area).height as usize;
        let num_padding_lines = num_available_lines.saturating_sub(jump_stack_depth);
        let mut text = vec![Line::from(""); num_padding_lines];

        let address_width = Self::address_render_width(state);
        for (return_address, call_address) in jump_stack.iter().rev() {
            let return_address = return_address.value();
            let call_address = call_address.value();
            let addresses = Span::from(format!(
                "({return_address:>address_width$}, {call_address:>address_width$})"
            ));
            let separator = Span::from("  ");
            let label = Span::from(state.program.label_for_address(call_address));
            let line = Line::from(vec![addresses, separator, label]);
            text.push(line);
        }
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_sponge_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let title = Title::from(" Sponge ");
        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        let borders = match self.sponge {
            true => Borders::ALL,
            false => Borders::TOP | Borders::RIGHT | Borders::BOTTOM,
        };
        let block = Block::default()
            .borders(borders)
            .border_set(border_set)
            .title(title)
            .padding(Padding::new(1, 1, 1, 0));

        let render_area = render_info.areas.sponge;
        let Some(Tip5 { state: sponge }) = &render_info.state.vm_state.sponge else {
            let paragraph = Paragraph::new("").block(block);
            frame.render_widget(paragraph, render_area);
            return;
        };

        let num_available_lines = block.inner(render_area).height as usize;
        let num_padding_lines = num_available_lines.saturating_sub(sponge.len());
        let mut text = vec![Line::from(""); num_padding_lines];
        for (i, sp) in sponge.iter().enumerate() {
            let sponge_index = Span::from(format!("{i:>3}")).dim();
            let separator = Span::from("  ");
            let sponge_element = Span::from(format!("{sp}"));
            let line = Line::from(vec![sponge_index, separator, sponge_element]);
            text.push(line);
        }
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_public_input_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let public_input = self
            .maybe_render_public_input(render_info.state)
            .unwrap_or_default();

        let border_set = symbols::border::Set {
            bottom_left: symbols::line::ROUNDED.vertical_right,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        let block = Block::default()
            .padding(Padding::horizontal(1))
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_set(border_set);
        let paragraph = Paragraph::new(public_input).block(block);
        frame.render_widget(paragraph, render_info.areas.public_input);
    }

    fn maybe_render_public_input(self, state: &TritonVMState) -> Option<Line> {
        if state.vm_state.public_input.is_empty() || !self.inputs {
            return None;
        }
        let header = Span::from("Public input").bold();
        let colon = Span::from(": [");
        let input = state.vm_state.public_input.iter().join(", ");
        let input = Span::from(input);
        let footer = Span::from("]");
        Some(Line::from(vec![header, colon, input, footer]))
    }

    fn render_secret_input_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let secret_input = self
            .maybe_render_secret_input(render_info.state)
            .unwrap_or_default();

        let border_set = symbols::border::Set {
            bottom_left: symbols::line::ROUNDED.vertical_right,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        let block = Block::default()
            .padding(Padding::horizontal(1))
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_set(border_set);
        let paragraph = Paragraph::new(secret_input).block(block);
        frame.render_widget(paragraph, render_info.areas.secret_input);
    }

    fn maybe_render_secret_input(self, state: &TritonVMState) -> Option<Line> {
        if state.vm_state.secret_individual_tokens.is_empty() || !self.inputs {
            return None;
        }
        let header = Span::from("Secret input").bold();
        let colon = Span::from(": [");
        let input = state.vm_state.secret_individual_tokens.iter().join(", ");
        let input = Span::from(input);
        let footer = Span::from("]");
        Some(Line::from(vec![header, colon, input, footer]))
    }

    fn render_message_widget(self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let message = self.message(render_info.state);
        let status = match render_info.state.vm_state.halting {
            true => Title::from(" HALT ".bold().green()),
            false => Title::default(),
        };

        let block = Block::default()
            .padding(Padding::horizontal(1))
            .title(status)
            .title_position(Position::Bottom)
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_type(BorderType::Rounded);
        let paragraph = Paragraph::new(message).block(block);
        frame.render_widget(paragraph, render_info.areas.message_box);
    }

    fn message(&self, state: &TritonVMState) -> Line {
        self.maybe_render_error_message(state)
            .or_else(|| self.maybe_render_warning_message(state))
            .or_else(|| self.maybe_render_public_output(state))
            .unwrap_or_else(|| self.render_welcome_message())
    }

    fn maybe_render_error_message(&self, state: &TritonVMState) -> Option<Line> {
        let message = state.error?.to_string().into();
        let error = "ERROR".bold().red();
        let colon = ": ".into();
        Some(Line::from(vec![error, colon, message]))
    }

    fn maybe_render_warning_message(&self, state: &TritonVMState) -> Option<Line> {
        let message = state.warning.as_ref()?.to_string().into();
        let warning = "WARNING".bold().yellow();
        let colon = ": ".into();
        Some(Line::from(vec![warning, colon, message]))
    }

    fn maybe_render_public_output(&self, state: &TritonVMState) -> Option<Line> {
        if state.vm_state.public_output.is_empty() {
            return None;
        }
        let header = Span::from("Public output").bold();
        let colon = Span::from(": [");
        let output = state.vm_state.public_output.iter().join(", ");
        let output = Span::from(output);
        let footer = Span::from("]");
        Some(Line::from(vec![header, colon, output, footer]))
    }

    fn render_welcome_message(&self) -> Line {
        let welcome = "Welcome to the Triton VM TUI! ".into();
        let help_hint = "Press `h` for help.".dim();
        Line::from(vec![welcome, help_hint])
    }
}

impl Component for Home {
    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        if let Action::Toggle(toggle) = action {
            self.toggle_widget(toggle);
        }
        Ok(None)
    }

    fn draw(&mut self, frame: &mut Frame<'_>, state: &TritonVMState) -> Result<()> {
        let widget_areas = self.distribute_area_for_widgets(state, frame.size());
        let render_info = RenderInfo {
            state,
            areas: widget_areas,
        };

        self.render_op_stack_widget(frame, render_info);
        self.render_type_hint_widget(frame, render_info);
        self.render_program_widget(frame, render_info);
        self.render_call_stack_widget(frame, render_info);
        self.render_sponge_widget(frame, render_info);
        self.render_public_input_widget(frame, render_info);
        self.render_secret_input_widget(frame, render_info);
        self.render_message_widget(frame, render_info);
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
struct RenderInfo<'s> {
    state: &'s TritonVMState,
    areas: WidgetAreas,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct WidgetAreas {
    op_stack: Rect,
    type_hint: Rect,
    program: Rect,
    call_stack: Rect,
    sponge: Rect,
    public_input: Rect,
    secret_input: Rect,
    message_box: Rect,
}

#[cfg(test)]
mod tests {
    use proptest_arbitrary_interop::arb;
    use ratatui::backend::TestBackend;
    use test_strategy::proptest;
    use triton_vm::prelude::*;

    use crate::args::TuiArgs;

    use super::*;

    #[proptest]
    fn render_arbitrary_vm_state(
        #[strategy(arb())] mut home: Home,
        #[strategy(arb())] program: Program,
        #[strategy(arb())] mut vm_state: VMState,
    ) {
        vm_state.program = program.instructions.clone();

        let mut complete_state = TritonVMState::new(&TuiArgs::default()).unwrap();
        complete_state.vm_state = vm_state;
        complete_state.program = program;

        let backend = TestBackend::new(150, 50);
        let mut terminal = Terminal::new(backend)?;
        terminal
            .draw(|f| home.draw(f, &complete_state).unwrap())
            .unwrap();
    }
}
