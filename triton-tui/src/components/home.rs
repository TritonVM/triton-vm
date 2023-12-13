use color_eyre::eyre::Result;
use itertools::Itertools;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::*;
use strum::EnumCount;

use triton_vm::instruction::*;
use triton_vm::op_stack::OpStackElement;

use crate::action::Action;
use crate::triton_vm_state::TritonVMState;
use crate::type_hint_stack::*;

use super::Component;
use super::Frame;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Home {
    show_type_hints: bool,
}

impl Default for Home {
    fn default() -> Self {
        let show_type_hints = true;
        Self { show_type_hints }
    }
}

impl Home {
    fn address_render_width(&self, state: &TritonVMState) -> usize {
        let max_address = state.program.len_bwords();
        max_address.to_string().len()
    }

    fn distribute_area_for_widgets(&self, state: &TritonVMState, area: Rect) -> WidgetAreas {
        let message_box_height = Constraint::Min(2);
        let constraints = [Constraint::Percentage(100), message_box_height];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);
        let state_area = layout[0];
        let message_box_area = layout[1];

        let op_stack_widget_width = Constraint::Min(30);
        let type_hint_widget = match self.show_type_hints {
            true => Constraint::Min(30),
            false => Constraint::Max(0),
        };
        let remaining_width = Constraint::Percentage(100);
        let sponge_state_width = match state.vm_state.sponge_state.is_some() {
            true => Constraint::Min(32),
            false => Constraint::Min(1),
        };
        let state_layout_constraints = [
            op_stack_widget_width,
            type_hint_widget,
            remaining_width,
            sponge_state_width,
        ];
        let state_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(state_layout_constraints)
            .split(state_area);
        let op_stack_area = state_layout[0];
        let type_hint_area = state_layout[1];
        let program_and_call_stack_area = state_layout[2];
        let sponge_state_area = state_layout[3];

        let program_widget_width = Constraint::Percentage(50);
        let call_stack_widget_width = Constraint::Percentage(50);
        let constraints = [program_widget_width, call_stack_widget_width];
        let program_and_call_stack_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(program_and_call_stack_area);

        WidgetAreas {
            op_stack: op_stack_area,
            type_hint: type_hint_area,
            program: program_and_call_stack_layout[0],
            call_stack: program_and_call_stack_layout[1],
            sponge: sponge_state_area,
            message_box: message_box_area,
        }
    }

    fn render_op_stack_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let op_stack = &render_info.state.vm_state.op_stack.stack;
        let render_area = render_info.areas.op_stack;

        let stack_size = op_stack.len();
        let title = format!(" Stack (size: {stack_size:>4}) ");
        let title = Title::from(title).alignment(Alignment::Left);
        let num_padding_lines = (render_area.height as usize).saturating_sub(stack_size + 3);
        let mut text = vec![Line::from(""); num_padding_lines];
        for (i, st) in op_stack.iter().rev().enumerate() {
            let stack_index_style = match i {
                i if i < OpStackElement::COUNT => Style::new().bold(),
                _ => Style::new().dim(),
            };
            let stack_index = Span::from(format!("{i:>3}")).set_style(stack_index_style);
            let separator = Span::from("  ");
            let stack_element = Span::from(format!("{st}"));
            let line = Line::from(vec![stack_index, separator, stack_element]);
            text.push(line);
        }

        let border_set = symbols::border::Set {
            bottom_left: symbols::line::ROUNDED.vertical_right,
            ..symbols::border::ROUNDED
        };
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set)
            .title(title);
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_type_hint_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        if !self.show_type_hints {
            return;
        }
        let render_area = render_info.areas.type_hint;
        let type_hints = &render_info.state.type_hint_stack.type_hints;

        let highest_hint = type_hints.last().cloned().flatten();
        let lowest_hint = type_hints.first().cloned().flatten();

        let num_padding_lines = (render_area.height as usize).saturating_sub(type_hints.len() + 3);
        let mut text = vec![Line::from(""); num_padding_lines];

        text.push(Self::render_element_type_hint(&highest_hint));
        for (hint_0, hint_1, hint_2) in type_hints.iter().rev().tuple_windows() {
            if ElementTypeHint::is_continuous_sequence(&[hint_0, hint_1, hint_2]) {
                text.push("⋅".dim().into());
            } else {
                text.push(Self::render_element_type_hint(hint_1));
            }
        }
        text.push(Self::render_element_type_hint(&lowest_hint));

        let block = Block::default()
            .padding(Padding::new(0, 1, 1, 0))
            .borders(Borders::TOP | Borders::BOTTOM);
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_element_type_hint(element_type_hint: &Option<ElementTypeHint>) -> Line {
        let Some(element_type_hint) = element_type_hint else {
            return Line::default();
        };

        let mut line = vec![];
        line.push(element_type_hint.variable_name.clone().into());
        if let Some(ref type_name) = element_type_hint.type_name {
            line.push(": ".dim());
            line.push(type_name.into());
        }
        if let Some(index) = element_type_hint.index {
            line.push(format!(" ({index})").dim());
        }
        line.into()
    }

    fn render_program_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let state = &render_info.state;
        let render_area = render_info.areas.program;

        let cycle_count = state.vm_state.cycle_count;
        let title = format!(" Program (cycle: {cycle_count:>5}) ");
        let title = Title::from(title).alignment(Alignment::Left);

        let address_width = self.address_render_width(state).max(2);
        let mut address = 0;
        let mut text = vec![];
        let instruction_pointer = state.vm_state.instruction_pointer;
        let mut line_number_of_ip = 0;
        let mut is_breakpoint = false;
        for (line_number, labelled_instruction) in state
            .program
            .labelled_instructions()
            .into_iter()
            .enumerate()
        {
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
                line_number_of_ip = line_number;
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
        let render_area_for_lines = render_area.height.saturating_sub(3);
        let num_total_lines = text.len() as u16;
        let num_lines_to_show_at_top = render_area_for_lines / 2;
        let maximum_scroll_amount = num_total_lines.saturating_sub(render_area_for_lines);
        let num_lines_to_scroll = (line_number_of_ip as u16)
            .saturating_sub(num_lines_to_show_at_top)
            .min(maximum_scroll_amount);

        let paragraph = Paragraph::new(text)
            .block(block)
            .alignment(Alignment::Left)
            .scroll((num_lines_to_scroll, 0));
        frame.render_widget(paragraph, render_area);
    }

    fn render_call_stack_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let state = &render_info.state;
        let jump_stack = &state.vm_state.jump_stack;
        let render_area = render_info.areas.call_stack;

        let jump_stack_depth = jump_stack.len();
        let title = format!(" Calls (depth: {jump_stack_depth:>3}) ");
        let title = Title::from(title).alignment(Alignment::Left);

        let num_padding_lines = (render_area.height as usize).saturating_sub(jump_stack_depth + 3);
        let mut text = vec![Line::from(""); num_padding_lines];
        let address_width = self.address_render_width(state);
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
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_sponge_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let sponge_state = &render_info.state.vm_state.sponge_state;
        let render_area = render_info.areas.sponge;

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        let right_border = Block::default()
            .borders(Borders::TOP | Borders::RIGHT | Borders::BOTTOM)
            .border_set(border_set);
        let Some(state) = sponge_state else {
            let paragraph = Paragraph::new("").block(right_border);
            frame.render_widget(paragraph, render_area);
            return;
        };

        let title = Title::from(" Sponge ").alignment(Alignment::Left);
        let num_padding_lines = (render_area.height as usize).saturating_sub(state.len() + 3);
        let mut text = vec![Line::from(""); num_padding_lines];
        for (i, sp) in state.iter().enumerate() {
            let sponge_index = Span::from(format!("{i:>3}")).dim();
            let separator = Span::from("  ");
            let sponge_element = Span::from(format!("{sp}"));
            let line = Line::from(vec![sponge_index, separator, sponge_element]);
            text.push(line);
        }

        let block = right_border
            .padding(Padding::new(1, 1, 1, 0))
            .title(title)
            .borders(Borders::ALL);
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_area);
    }

    fn render_message_widget(&self, frame: &mut Frame<'_>, render_info: RenderInfo) {
        let title = Title::from(" Triton VM ");
        let halting = match render_info.state.vm_state.halting {
            true => Title::from(" HALT ".bold().green()),
            false => Title::default(),
        };

        let state = &render_info.state;
        let mut line = Line::from("");
        if let Some(message) = self.maybe_render_public_output(state) {
            line = message;
        }
        if let Some(message) = self.maybe_render_warning_message(state) {
            line = message;
        }
        if let Some(message) = self.maybe_render_error_message(state) {
            line = message;
        }

        let block = Block::default()
            .padding(Padding::horizontal(1))
            .title(title)
            .title(halting)
            .title_position(Position::Bottom)
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_type(BorderType::Rounded);
        let paragraph = Paragraph::new(line).block(block).alignment(Alignment::Left);
        frame.render_widget(paragraph, render_info.areas.message_box);
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

    fn maybe_render_warning_message(&self, state: &TritonVMState) -> Option<Line> {
        let Some(ref message) = state.warning else {
            return None;
        };
        let warning = "WARNING".bold().yellow();
        let colon = ": ".into();
        let message = message.to_string().into();
        Some(Line::from(vec![warning, colon, message]))
    }

    fn maybe_render_error_message(&self, state: &TritonVMState) -> Option<Line> {
        let error = "ERROR".bold().red();
        let colon = ": ".into();
        let message = state.error?.to_string().into();
        Some(Line::from(vec![error, colon, message]))
    }
}

impl Component for Home {
    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        if matches!(action, Action::ToggleTypeHintDisplay) {
            self.show_type_hints = !self.show_type_hints;
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
        self.render_message_widget(frame, render_info);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderInfo<'s> {
    state: &'s TritonVMState,
    areas: WidgetAreas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WidgetAreas {
    op_stack: Rect,
    type_hint: Rect,
    program: Rect,
    call_stack: Rect,
    sponge: Rect,
    message_box: Rect,
}
