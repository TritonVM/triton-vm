use color_eyre::eyre::anyhow;
use color_eyre::eyre::Result;
use fs_err as fs;
use itertools::Itertools;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::*;
use strum::EnumCount;
use tracing::info;

use triton_vm::error::InstructionError;
use triton_vm::instruction::*;
use triton_vm::op_stack::OpStackElement;
use triton_vm::vm::VMState;
use triton_vm::*;

use crate::action::Action;
use crate::args::Args;

use super::Component;
use super::Frame;

#[derive(Debug)]
pub(crate) struct Home {
    args: Args,
    program: Program,
    non_determinism: NonDeterminism<BFieldElement>,
    vm_state: VMState,
    error: Option<InstructionError>,
}

impl Home {
    pub fn new(args: Args) -> Result<Self> {
        let program = Self::program_from_args(&args)?;
        let public_input = Self::public_input_from_args(&args)?;

        let non_determinism = NonDeterminism::default();
        let vm_state = VMState::new(&program, public_input.clone(), non_determinism.clone());

        let home = Self {
            args,
            program,
            non_determinism,
            vm_state,
            error: None,
        };
        Ok(home)
    }

    fn program_from_args(args: &Args) -> Result<Program> {
        let source_code = fs::read_to_string(&args.program)?;
        let program = Program::from_code(&source_code)
            .map_err(|err| anyhow!("program parsing error: {err}"))?;
        Ok(program)
    }

    fn public_input_from_args(args: &Args) -> Result<PublicInput> {
        let Some(input_path) = args.input.clone() else {
            return Ok(PublicInput::default());
        };
        let file_content = fs::read_to_string(input_path)?;
        let string_tokens = file_content.split_whitespace();
        let mut elements = vec![];
        for string_token in string_tokens {
            let element = string_token.parse::<u64>()?;
            elements.push(element.into());
        }
        Ok(PublicInput::new(elements))
    }

    fn vm_has_stopped(&self) -> bool {
        self.vm_state.halting || self.error.is_some()
    }

    fn vm_is_running(&self) -> bool {
        !self.vm_has_stopped()
    }

    fn at_breakpoint(&self) -> bool {
        let ip = self.vm_state.instruction_pointer as u64;
        self.program.is_breakpoint(ip)
    }

    /// Handle [`Action::ProgramContinue`].
    fn program_continue(&mut self) {
        self.program_step();
        while self.vm_is_running() && !self.at_breakpoint() {
            self.program_step();
        }
    }

    /// Handle [`Action::ProgramStep`].
    fn program_step(&mut self) {
        if self.vm_has_stopped() {
            return;
        }
        let maybe_error = self.vm_state.step();
        if let Err(err) = maybe_error {
            info!("Error stepping VM: {err}");
            self.error = Some(err);
        }
    }

    /// Handle [`Action::ProgramNext`].
    fn program_next(&mut self) {
        let instruction = self.vm_state.current_instruction();
        let instruction_is_call = matches!(instruction, Ok(Instruction::Call(_)));
        self.program_step();
        if instruction_is_call {
            self.program_finish();
        }
    }

    /// Handle [`Action::ProgramFinish`].
    fn program_finish(&mut self) {
        let current_jump_stack_depth = self.vm_state.jump_stack.len();
        while self.vm_is_running() && self.vm_state.jump_stack.len() >= current_jump_stack_depth {
            self.program_step();
        }
    }

    fn program_reset(&mut self) -> Result<()> {
        self.program = Self::program_from_args(&self.args)?;
        let public_input = Self::public_input_from_args(&self.args)?;
        self.vm_state = VMState::new(&self.program, public_input, self.non_determinism.clone());
        self.error = None;
        Ok(())
    }

    fn address_render_width(&self) -> usize {
        let max_address = self.program.len_bwords();
        max_address.to_string().len()
    }

    fn distribute_area_for_widgets(area: Rect) -> WidgetAreas {
        let message_box_height = Constraint::Min(2);
        let constraints = [Constraint::Percentage(100), message_box_height];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);
        let state_area = layout[0];
        let message_box_area = layout[1];

        let op_stack_widget_width = Constraint::Min(32);
        let remaining_width = Constraint::Percentage(100);
        let state_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([op_stack_widget_width, remaining_width])
            .split(state_area);
        let op_stack_area = state_layout[0];
        let program_and_call_stack_area = state_layout[1];

        let program_widget_width = Constraint::Percentage(50);
        let call_stack_widget_width = Constraint::Percentage(50);
        let constraints = [program_widget_width, call_stack_widget_width];
        let program_and_call_stack_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(program_and_call_stack_area);

        WidgetAreas {
            op_stack: op_stack_area,
            program: program_and_call_stack_layout[0],
            call_stack: program_and_call_stack_layout[1],
            message_box: message_box_area,
        }
    }

    fn render_op_stack_widget(&self, f: &mut Frame, area: Rect) {
        let stack_size = self.vm_state.op_stack.stack.len();
        let title = format!(" Stack (size: {stack_size:>4}) ");
        let title = Title::from(title).alignment(Alignment::Left);
        let num_padding_lines = (area.height as usize).saturating_sub(stack_size + 3);
        let mut text = vec![Line::from(""); num_padding_lines];
        for (i, st) in self.vm_state.op_stack.stack.iter().rev().enumerate() {
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
        f.render_widget(paragraph, area);
    }

    fn render_program_widget(&self, f: &mut Frame, area: Rect) {
        let cycle_count = self.vm_state.cycle_count;
        let title = format!(" Program (cycle: {cycle_count:>5}) ");
        let title = Title::from(title).alignment(Alignment::Left);
        let halting = match self.vm_state.halting {
            true => Title::from(" HALT ".bold().green()).alignment(Alignment::Center),
            false => Title::from(""),
        };

        let address_width = self.address_render_width();
        let mut address = 0;
        let mut text = vec![];
        let instruction_pointer = self.vm_state.instruction_pointer;
        let mut line_number_of_ip = 0;
        for (line_number, labelled_instruction) in
            self.program.labelled_instructions().into_iter().enumerate()
        {
            let mut ip_points_here = instruction_pointer == address;
            ip_points_here &= matches!(labelled_instruction, LabelledInstruction::Instruction(_));
            if ip_points_here {
                line_number_of_ip = line_number;
            }
            let ip = match ip_points_here {
                true => Span::from("→").bold(),
                false => Span::from(" "),
            };
            let line_number = match labelled_instruction {
                LabelledInstruction::Instruction(_) => format!(" {address:>address_width$}"),
                _ => format!(" {:>address_width$}", ""),
            };
            let line_number = Span::from(line_number).dim();
            let separator = Span::from("  ");
            let instruction = Span::from(format!("{labelled_instruction}"));
            let line = Line::from(vec![ip, line_number, separator, instruction]);
            text.push(line);
            if let LabelledInstruction::Instruction(instruction) = labelled_instruction {
                address += instruction.size();
            }
        }

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            ..symbols::border::ROUNDED
        };

        let halting = halting.position(Position::Bottom);
        let halting = halting.alignment(Alignment::Center);
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(title)
            .title(halting)
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set);
        let render_area_for_lines = area.height.saturating_sub(3);
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
        f.render_widget(paragraph, area);
    }

    fn render_call_stack_widget(&self, f: &mut Frame, area: Rect) {
        let jump_stack_depth = self.vm_state.jump_stack.len();
        let title = format!(" Calls (depth: {jump_stack_depth:>3}) ");
        let title = Title::from(title).alignment(Alignment::Left);

        let num_padding_lines = (area.height as usize).saturating_sub(jump_stack_depth + 3);
        let mut text = vec![Line::from(""); num_padding_lines];
        let address_width = self.address_render_width();
        for (return_address, call_address) in self.vm_state.jump_stack.iter().rev() {
            let return_address = return_address.value();
            let call_address = call_address.value();
            let addresses = Span::from(format!(
                "({return_address:>address_width$}, {call_address:>address_width$})"
            ));
            let separator = Span::from("  ");
            let label = Span::from(self.program.label_for_address(call_address));
            let line = Line::from(vec![addresses, separator, label]);
            text.push(line);
        }

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(title)
            .borders(Borders::ALL)
            .border_set(border_set);
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        f.render_widget(paragraph, area);
    }

    fn render_message_widget(&self, f: &mut Frame, area: Rect) {
        let mut line = Line::from("");
        if let Some(message) = self.maybe_render_public_output() {
            line = message;
        }
        if let Some(message) = self.maybe_render_error_message() {
            line = message;
        }

        let block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1));
        let paragraph = Paragraph::new(line).block(block).alignment(Alignment::Left);
        f.render_widget(paragraph, area);
    }

    fn maybe_render_public_output(&self) -> Option<Line> {
        if self.vm_state.public_output.is_empty() {
            return None;
        }
        let header = Span::from("Public output").bold();
        let colon = Span::from(": [");
        let output = self.vm_state.public_output.iter().join(", ");
        let output = Span::from(output);
        let footer = Span::from("]");
        Some(Line::from(vec![header, colon, output, footer]))
    }

    fn maybe_render_error_message(&self) -> Option<Line> {
        let header = Span::from("ERROR ").red().bold();
        let err = Span::from(self.error?.to_string());
        Some(Line::from(vec![header, err]))
    }
}

impl Component for Home {
    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::ProgramContinue => self.program_continue(),
            Action::ProgramStep => self.program_step(),
            Action::ProgramNext => self.program_next(),
            Action::ProgramFinish => self.program_finish(),
            Action::ProgramReset => self.program_reset()?,
            _ => {}
        }
        Ok(None)
    }

    fn draw(&mut self, f: &mut Frame<'_>, area: Rect) -> Result<()> {
        let widget_areas = Self::distribute_area_for_widgets(area);
        self.render_op_stack_widget(f, widget_areas.op_stack);
        self.render_program_widget(f, widget_areas.program);
        self.render_call_stack_widget(f, widget_areas.call_stack);
        self.render_message_widget(f, widget_areas.message_box);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WidgetAreas {
    op_stack: Rect,
    program: Rect,
    call_stack: Rect,
    message_box: Rect,
}
