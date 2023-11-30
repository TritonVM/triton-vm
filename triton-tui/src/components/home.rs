use color_eyre::eyre::Result;
use ratatui::prelude::*;
use ratatui::widgets::block::*;
use ratatui::widgets::*;
use strum::EnumCount;
use tokio::sync::mpsc::UnboundedSender;
use tracing::info;

use triton_vm::error::InstructionError;
use triton_vm::instruction::LabelledInstruction;
use triton_vm::op_stack::OpStackElement;

use crate::action::Action;
use crate::config::Config;

use super::Component;
use super::Frame;

#[derive(Debug)]
pub(crate) struct Home {
    command_tx: Option<UnboundedSender<Action>>,
    config: Config,
    program: triton_vm::Program,
    vm_state: triton_vm::vm::VMState,
    max_address: u64,
    error: Option<InstructionError>,
}

impl Home {
    pub fn new() -> Self {
        let program = triton_vm::example_programs::VERIFY_SUDOKU.clone();
        let max_address = program.len_bwords() as u64;

        let public_input = vec![
            1, 2, 3, /**/ 4, 5, 6, /**/ 7, 8, 9, //
            4, 5, 6, /**/ 7, 8, 9, /**/ 1, 2, 3, //
            7, 8, 9, /**/ 1, 2, 3, /**/ 4, 5, 6, //
            /*************************************/
            2, 3, 4, /**/ 5, 6, 7, /**/ 8, 9, 1, //
            5, 6, 7, /**/ 8, 9, 1, /**/ 2, 3, 4, //
            8, 9, 1, /**/ 2, 3, 4, /**/ 5, 6, 7, //
            /*************************************/
            3, 4, 5, /**/ 6, 7, 8, /**/ 9, 1, 2, //
            6, 7, 8, /**/ 9, 1, 2, /**/ 3, 4, 5, //
            9, 1, 2, /**/ 3, 4, 5, /**/ 6, 7, 8, //
        ]
        .into();
        let non_determinism = [].into();
        let vm_state = triton_vm::vm::VMState::new(&program, public_input, non_determinism);

        Self {
            command_tx: None,
            config: Config::default(),
            program,
            vm_state,
            max_address,
            error: None,
        }
    }

    fn program_step(&mut self) -> Result<()> {
        if !self.vm_state.halting && self.error.is_none() {
            let maybe_error = self.vm_state.step();
            if let Err(err) = maybe_error {
                info!("Error stepping VM: {err}");
                self.error = Some(err);
            }
        }
        Ok(())
    }

    fn address_render_width(&self) -> usize {
        format!("{}", self.max_address).len()
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
        let block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(title)
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
        let mut text = match self.error {
            Some(ref err) => format!("{err}"),
            None => String::new(),
        };
        if self.vm_state.halting {
            text = "Triton VM halted gracefully".to_string();
        }

        let block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1));
        let paragraph = Paragraph::new(text).block(block).alignment(Alignment::Left);
        f.render_widget(paragraph, area);
    }
}

impl Component for Home {
    fn register_action_handler(&mut self, tx: UnboundedSender<Action>) -> Result<()> {
        self.command_tx = Some(tx);
        Ok(())
    }

    fn register_config_handler(&mut self, config: Config) -> Result<()> {
        self.config = config;
        Ok(())
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::ProgramStep => self.program_step()?,
            Action::Tick => info!("tick"),
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
