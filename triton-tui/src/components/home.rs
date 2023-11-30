use color_eyre::eyre::Result;
use crossterm::event::*;
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
        let program = triton_vm::example_programs::FIBONACCI_SEQUENCE.clone();
        let max_address = program.len_bwords() as u64;

        let public_input = vec![4].into();
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

    fn run_program(&mut self) -> Result<()> {
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

    fn render_op_stack_widget(&self, f: &mut Frame, op_stack_widget_area: Rect) {
        let stack_size = self.vm_state.op_stack.stack.len();
        let op_stack_title = format!(" Stack (size: {stack_size:>4}) ");
        let op_stack_title = Title::from(op_stack_title).alignment(Alignment::Left);
        let num_padding_lines =
            (op_stack_widget_area.height as usize).saturating_sub(stack_size + 3);
        let mut op_stack_text = vec![Line::from(""); num_padding_lines];
        for (i, st) in self.vm_state.op_stack.stack.iter().rev().enumerate() {
            let stack_index_style = match i {
                i if i < OpStackElement::COUNT => Style::new().bold(),
                _ => Style::new().gray(),
            };
            let stack_index = Span::from(format!("{i:>3}")).set_style(stack_index_style);
            let separator = Span::from("  ");
            let stack_element = Span::from(format!("{st}"));
            let line = Line::from(vec![stack_index, separator, stack_element]);
            op_stack_text.push(line);
        }

        let border_set = symbols::border::Set {
            bottom_left: symbols::line::ROUNDED.vertical_right,
            ..symbols::border::ROUNDED
        };
        let op_stack_block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set)
            .title(op_stack_title);
        let op_stack_paragraph = Paragraph::new(op_stack_text)
            .block(op_stack_block)
            .alignment(Alignment::Left);

        f.render_widget(op_stack_paragraph, op_stack_widget_area);
    }

    fn render_program_widget(&self, f: &mut Frame, program_widget_area: Rect) {
        let cycle_count = self.vm_state.cycle_count;
        let program_title = format!(" Program (cycle: {cycle_count:>5}) ");
        let program_title = Title::from(program_title).alignment(Alignment::Left);
        let address_width = self.address_render_width();
        let mut address = 0;
        let mut program_text = vec![];
        for labelled_instruction in self.program.labelled_instructions() {
            let ip_is_address = self.vm_state.instruction_pointer == address;
            let instruction_pointer = match labelled_instruction {
                LabelledInstruction::Instruction(_) if ip_is_address => Span::from("→").bold(),
                _ => Span::from(" "),
            };
            let address_text = match labelled_instruction {
                LabelledInstruction::Instruction(_) => format!(" {address:>address_width$}"),
                _ => format!(" {:>address_width$}", ""),
            };
            let address_text = Span::from(address_text);
            let separator = Span::from("  ");
            let instruction = Span::from(format!("{labelled_instruction}"));
            let line = Line::from(vec![
                instruction_pointer,
                address_text,
                separator,
                instruction,
            ]);
            program_text.push(line);
            if let LabelledInstruction::Instruction(instruction) = labelled_instruction {
                address += instruction.size();
            }
        }

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            ..symbols::border::ROUNDED
        };
        let program_block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(program_title)
            .borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM)
            .border_set(border_set);
        let program_paragraph = Paragraph::new(program_text)
            .block(program_block)
            .alignment(Alignment::Left);

        f.render_widget(program_paragraph, program_widget_area);
    }

    fn render_call_stack_widget(&self, f: &mut Frame, call_stack_widget_area: Rect) {
        let jump_stack_depth = self.vm_state.jump_stack.len();
        let call_stack_title = format!(" Calls (depth: {jump_stack_depth:>3}) ");
        let call_stack_title = Title::from(call_stack_title).alignment(Alignment::Left);

        let num_padding_lines =
            (call_stack_widget_area.height as usize).saturating_sub(jump_stack_depth + 3);
        let mut call_stack_text = vec![Line::from(""); num_padding_lines];
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
            call_stack_text.push(line);
        }

        let border_set = symbols::border::Set {
            top_left: symbols::line::ROUNDED.horizontal_down,
            bottom_left: symbols::line::ROUNDED.horizontal_up,
            bottom_right: symbols::line::ROUNDED.vertical_left,
            ..symbols::border::ROUNDED
        };
        let call_stack_block = Block::default()
            .padding(Padding::new(1, 1, 1, 0))
            .title(call_stack_title)
            .borders(Borders::ALL)
            .border_set(border_set);
        let call_stack_paragraph = Paragraph::new(call_stack_text)
            .block(call_stack_block)
            .alignment(Alignment::Left);

        f.render_widget(call_stack_paragraph, call_stack_widget_area);
    }

    fn render_message_widget(&self, f: &mut Frame, message_box_area: Rect) {
        let mut message_block_text = match self.error {
            Some(ref err) => format!("{err}"),
            None => String::new(),
        };
        if self.vm_state.halting {
            message_block_text = "Triton VM halted gracefully".to_string();
        }

        let message_block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1));
        let message_paragraph = Paragraph::new(message_block_text)
            .block(message_block)
            .alignment(Alignment::Left);
        f.render_widget(message_paragraph, message_box_area);
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

    fn handle_mouse_event(&mut self, event: MouseEvent) -> Result<Option<Action>> {
        let MouseEvent { kind, .. } = event;
        match kind {
            MouseEventKind::Down(MouseButton::Left) => Ok(Some(Action::RunProgram)),
            _ => Ok(None),
        }
    }

    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::RunProgram => self.run_program()?,
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
