use std::time::Instant;

use color_eyre::eyre::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use ratatui::Frame;

use crate::action::Action;
use crate::triton_vm_state::TritonVMState;

use super::Component;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FpsCounter {
    app_start_time: Instant,
    app_ticks: u32,
    app_ticks_per_s: f64,

    render_start_time: Instant,
    render_frames: u32,
    render_frames_per_s: f64,
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            app_start_time: Instant::now(),
            app_ticks: 0,
            app_ticks_per_s: 0.0,
            render_start_time: Instant::now(),
            render_frames: 0,
            render_frames_per_s: 0.0,
        }
    }

    fn app_tick(&mut self) -> Result<()> {
        self.app_ticks += 1;
        let now = Instant::now();
        let elapsed = (now - self.app_start_time).as_secs_f64();
        if elapsed >= 1.0 {
            self.app_ticks_per_s = self.app_ticks as f64 / elapsed;
            self.app_start_time = now;
            self.app_ticks = 0;
        }
        Ok(())
    }

    fn render_tick(&mut self) -> Result<()> {
        self.render_frames += 1;
        let now = Instant::now();
        let elapsed = (now - self.render_start_time).as_secs_f64();
        if elapsed >= 1.0 {
            self.render_frames_per_s = self.render_frames as f64 / elapsed;
            self.render_start_time = now;
            self.render_frames = 0;
        }
        Ok(())
    }
}

impl Component for FpsCounter {
    fn update(&mut self, action: Action) -> Result<Option<Action>> {
        match action {
            Action::Tick => self.app_tick()?,
            Action::Render => self.render_tick()?,
            _ => {}
        }
        Ok(None)
    }

    fn draw(&mut self, frame: &mut Frame<'_>, _: &TritonVMState) -> Result<()> {
        let constraints = vec![
            Constraint::Length(1), // first row
            Constraint::Min(0),
        ];
        let rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(frame.size());

        let rect = rects[0];

        let s = format!(
            "{:.2} ticks per sec (app) {:.2} frames per sec (render)",
            self.app_ticks_per_s, self.render_frames_per_s
        );
        let block = Block::default().title(block::Title::from(s.dim()).alignment(Alignment::Right));
        frame.render_widget(block, rect);
        Ok(())
    }
}
