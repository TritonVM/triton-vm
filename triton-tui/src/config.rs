use std::collections::HashMap;

use color_eyre::eyre::Result;
use config::ConfigError;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use derive_deref::Deref;
use derive_deref::DerefMut;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use serde::de::Deserializer;
use serde::de::Error;
use serde::Deserialize;
use tracing::error;
use tracing::info;

use crate::action::Action;
use crate::args::*;
use crate::mode::Mode;

const DEFAULT_CONFIG: &str = include_str!("../.config/default_config.json");

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct Config {
    #[serde(default)]
    pub keybindings: KeyBindings,

    #[serde(default)]
    pub styles: Styles,
}

impl Config {
    pub fn new() -> Result<Self, ConfigError> {
        let default_config = serde_json::from_str(DEFAULT_CONFIG).map_err(|e| {
            let error = format!("Unable to parse default config: {e}");
            error!(error);
            ConfigError::custom(error)
        })?;

        let mut cfg = Self::aggregate_config_from_various_locations()?;
        cfg.add_missing_keybindings_using_defaults(&default_config);
        cfg.add_missing_styles_using_defaults(&default_config);

        Ok(cfg)
    }

    fn aggregate_config_from_various_locations() -> Result<Self, ConfigError> {
        let data_dir = get_data_dir();
        let config_dir = get_config_dir();
        let mut config_builder = config::Config::builder()
            .set_default("_data_dir", data_dir.to_str().unwrap())?
            .set_default("_config_dir", config_dir.to_str().unwrap())?;

        let config_files = [
            ("config.json", config::FileFormat::Json),
            ("config.yaml", config::FileFormat::Yaml),
            ("config.toml", config::FileFormat::Toml),
            ("config.ini", config::FileFormat::Ini),
        ];
        for (file, format) in config_files {
            let config_path = config_dir.join(file);
            if config_path.exists() {
                info!("Adding configuration file: {}", config_path.display());
                let config_file = config::File::from(config_path)
                    .format(format)
                    .required(false);
                config_builder = config_builder.add_source(config_file);
            } else {
                info!("Configuration file not found: {}", config_path.display());
            }
        }

        config_builder.build()?.try_deserialize()
    }

    fn add_missing_keybindings_using_defaults(&mut self, default_config: &Self) {
        for (&mode, default_bindings) in default_config.keybindings.iter() {
            let user_bindings = self.keybindings.entry(mode).or_default();
            for (key, cmd) in default_bindings {
                user_bindings
                    .entry(key.clone())
                    .or_insert_with(|| cmd.clone());
            }
        }
    }

    fn add_missing_styles_using_defaults(&mut self, default_config: &Self) {
        for (&mode, default_styles) in default_config.styles.iter() {
            let user_styles = self.styles.entry(mode).or_default();
            for (style_key, &style) in default_styles {
                user_styles
                    .entry(style_key.clone())
                    .or_insert_with(|| style);
            }
        }
    }
}

pub(crate) type KeyEvents = Vec<KeyEvent>;

#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub(crate) struct KeyBindings(pub HashMap<Mode, HashMap<KeyEvents, Action>>);

impl<'de> Deserialize<'de> for KeyBindings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let parsed_map = HashMap::<Mode, HashMap<String, Action>>::deserialize(deserializer)?;

        let mut keybindings = HashMap::new();
        for (mode, key_str_to_action_map) in parsed_map {
            let mut key_events_to_action_map = HashMap::new();
            for (key_str, action) in key_str_to_action_map {
                let key_events = parse_key_sequence(&key_str)
                    .map_err(|e| Error::custom(format!("Unable to parse `{key_str}`: {e}")))?;
                key_events_to_action_map.insert(key_events, action);
            }
            keybindings.insert(mode, key_events_to_action_map);
        }
        Ok(KeyBindings(keybindings))
    }
}

fn parse_key_event(raw: &str) -> Result<KeyEvent, String> {
    let raw_lower = raw.to_ascii_lowercase();
    let (remaining, modifiers) = extract_modifiers(&raw_lower);
    parse_key_code_with_modifiers(remaining, modifiers)
}

fn extract_modifiers(raw: &str) -> (&str, KeyModifiers) {
    let mut modifiers = KeyModifiers::empty();
    let mut current = raw;

    loop {
        match current {
            rest if rest.starts_with("ctrl-") => {
                modifiers.insert(KeyModifiers::CONTROL);
                current = &rest[5..];
            }
            rest if rest.starts_with("alt-") => {
                modifiers.insert(KeyModifiers::ALT);
                current = &rest[4..];
            }
            rest if rest.starts_with("shift-") => {
                modifiers.insert(KeyModifiers::SHIFT);
                current = &rest[6..];
            }
            _ => break,
        };
    }

    (current, modifiers)
}

fn parse_key_code_with_modifiers(
    raw: &str,
    mut modifiers: KeyModifiers,
) -> Result<KeyEvent, String> {
    let c = match raw {
        "esc" => KeyCode::Esc,
        "enter" => KeyCode::Enter,
        "left" => KeyCode::Left,
        "right" => KeyCode::Right,
        "up" => KeyCode::Up,
        "down" => KeyCode::Down,
        "home" => KeyCode::Home,
        "end" => KeyCode::End,
        "pageup" => KeyCode::PageUp,
        "pagedown" => KeyCode::PageDown,
        "backtab" => {
            modifiers.insert(KeyModifiers::SHIFT);
            KeyCode::BackTab
        }
        "backspace" => KeyCode::Backspace,
        "delete" => KeyCode::Delete,
        "insert" => KeyCode::Insert,
        "f1" => KeyCode::F(1),
        "f2" => KeyCode::F(2),
        "f3" => KeyCode::F(3),
        "f4" => KeyCode::F(4),
        "f5" => KeyCode::F(5),
        "f6" => KeyCode::F(6),
        "f7" => KeyCode::F(7),
        "f8" => KeyCode::F(8),
        "f9" => KeyCode::F(9),
        "f10" => KeyCode::F(10),
        "f11" => KeyCode::F(11),
        "f12" => KeyCode::F(12),
        "space" => KeyCode::Char(' '),
        "hyphen" | "minus" => KeyCode::Char('-'),
        "tab" => KeyCode::Tab,
        c if c.len() == 1 => {
            let mut c = c.chars().next().unwrap();
            if modifiers.contains(KeyModifiers::SHIFT) {
                c = c.to_ascii_uppercase();
            }
            KeyCode::Char(c)
        }
        _ => return Err(format!("Unable to parse {raw}")),
    };
    Ok(KeyEvent::new(c, modifiers))
}

fn _key_event_to_string(key_event: &KeyEvent) -> String {
    let char;
    let key_code = match key_event.code {
        KeyCode::Backspace => "backspace",
        KeyCode::Enter => "enter",
        KeyCode::Left => "left",
        KeyCode::Right => "right",
        KeyCode::Up => "up",
        KeyCode::Down => "down",
        KeyCode::Home => "home",
        KeyCode::End => "end",
        KeyCode::PageUp => "pageup",
        KeyCode::PageDown => "pagedown",
        KeyCode::Tab => "tab",
        KeyCode::BackTab => "backtab",
        KeyCode::Delete => "delete",
        KeyCode::Insert => "insert",
        KeyCode::F(c) => {
            char = format!("f({c})");
            &char
        }
        KeyCode::Char(' ') => "space",
        KeyCode::Char(c) => {
            char = c.to_string();
            &char
        }
        KeyCode::Esc => "esc",
        _ => "",
    };

    let mut key_string = Vec::with_capacity(4);
    if key_event.modifiers.intersects(KeyModifiers::CONTROL) {
        key_string.push("ctrl");
    }
    if key_event.modifiers.intersects(KeyModifiers::SHIFT) {
        key_string.push("shift");
    }
    if key_event.modifiers.intersects(KeyModifiers::ALT) {
        key_string.push("alt");
    }
    key_string.push(key_code);
    key_string.join("-")
}

fn parse_key_sequence(raw: &str) -> Result<KeyEvents, String> {
    let raw = raw
        .strip_prefix('<')
        .ok_or_else(|| format!("Missing `<` in `{raw}`"))?;
    let raw = raw
        .strip_suffix('>')
        .ok_or_else(|| format!("Missing `>` in `{raw}`"))?;
    let sequences = raw.split("><");

    sequences.map(parse_key_event).collect()
}

#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub(crate) struct Styles(pub HashMap<Mode, HashMap<String, Style>>);

impl<'de> Deserialize<'de> for Styles {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let parsed_map = HashMap::<Mode, HashMap<String, String>>::deserialize(deserializer)?;

        let styles = parsed_map
            .into_iter()
            .map(|(mode, inner_map)| {
                let converted_inner_map = inner_map
                    .into_iter()
                    .map(|(str, style)| (str, parse_style(&style)))
                    .collect();
                (mode, converted_inner_map)
            })
            .collect();

        Ok(Styles(styles))
    }
}

pub fn parse_style(line: &str) -> Style {
    let fg_bg_splitpoint = line.to_lowercase().find("on ").unwrap_or(line.len());
    let (fg, bg) = line.split_at(fg_bg_splitpoint);
    let (fg_color, fg_modifiers) = process_color_string(fg);
    let (bg_color, bg_modifiers) = process_color_string(&bg.replace("on ", ""));

    let mut style = Style::default();
    if let Some(fg) = parse_color(&fg_color) {
        style = style.fg(fg);
    }
    if let Some(bg) = parse_color(&bg_color) {
        style = style.bg(bg);
    }
    style = style.add_modifier(fg_modifiers | bg_modifiers);
    style
}

fn process_color_string(color_str: &str) -> (String, Modifier) {
    let color = color_str
        .replace("grey", "gray")
        .replace("bright ", "")
        .replace("bold ", "")
        .replace("underline ", "")
        .replace("inverse ", "");

    let mut modifiers = Modifier::empty();
    if color_str.contains("underline") {
        modifiers |= Modifier::UNDERLINED;
    }
    if color_str.contains("bold") {
        modifiers |= Modifier::BOLD;
    }
    if color_str.contains("inverse") {
        modifiers |= Modifier::REVERSED;
    }

    (color, modifiers)
}

fn parse_color(s: &str) -> Option<Color> {
    let s = s.trim_start();
    let s = s.trim_end();
    match s {
        s if s.contains("bright color") => {
            let s = s.trim_start_matches("bright ");
            let c = parse_color_as_color_index(s);
            Some(Color::Indexed(c.wrapping_shl(8)))
        }
        s if s.contains("color") => Some(Color::Indexed(parse_color_as_color_index(s))),
        s if s.contains("gray") => {
            let s = s.trim_start_matches("gray");
            let c = s.parse::<u8>().unwrap_or_default();
            Some(Color::Indexed(232 + c))
        }
        s if s.contains("rgb") => {
            let s = s.trim_start_matches("rgb");
            let red = (s.as_bytes()[0] as char).to_digit(10).unwrap_or_default() as u8;
            let green = (s.as_bytes()[1] as char).to_digit(10).unwrap_or_default() as u8;
            let blue = (s.as_bytes()[2] as char).to_digit(10).unwrap_or_default() as u8;
            let c = 16 + red * 36 + green * 6 + blue;
            Some(Color::Indexed(c))
        }
        "black" => Some(Color::Indexed(0)),
        "red" => Some(Color::Indexed(1)),
        "green" => Some(Color::Indexed(2)),
        "yellow" => Some(Color::Indexed(3)),
        "blue" => Some(Color::Indexed(4)),
        "magenta" => Some(Color::Indexed(5)),
        "cyan" => Some(Color::Indexed(6)),
        "white" => Some(Color::Indexed(7)),
        "bold black" => Some(Color::Indexed(8)),
        "bold red" => Some(Color::Indexed(9)),
        "bold green" => Some(Color::Indexed(10)),
        "bold yellow" => Some(Color::Indexed(11)),
        "bold blue" => Some(Color::Indexed(12)),
        "bold magenta" => Some(Color::Indexed(13)),
        "bold cyan" => Some(Color::Indexed(14)),
        "bold white" => Some(Color::Indexed(15)),
        _ => None,
    }
}

fn parse_color_as_color_index(s: &str) -> u8 {
    let maybe_color_index = s.trim_start_matches("color").parse();
    maybe_color_index.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use assert2::assert;

    use super::*;

    #[test]
    fn parse_default_style() {
        let style = parse_style("");
        assert!(Style::default() == style);
    }

    #[test]
    fn parse_foreground_style() {
        let style = parse_style("red");
        let red = Some(Color::Indexed(1));
        assert!(red == style.fg);
    }

    #[test]
    fn parse_background_style() {
        let style = parse_style("on blue");
        let blue = Some(Color::Indexed(4));
        assert!(blue == style.bg);
    }

    #[test]
    fn parse_style_modifiers() {
        let style = parse_style("underline red on blue");
        let red = Some(Color::Indexed(1));
        let blue = Some(Color::Indexed(4));
        assert!(red == style.fg);
        assert!(blue == style.bg);
    }

    #[test]
    fn parse_style_modifiers_with_multiple_backgrounds() {
        let style = parse_style("bold green on blue on underline red");
        let green = Some(Color::Indexed(2));
        assert!(green == style.fg);
        assert!(None == style.bg);
    }

    #[test]
    fn parse_color_string() {
        let (color, modifiers) = process_color_string("underline bold inverse gray");
        assert!("gray" == color);
        assert!(modifiers.contains(Modifier::UNDERLINED));
        assert!(modifiers.contains(Modifier::BOLD));
        assert!(modifiers.contains(Modifier::REVERSED));
    }

    #[test]
    fn parse_rgb_color() {
        let color = parse_color("rgb123");
        let digits = [1, 2, 3];

        let num_well_known_ansi_colors = 16;
        let ansi_color_resolution = 6_u8;
        let expected = digits
            .into_iter()
            .rev()
            .enumerate()
            .map(|(index, digit)| digit * ansi_color_resolution.pow(index as u32))
            .sum::<u8>()
            + num_well_known_ansi_colors;
        assert!(color == Some(Color::Indexed(expected)));
    }

    #[test]
    fn parse_unknown_color() {
        let no_color = parse_color("unknown");
        assert!(None == no_color);
    }

    #[test]
    fn quit_from_default_config() -> Result<()> {
        let c = Config::new()?;
        let quitting_key_sequence = parse_key_sequence("<q>").unwrap_or_default();
        let mode_home_key_to_event_map = c.keybindings.get(&Mode::Home).unwrap();
        let quitting_action = mode_home_key_to_event_map
            .get(&quitting_key_sequence)
            .unwrap();
        assert!(&Action::Quit == quitting_action);
        Ok(())
    }

    #[test]
    fn parse_keys_without_modifiers() {
        let empty_modifiers = KeyModifiers::empty();
        let key_code_a = KeyCode::Char('a');
        let key_event_a = KeyEvent::new(key_code_a, empty_modifiers);
        assert!(key_event_a == parse_key_event("a").unwrap());

        let key_event_enter = KeyEvent::new(KeyCode::Enter, empty_modifiers);
        assert!(key_event_enter == parse_key_event("enter").unwrap());

        let key_event_esc = KeyEvent::new(KeyCode::Esc, empty_modifiers);
        assert!(key_event_esc == parse_key_event("esc").unwrap());
    }

    #[test]
    fn parse_keys_with_modifiers() {
        let ctrl_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL);
        assert!(ctrl_a == parse_key_event("ctrl-a").unwrap());

        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        assert!(alt_enter == parse_key_event("alt-enter").unwrap());

        let shift_esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::SHIFT);
        assert!(shift_esc == parse_key_event("shift-esc").unwrap());
    }

    #[test]
    fn parse_keys_with_multiple_modifiers() {
        let ctrl_alt = KeyModifiers::CONTROL | KeyModifiers::ALT;
        let ctr_alt_a = KeyEvent::new(KeyCode::Char('a'), ctrl_alt);
        assert!(ctr_alt_a == parse_key_event("ctrl-alt-a").unwrap());

        let ctrl_shift = KeyModifiers::CONTROL | KeyModifiers::SHIFT;
        let ctr_shift_enter = KeyEvent::new(KeyCode::Enter, ctrl_shift);
        assert_eq!(
            ctr_shift_enter,
            parse_key_event("ctrl-shift-enter").unwrap()
        );
    }

    #[test]
    fn stringify_key_event() {
        let ctrl_alt = KeyModifiers::CONTROL | KeyModifiers::ALT;
        let ctrl_alt_a = KeyEvent::new(KeyCode::Char('a'), ctrl_alt);
        let generated_string = _key_event_to_string(&ctrl_alt_a);

        let expected = "ctrl-alt-a".to_string();
        assert!(expected == generated_string);
    }

    #[test]
    fn parsing_invalid_keys_gives_error() {
        assert!(let Err(_) = parse_key_event("invalid-key"));
        assert!(let Err(_) = parse_key_event("ctrl-invalid-key"));
    }

    #[test]
    fn key_parsing_is_case_insensitive() {
        let ctrl_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL);
        assert!(ctrl_a == parse_key_event("CTRL-a").unwrap());

        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        assert!(alt_enter == parse_key_event("AlT-eNtEr").unwrap());
    }
}
