use arbitrary::Arbitrary;
use serde::*;
use strum::EnumCount;

#[derive(
    Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, EnumCount, Arbitrary,
)]
#[repr(usize)]
pub(crate) enum Mode {
    #[default]
    Home,
    Memory,
    Help,
}

impl Mode {
    pub const fn id(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mode_is_home() {
        assert_eq!(Mode::Home, Mode::default());
    }

    #[test]
    fn default_mode_id_is_zero() {
        assert_eq!(0, Mode::default().id());
    }
}
