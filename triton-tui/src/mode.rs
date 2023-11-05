use serde::*;
use strum::EnumCount;

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, EnumCount)]
#[repr(usize)]
pub(crate) enum Mode {
    #[default]
    Home,
    Help,
}

impl Mode {
    pub const fn id(self) -> usize {
        self as usize
    }
}
