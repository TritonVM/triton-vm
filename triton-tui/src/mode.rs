use serde::Deserialize;
use serde::Serialize;

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum Mode {
    #[default]
    Home,
}
