use crate::components::Component;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Memory {
    /// The address touched last by any `read_mem` or `write_mem` instruction.
    pub most_recent_address: u64,

    /// The address to show. Can be manually set (and unset) by the user.
    pub user_address: Option<u64>,

    pub undo_stack: Vec<UndoInformation>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UndoInformation {
    pub most_recent_address: u64,
    pub user_address: Option<u64>,
}

impl Component for Memory {}
