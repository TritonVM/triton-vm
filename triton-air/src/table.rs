use arbitrary::Arbitrary;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;

use crate::AIR;
use crate::table::cascade::CascadeTable;
use crate::table::hash::HashTable;
use crate::table::jump_stack::JumpStackTable;
use crate::table::lookup::LookupTable;
use crate::table::op_stack::OpStackTable;
use crate::table::processor::ProcessorTable;
use crate::table::program::ProgramTable;
use crate::table::ram::RamTable;
use crate::table::u32::U32Table;

pub mod cascade;
pub mod hash;
pub mod jump_stack;
pub mod lookup;
pub mod op_stack;
pub mod processor;
pub mod program;
pub mod ram;
pub mod u32;

/// The total number of main columns across all tables.
/// The degree lowering columns are _not_ included.
pub const NUM_MAIN_COLUMNS: usize = <ProgramTable as AIR>::MainColumn::COUNT
    + <ProcessorTable as AIR>::MainColumn::COUNT
    + <OpStackTable as AIR>::MainColumn::COUNT
    + <RamTable as AIR>::MainColumn::COUNT
    + <JumpStackTable as AIR>::MainColumn::COUNT
    + <HashTable as AIR>::MainColumn::COUNT
    + <CascadeTable as AIR>::MainColumn::COUNT
    + <LookupTable as AIR>::MainColumn::COUNT
    + <U32Table as AIR>::MainColumn::COUNT;

/// The total number of auxiliary columns across all tables.
/// The degree lowering columns as well as any randomizer polynomials are _not_
/// included.
pub const NUM_AUX_COLUMNS: usize = <ProgramTable as AIR>::AuxColumn::COUNT
    + <ProcessorTable as AIR>::AuxColumn::COUNT
    + <OpStackTable as AIR>::AuxColumn::COUNT
    + <RamTable as AIR>::AuxColumn::COUNT
    + <JumpStackTable as AIR>::AuxColumn::COUNT
    + <HashTable as AIR>::AuxColumn::COUNT
    + <CascadeTable as AIR>::AuxColumn::COUNT
    + <LookupTable as AIR>::AuxColumn::COUNT
    + <U32Table as AIR>::AuxColumn::COUNT;

pub const PROGRAM_TABLE_START: usize = 0;
pub const PROGRAM_TABLE_END: usize = PROGRAM_TABLE_START + <ProgramTable as AIR>::MainColumn::COUNT;
pub const PROCESSOR_TABLE_START: usize = PROGRAM_TABLE_END;
pub const PROCESSOR_TABLE_END: usize =
    PROCESSOR_TABLE_START + <ProcessorTable as AIR>::MainColumn::COUNT;
pub const OP_STACK_TABLE_START: usize = PROCESSOR_TABLE_END;
pub const OP_STACK_TABLE_END: usize =
    OP_STACK_TABLE_START + <OpStackTable as AIR>::MainColumn::COUNT;
pub const RAM_TABLE_START: usize = OP_STACK_TABLE_END;
pub const RAM_TABLE_END: usize = RAM_TABLE_START + <RamTable as AIR>::MainColumn::COUNT;
pub const JUMP_STACK_TABLE_START: usize = RAM_TABLE_END;
pub const JUMP_STACK_TABLE_END: usize =
    JUMP_STACK_TABLE_START + <JumpStackTable as AIR>::MainColumn::COUNT;
pub const HASH_TABLE_START: usize = JUMP_STACK_TABLE_END;
pub const HASH_TABLE_END: usize = HASH_TABLE_START + <HashTable as AIR>::MainColumn::COUNT;
pub const CASCADE_TABLE_START: usize = HASH_TABLE_END;
pub const CASCADE_TABLE_END: usize = CASCADE_TABLE_START + <CascadeTable as AIR>::MainColumn::COUNT;
pub const LOOKUP_TABLE_START: usize = CASCADE_TABLE_END;
pub const LOOKUP_TABLE_END: usize = LOOKUP_TABLE_START + <LookupTable as AIR>::MainColumn::COUNT;
pub const U32_TABLE_START: usize = LOOKUP_TABLE_END;
pub const U32_TABLE_END: usize = U32_TABLE_START + <U32Table as AIR>::MainColumn::COUNT;

pub const AUX_PROGRAM_TABLE_START: usize = 0;
pub const AUX_PROGRAM_TABLE_END: usize =
    AUX_PROGRAM_TABLE_START + <ProgramTable as AIR>::AuxColumn::COUNT;
pub const AUX_PROCESSOR_TABLE_START: usize = AUX_PROGRAM_TABLE_END;
pub const AUX_PROCESSOR_TABLE_END: usize =
    AUX_PROCESSOR_TABLE_START + <ProcessorTable as AIR>::AuxColumn::COUNT;
pub const AUX_OP_STACK_TABLE_START: usize = AUX_PROCESSOR_TABLE_END;
pub const AUX_OP_STACK_TABLE_END: usize =
    AUX_OP_STACK_TABLE_START + <OpStackTable as AIR>::AuxColumn::COUNT;
pub const AUX_RAM_TABLE_START: usize = AUX_OP_STACK_TABLE_END;
pub const AUX_RAM_TABLE_END: usize = AUX_RAM_TABLE_START + <RamTable as AIR>::AuxColumn::COUNT;
pub const AUX_JUMP_STACK_TABLE_START: usize = AUX_RAM_TABLE_END;
pub const AUX_JUMP_STACK_TABLE_END: usize =
    AUX_JUMP_STACK_TABLE_START + <JumpStackTable as AIR>::AuxColumn::COUNT;
pub const AUX_HASH_TABLE_START: usize = AUX_JUMP_STACK_TABLE_END;
pub const AUX_HASH_TABLE_END: usize = AUX_HASH_TABLE_START + <HashTable as AIR>::AuxColumn::COUNT;
pub const AUX_CASCADE_TABLE_START: usize = AUX_HASH_TABLE_END;
pub const AUX_CASCADE_TABLE_END: usize =
    AUX_CASCADE_TABLE_START + <CascadeTable as AIR>::AuxColumn::COUNT;
pub const AUX_LOOKUP_TABLE_START: usize = AUX_CASCADE_TABLE_END;
pub const AUX_LOOKUP_TABLE_END: usize =
    AUX_LOOKUP_TABLE_START + <LookupTable as AIR>::AuxColumn::COUNT;
pub const AUX_U32_TABLE_START: usize = AUX_LOOKUP_TABLE_END;
pub const AUX_U32_TABLE_END: usize = AUX_U32_TABLE_START + <U32Table as AIR>::AuxColumn::COUNT;

/// Uniquely determines one of Triton VM's tables.
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter, Arbitrary)]
pub enum TableId {
    Program,
    Processor,
    OpStack,
    Ram,
    JumpStack,
    Hash,
    Cascade,
    Lookup,
    U32,
}
