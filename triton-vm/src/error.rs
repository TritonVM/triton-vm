use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;

use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;

use InstructionError::*;

#[derive(Debug, Clone)]
pub enum InstructionError {
    InstructionPointerUnderflow,
    InstructionPointerOverflow(usize),
    OpStackTooShallow,
    JumpStackTooShallow,
    AssertionFailed(usize, u32, BFieldElement),
    InverseOfZero,
    RunawayInstructionArg,
    UngracefulTermination,
    FailedU32Conversion(BFieldElement),
}

impl Display for InstructionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionPointerUnderflow => {
                write!(f, "Instruction pointer points to before start of program",)
            }

            InstructionPointerOverflow(ip) => {
                write!(f, "Instruction pointer {} points outside of program", ip)
            }

            OpStackTooShallow => {
                write!(f, "Operational stack is too shallow")
            }

            JumpStackTooShallow => {
                write!(f, "Jump stack does not contain return address")
            }

            AssertionFailed(ip, clk, st0) => {
                write!(
                    f,
                    "Assertion failed: st0 must be 1. ip: {}, clk: {}, st0: {}",
                    ip, clk, st0
                )
            }

            InverseOfZero => {
                write!(f, "0 does not have a multiplicative inverse")
            }

            RunawayInstructionArg => {
                write!(
                    f,
                    "A numeric argument to an instruction occurred out of place"
                )
            }

            UngracefulTermination => {
                write!(
                    f,
                    "The Virtual Machine must terminate using instruction Halt"
                )
            }

            FailedU32Conversion(word) => {
                write!(
                    f,
                    "Failed to convert BFieldElement {} into u32",
                    word.value()
                )
            }
        }
    }
}

impl Error for InstructionError {}

pub fn vm_err<T>(runtime_error: InstructionError) -> Result<T> {
    Err(vm_fail(runtime_error))
}

pub fn vm_fail(runtime_error: InstructionError) -> anyhow::Error {
    anyhow::Error::new(runtime_error)
}
