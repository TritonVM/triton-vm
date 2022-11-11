import functools
from enum import IntEnum, IntFlag, auto

class Instruction(IntEnum):
    Halt = 0
    Pop = auto()
    Push = auto()
    Divine = auto()
    Dup = auto()
    Swap = auto()
    Nop = auto()
    Skiz = auto()
    Call = auto()
    Return = auto()
    Recurse = auto()
    Assert = auto()
    ReadMem = auto()
    WriteMem = auto()
    Hash = auto()
    DivineSibling = auto()
    AssertVector = auto()
    Add = auto()
    Mul = auto()
    Invert = auto()
    Split = auto()
    Eq = auto()
    Lsb = auto()
    XxAdd = auto()
    XxMul = auto()
    XInvert = auto()
    XbMul = auto()
    ReadIo = auto()
    WriteIo = auto()

class InstructionBucket(IntFlag):
    HasArg = auto()
    ShrinkStack = auto()

# ===

def in_bucket(instruction_bucket, instruction):
    if instruction_bucket == InstructionBucket.HasArg:
        return instruction in [Instruction.Push, Instruction.Dup, Instruction.Swap, Instruction.Call]
    if instruction_bucket == InstructionBucket.ShrinkStack:
        return instruction in [Instruction.Pop, Instruction.Skiz, Instruction.Assert, Instruction.WriteIo,
                               Instruction.Add, Instruction.Mul, Instruction.Eq, Instruction.XbMul]
    return False

def flag_set(instruction):
    instruction_flags = [bucket for bucket in InstructionBucket if in_bucket(bucket, instruction)]
    return functools.reduce(lambda x, y: x | y, instruction_flags, 0)

def opcode(instruction):
    instruction_flag_set = flag_set(instruction)
    index_within_flag_set = 0
    for inst in Instruction:
        if inst < instruction and instruction_flag_set == flag_set(inst):
            index_within_flag_set += 1
    return index_within_flag_set * 2**len(InstructionBucket) + instruction_flag_set

def print_all_opcodes():
    for instruction in Instruction:
        print(f"{opcode(instruction):>02} {str(instruction)}")

def print_max_opcode():
    print(f"highest opcode: {max([opcode(instruction) for instruction in Instruction])}")

# ===

def opcodes_are_unique_test():
    all_opcodes = [opcode(instruction) for instruction in Instruction]
    all_opcodes = sorted(all_opcodes)
    assert(list(set(all_opcodes)) == list(all_opcodes))

def tests():
    opcodes_are_unique_test()

# ===

tests()

print_all_opcodes()
print()
print_max_opcode()
