// See the corresponding attribute in triton_vm/lib.rs
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub use twenty_first;

pub mod error;
pub mod instruction;
pub mod op_stack;
pub mod parser;
pub mod program;

/// Compile an entire program written in [Triton assembly][tasm].
/// Triton VM can run the resulting [`Program`](program::Program); see there for
/// details.
///
/// It is possible to use string-like interpolation to insert instructions,
/// arguments, labels, or other substrings into the program.
///
/// # Examples
///
/// ```
/// # use triton_isa::triton_program;
/// # use twenty_first::prelude::*;
/// let program = triton_program!(
///     read_io 1 push 5 mul
///     call check_eq_15
///     push 17 write_io 1
///     halt
///     // assert that the top of the stack is 15
///     check_eq_15:
///         push 15 eq assert
///         return
/// );
/// ```
///
/// Any type with an appropriate [`Display`](std::fmt::Display) implementation
/// can be interpolated. This includes, for example, primitive types like `u64`
/// and `&str`, but also [`Instruction`](instruction::Instruction)s,
/// [`BFieldElement`](twenty_first::prelude::BFieldElement)s, and
/// [`Label`](instruction::LabelledInstruction)s, among others.
///
/// ```
/// # use twenty_first::prelude::*;
/// # use triton_isa::triton_program;
/// # use triton_isa::instruction::Instruction;
/// let element_0 = BFieldElement::new(0);
/// let label = "my_label";
/// let instruction_push = Instruction::Push(bfe!(42));
/// let dup_arg = 1;
/// let program = triton_program!(
///     push {element_0}
///     call {label} halt
///     {label}:
///        {instruction_push}
///        dup {dup_arg}
///        skiz recurse return
/// );
/// ```
///
/// # Panics
///
/// **Panics** if the program cannot be parsed.
/// Examples for parsing errors are:
/// - unknown (_e.g._ misspelled) instructions
/// - invalid instruction arguments, _e.g._, `push 1.5` or `swap 42`
/// - missing or duplicate labels
/// - invalid labels, _e.g._, using a reserved keyword or starting a label with
///   a digit
///
/// For a version that returns a `Result`, see
/// [`Program::from_code()`][from_code].
///
/// [tasm]: https://triton-vm.org/spec/instructions.html
/// [from_code]: program::Program::from_code
#[macro_export]
macro_rules! triton_program {
    {$($source_code:tt)*} => {{
        let labelled_instructions = $crate::triton_asm!($($source_code)*);
        $crate::program::Program::new(&labelled_instructions)
    }};
}

/// Compile [Triton assembly][tasm] into a list of labelled
/// [`Instruction`](instruction::LabelledInstruction)s.
/// Similar to [`triton_program!`](triton_program), it is possible to use
/// string-like interpolation to insert instructions, arguments, labels, or
/// other expressions.
///
/// Similar to [`vec!`], a single instruction can be repeated a specified number
/// of times.
///
/// Furthermore, a list of
/// [`LabelledInstruction`](instruction::LabelledInstruction)s can be inserted
/// like so: `{&list}`.
///
/// The labels for instruction `call`, if any, are also parsed. Instruction
/// `call` can refer to a label defined later in the program, _i.e.,_ labels are
/// not checked for existence or uniqueness by this parser.
///
/// # Examples
///
/// ```
/// # use triton_isa::triton_asm;
/// let push_argument = 42;
/// let instructions = triton_asm!(
///     push 1 call some_label
///     push {push_argument}
///     some_other_label: skiz halt return
/// );
/// assert_eq!(7, instructions.len());
/// ```
///
/// One instruction repeated several times:
///
/// ```
/// # use triton_isa::triton_asm;
/// # use triton_isa::instruction::LabelledInstruction;
/// # use triton_isa::instruction::AnInstruction::SpongeAbsorb;
/// let instructions = triton_asm![sponge_absorb; 3];
/// assert_eq!(3, instructions.len());
/// assert_eq!(LabelledInstruction::Instruction(SpongeAbsorb), instructions[0]);
/// assert_eq!(LabelledInstruction::Instruction(SpongeAbsorb), instructions[1]);
/// assert_eq!(LabelledInstruction::Instruction(SpongeAbsorb), instructions[2]);
/// ```
///
/// Inserting substring of labelled instructions:
///
/// ```
/// # use triton_isa::instruction::AnInstruction::Push;
/// # use triton_isa::instruction::AnInstruction::Pop;
/// # use triton_isa::op_stack::NumberOfWords::N1;
/// # use triton_isa::instruction::LabelledInstruction;
/// # use triton_isa::triton_asm;
/// # use twenty_first::prelude::*;
/// let insert_me = triton_asm!(
///     pop 1
///     nop
///     pop 1
/// );
/// let surrounding_code = triton_asm!(
///     push 0
///     {&insert_me}
///     push 1
/// );
/// # let zero = bfe!(0);
/// # assert_eq!(LabelledInstruction::Instruction(Push(zero)), surrounding_code[0]);
/// assert_eq!(LabelledInstruction::Instruction(Pop(N1)), surrounding_code[1]);
/// assert_eq!(LabelledInstruction::Instruction(Pop(N1)), surrounding_code[3]);
/// # let one = bfe!(1);
/// # assert_eq!(LabelledInstruction::Instruction(Push(one)), surrounding_code[4]);
/// ```
///
/// # Panics
///
/// **Panics** if the instructions cannot be parsed.
/// For examples, see [`triton_program!`](triton_program), with the exception
/// that labels are not checked for existence or uniqueness.
///
/// [tasm]: https://triton-vm.org/spec/instructions.html
#[macro_export]
macro_rules! triton_asm {
    (@fmt $fmt:expr, $($args:expr,)*; ) => {
        format_args!($fmt $(,$args)*).to_string()
    };
    (@fmt $fmt:expr, $($args:expr,)*;
        hint $var:ident: $ty:ident = stack[$start:literal..$end:literal] $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " hint {}: {} = stack[{}..{}] "),
            $($args,)* stringify!($var), stringify!($ty), $start, $end,;
            $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*;
        hint $var:ident = stack[$start:literal..$end:literal] $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " hint {} = stack[{}..{}] "),
            $($args,)* stringify!($var), $start, $end,;
            $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*;
        hint $var:ident: $ty:ident = stack[$index:literal] $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " hint {}: {} = stack[{}] "),
            $($args,)* stringify!($var), stringify!($ty), $index,;
            $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*;
        hint $var:ident = stack[$index:literal] $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " hint {} = stack[{}] "),
            $($args,)* stringify!($var), $index,;
            $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*; $label_declaration:ident: $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " ", stringify!($label_declaration), ": "), $($args,)*; $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*; $instruction:ident $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " ", stringify!($instruction), " "), $($args,)*; $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*; $instruction_argument:literal $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " ", stringify!($instruction_argument), " "), $($args,)*; $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*; {$label_declaration:expr}: $($tail:tt)*) => {
        $crate::triton_asm!(@fmt concat!($fmt, "{}: "), $($args,)* $label_declaration,; $($tail)*)
    };
    (@fmt $fmt:expr, $($args:expr,)*; {&$instruction_list:expr} $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, "{} "), $($args,)*
            $instruction_list.iter().map(|instr| instr.to_string()).collect::<Vec<_>>().join(" "),;
            $($tail)*
        )
    };
    (@fmt $fmt:expr, $($args:expr,)*; {$expression:expr} $($tail:tt)*) => {
        $crate::triton_asm!(@fmt concat!($fmt, "{} "), $($args,)* $expression,; $($tail)*)
    };

    // repeated instructions
    [pop $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(pop $arg); $num ] };
    [push $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(push $arg); $num ] };
    [divine $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(divine $arg); $num ] };
    [pick $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(pick $arg); $num ] };
    [place $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(place $arg); $num ] };
    [dup $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(dup $arg); $num ] };
    [swap $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(swap $arg); $num ] };
    [call $arg:ident; $num:expr] => { vec![ $crate::triton_instr!(call $arg); $num ] };
    [read_mem $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(read_mem $arg); $num ] };
    [write_mem $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(write_mem $arg); $num ] };
    [read_io $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(read_io $arg); $num ] };
    [write_io $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(write_io $arg); $num ] };
    [$instr:ident; $num:expr] => { vec![ $crate::triton_instr!($instr); $num ] };

    // entry point
    {$($source_code:tt)*} => {{
        let source_code = $crate::triton_asm!(@fmt "",; $($source_code)*);
        let (_, instructions) = $crate::parser::tokenize(&source_code).unwrap();
        $crate::parser::to_labelled_instructions(&instructions)
    }};
}

/// Compile a single [Triton assembly][tasm] instruction into a
/// [`LabelledInstruction`](instruction::LabelledInstruction).
///
/// # Examples
///
/// ```
/// # use triton_isa::triton_instr;
/// # use triton_isa::instruction::LabelledInstruction;
/// # use triton_isa::instruction::AnInstruction::Call;
/// let instruction = triton_instr!(call my_label);
/// assert_eq!(LabelledInstruction::Instruction(Call("my_label".to_string())), instruction);
/// ```
///
/// [tasm]: https://triton-vm.org/spec/instructions.html
#[macro_export]
macro_rules! triton_instr {
    (pop $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Pop(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (push $arg:expr) => {{
        let argument = $crate::twenty_first::prelude::BFieldElement::from($arg);
        let instruction = $crate::instruction::AnInstruction::<String>::Push(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (divine $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Divine(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (pick $arg:literal) => {{
        let argument = $crate::op_stack::OpStackElement::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Pick(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (place $arg:literal) => {{
        let argument = $crate::op_stack::OpStackElement::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Place(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (dup $arg:literal) => {{
        let argument = $crate::op_stack::OpStackElement::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Dup(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (swap $arg:literal) => {{
        let argument = $crate::op_stack::OpStackElement::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Swap(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (call $arg:ident) => {{
        let argument = stringify!($arg).to_string();
        let instruction = $crate::instruction::AnInstruction::<String>::Call(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (read_mem $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::ReadMem(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (write_mem $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::WriteMem(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (addi $arg:expr) => {{
        let argument = $crate::twenty_first::prelude::BFieldElement::from($arg);
        let instruction = $crate::instruction::AnInstruction::<String>::AddI(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (read_io $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::ReadIo(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (write_io $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::WriteIo(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    ($instr:ident) => {{
        let (_, instructions) = $crate::parser::tokenize(stringify!($instr)).unwrap();
        instructions[0].to_labelled_instruction()
    }};
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn public_types_implement_usual_auto_traits() {
        fn implements_auto_traits<T: Sized + Send + Sync + Unpin>() {}

        implements_auto_traits::<error::AssertionError>();
        implements_auto_traits::<error::InstructionError>();
        implements_auto_traits::<error::NumberOfWordsError>();
        implements_auto_traits::<error::OpStackElementError>();
        implements_auto_traits::<error::OpStackError>();
        implements_auto_traits::<error::ParseError>();
        implements_auto_traits::<error::ProgramDecodingError>();

        implements_auto_traits::<instruction::Instruction>();
        implements_auto_traits::<instruction::AnInstruction<usize>>();
        implements_auto_traits::<instruction::InstructionBit>();
        implements_auto_traits::<instruction::TypeHint>();

        implements_auto_traits::<op_stack::NumberOfWords>();
        implements_auto_traits::<op_stack::OpStack>();
        implements_auto_traits::<op_stack::OpStackElement>();
        implements_auto_traits::<op_stack::UnderflowIO>();

        implements_auto_traits::<parser::InstructionToken>();

        implements_auto_traits::<program::InstructionIter>();
        implements_auto_traits::<program::Program>();
    }
}
