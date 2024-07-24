//! Triton Virtual Machine is a Zero-Knowledge Proof System (ZKPS) for proving
//! correct execution of programs written in Triton assembly. The proof system
//! is a [zk-STARK](Stark), which is a state-of-the-art ZKPS.
//!
//! Generally, all arithmetic performed by Triton VM happens in the prime field with
//! 2^64 - 2^32 + 1 elements. Instructions for u32 operations are provided.
//!
//! For a full overview over all available instructions and their effects, see the
//! [specification](https://triton-vm.org/spec/instructions.html).
//!
//! [Triton VM's STARK](Stark) is parametric, but it is highly recommended to
//! use the provided [default](Stark::default). Furthermore, certain runtime
//! characteristics are [configurable](config), and usually don't need changing.
//!
//! # Non-Determinism
//!
//! Triton VM is a non-deterministic machine. That is,
//! 1. Triton VM's random access memory can be initialized arbitrarily, and
//! 1. for a select few instructions (namely `divine` and `merkle_step`),
//!   correct state transition is not fully determined by the current state and
//!   Triton VM's public input.
//!
//! The input for those non-deterministic instructions use dedicated input
//! streams. Those, together with the initial RAM, are collectively called
//! [`NonDeterminism`].
//!
//! # Examples
//!
//! Below are a few examples on how to use Triton VM. They show the instruction
//! set architecture in action and highlight the core methods required to
//! generate & verify a proof of correct execution. Some of these are
//! convenience function [`prove_program()`] as well as the [`prove()`] and
//! [`verify()`] methods.
//!
//! ## Factorial
//!
//! Compute the factorial of the given public input.
//!
//! The execution of the factorial program is already fully determined by the public input.
//! Hence, in this case, there is no need for specifying non-determinism.
//! Keep reading for an example that does use non-determinism.
//!
//! The [`triton_program!`] macro is used to conveniently write Triton assembly. In below example,
//! the state of the operational stack is shown as a comment after most instructions.
//!
//! ```
//! # use triton_vm::prelude::*;
//! let factorial_program = triton_program!(
//!     read_io 1           // n
//!     push 1              // n 1
//!     call factorial      // 0 n!
//!     write_io 1          // 0
//!     halt
//!
//!     factorial:          // n acc
//!         // if n == 0: return
//!         dup 1           // n acc n
//!         push 0 eq       // n acc n==0
//!         skiz            // n acc
//!             return      // 0 acc
//!         // else: multiply accumulator with n and recurse
//!         dup 1           // n acc n
//!         mul             // n acc·n
//!         swap 1          // acc·n n
//!         push -1 add     // acc·n n-1
//!         swap 1          // n-1 acc·n
//!         recurse
//! );
//! let public_input = PublicInput::new(bfe_vec![10]);
//! let non_determinism = NonDeterminism::default();
//!
//! let (stark, claim, proof) =
//!     triton_vm::prove_program(&factorial_program, public_input, non_determinism).unwrap();
//!
//! let verdict = triton_vm::verify(stark, &claim, &proof);
//! assert!(verdict);
//!
//! assert_eq!(1, claim.output.len());
//! assert_eq!(3_628_800, claim.output[0].value());
//! ```
//!
//! ## Non-Determinism
//!
//! In the following example, a public field elements equality to the sum of some squared secret
//! elements is proven. For demonstration purposes, some of the secret elements come from secret in,
//! and some are read from RAM, which can be initialized arbitrarily.
//!
//! Note that the non-determinism is not required for proof verification, and does not appear in
//! the claim or the proof. It is only used for proof generation. This way, the verifier can be
//! convinced that the prover did indeed know some input that satisfies the claim, but learns
//! nothing beyond that fact.
//!
//! The third potential source of non-determinism is intended for verifying Merkle authentication
//! paths. It is not used in this example. See [`NonDeterminism`] for more information.
//!
//! ```
//! # use triton_vm::*;
//! # use triton_vm::prelude::*;
//! let sum_of_squares_program = triton_program!(
//!     read_io 1                       // n
//!     call sum_of_squares_secret_in   // n sum_1
//!     call sum_of_squares_ram         // n sum_1 sum_2
//!     add                             // n sum_1+sum_2
//!     eq                              // n==(sum_1+sum_2)
//!     assert                          // abort the VM if n!=(sum_1+sum_2)
//!     halt
//!
//!     sum_of_squares_secret_in:
//!         divine 1 dup 0 mul          // s₁²
//!         divine 1 dup 0 mul add      // s₁²+s₂²
//!         divine 1 dup 0 mul add      // s₁²+s₂²+s₃²
//!         return
//!
//!     sum_of_squares_ram:
//!         push 17                     // 18
//!         read_mem 1                  // s₄ 17
//!         pop 1                       // s₄
//!         dup 0 mul                   // s₄²
//!         push 42                     // s₄² 43
//!         read_mem 1                  // s₄² s₅ 42
//!         pop 1                       // s₄² s₅
//!         dup 0 mul                   // s₄² s₅²
//!         add                         // s₄²+s₅²
//!         return
//! );
//! let public_input = PublicInput::from([bfe!(597)]);
//! let secret_input = [5, 9, 11].map(|v| bfe!(v));
//! let initial_ram = [(17, 3), (42, 19)].map(|(address, v)| (bfe!(address), bfe!(v)));
//! let non_determinism = NonDeterminism::from(secret_input).with_ram(initial_ram);
//!
//! let (stark, claim, proof) =
//!    prove_program(&sum_of_squares_program, public_input, non_determinism).unwrap();
//!
//! let verdict = verify(stark, &claim, &proof);
//! assert!(verdict);
//! ```
//!
//! ## Crashing Triton VM
//!
//! Successful termination of a program is not guaranteed. For example, a program must execute
//! `halt` as its last instruction. Certain instructions, such as `assert`, `invert`, or the u32
//! instructions, can also cause the VM to crash. Upon crashing Triton VM, methods like
//! [`run`](Program::run) and [`trace_execution`][trace_execution] will return a
//! [`VMError`][vm_error]. This can be helpful for debugging.
//!
//! ```
//! # use triton_vm::*;
//! # use triton_vm::prelude::*;
//! let crashing_program = triton_program!(push 2 assert halt);
//! let vm_error = crashing_program.run([].into(), [].into()).unwrap_err();
//! assert!(matches!(vm_error.source, InstructionError::AssertionFailed));
//! // inspect the VM state
//! eprintln!("{vm_error}");
//! ```
//!
//! [vm_error]: error::VMError
//! [trace_execution]: Program::trace_execution

#![recursion_limit = "4096"]
#![warn(let_underscore_drop)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]

pub use twenty_first;

use crate::error::ProvingError;
use crate::prelude::*;

pub mod aet;
pub mod arithmetic_domain;
pub mod config;
pub mod error;
pub mod example_programs;
pub mod fri;
pub mod instruction;
mod ndarray_helper;
pub mod op_stack;
pub mod parser;
pub mod prelude;
pub mod profiler;
pub mod program;
pub mod proof;
pub mod proof_item;
pub mod proof_stream;
pub mod stark;
pub mod table;
pub mod vm;

#[cfg(test)]
mod shared_tests;

/// Compile an entire program written in [Triton assembly][tasm].
/// The resulting [`Program`](Program) can be [run](Program::run).
///
/// It is possible to use string-like interpolation to insert instructions, arguments, labels,
/// or other substrings into the program.
///
/// # Examples
///
/// ```
/// # use triton_vm::prelude::*;
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
/// let public_input = PublicInput::from([bfe!(3)]);
/// let secret_input = NonDeterminism::default();
/// let output = program.run(public_input, secret_input).unwrap();
/// assert_eq!(17, output[0].value());
/// ```
///
/// Any type with an appropriate [`Display`](std::fmt::Display) implementation can be
/// interpolated. This includes, for example, primitive types like `u64` and `&str`, but also
/// [`Instruction`](instruction::Instruction)s,
/// [`BFieldElement`](BFieldElement)s, and
/// [`Label`](instruction::LabelledInstruction)s, among others.
///
/// ```
/// # use triton_vm::prelude::*;
/// # use triton_vm::instruction::Instruction;
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
/// - invalid labels, _e.g._, using a reserved keyword or starting a label with a digit
///
/// For a version that returns a `Result`, see [`Program::from_code()`][from_code].
///
/// [tasm]: https://triton-vm.org/spec/instructions.html
/// [from_code]: Program::from_code
#[macro_export]
macro_rules! triton_program {
    {$($source_code:tt)*} => {{
        let labelled_instructions = $crate::triton_asm!($($source_code)*);
        $crate::program::Program::new(&labelled_instructions)
    }};
}

/// Compile [Triton assembly][tasm] into a list of labelled
/// [`Instruction`](instruction::LabelledInstruction)s.
/// Similar to [`triton_program!`](triton_program), it is possible to use string-like
/// interpolation to insert instructions, arguments, labels, or other expressions.
///
/// Similar to [`vec!`], a single instruction can be repeated a specified number of times.
///
/// Furthermore, a list of [`LabelledInstruction`](instruction::LabelledInstruction)s
/// can be inserted like so: `{&list}`.
///
/// The labels for instruction `call`, if any, are also parsed. Instruction `call` can refer to
/// a label defined later in the program, _i.e.,_ labels are not checked for existence or
/// uniqueness by this parser.
///
/// # Examples
///
/// ```
/// # use triton_vm::triton_asm;
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
/// # use triton_vm::triton_asm;
/// # use triton_vm::instruction::LabelledInstruction;
/// # use triton_vm::instruction::AnInstruction::SpongeAbsorb;
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
/// # use triton_vm::prelude::*;
/// # use triton_vm::instruction::AnInstruction::Push;
/// # use triton_vm::instruction::AnInstruction::Pop;
/// # use triton_vm::op_stack::NumberOfWords::N1;
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
///```
///
/// # Panics
///
/// **Panics** if the instructions cannot be parsed.
/// For examples, see [`triton_program!`](triton_program), with the exception that
/// labels are not checked for existence or uniqueness.
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
/// # use triton_vm::triton_instr;
/// # use triton_vm::instruction::LabelledInstruction;
/// # use triton_vm::instruction::AnInstruction::Call;
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
        let argument = $crate::prelude::BFieldElement::from($arg);
        let instruction = $crate::instruction::AnInstruction::<String>::Push(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (divine $arg:literal) => {{
        let argument = $crate::op_stack::NumberOfWords::try_from($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Divine(argument);
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

/// Prove correct execution of a program written in Triton assembly.
/// This is a convenience function, abstracting away the details of the STARK construction.
/// If you want to have more control over the STARK construction, this method can serve as a
/// reference for how to use Triton VM.
///
/// Note that all arithmetic is in the prime field with 2^64 - 2^32 + 1 elements. If the
/// provided public input or secret input contains elements larger than this, proof generation
/// will be aborted.
///
/// The program executed by Triton VM must terminate gracefully, i.e., with instruction `halt`.
/// If the program crashes, _e.g._, due to an out-of-bounds instruction pointer or a failing
/// `assert` instruction, proof generation will fail.
///
/// The default STARK parameters used by Triton VM give a (conjectured) security level of 160 bits.
pub fn prove_program(
    program: &Program,
    public_input: PublicInput,
    non_determinism: NonDeterminism,
) -> Result<(Stark, Claim, Proof), ProvingError> {
    // Generate
    // - the witness required for proof generation, i.e., the Algebraic Execution Trace (AET), and
    // - the (public) output of the program.
    //
    // Crashes in the VM can occur for many reasons. For example:
    // - due to failing `assert` instructions,
    // - due to an out-of-bounds instruction pointer,
    // - if the program does not terminate gracefully, _i.e._, with instruction `halt`,
    // - if any of the two inputs does not conform to the program,
    // - because of a bug in the program, among other things.
    // If the VM crashes, proof generation will fail.
    let (aet, public_output) = program.trace_execution(public_input.clone(), non_determinism)?;

    // Set up the claim that is to be proven. The claim contains all public information. The
    // proof is zero-knowledge with respect to everything else.
    //
    // While it is more convenient to construct a `Claim::about_program(&program)`, this API is
    // purposefully not used here to highlight that only a program's hash digest, not the full
    // program, is part of the claim.
    let claim = Claim {
        program_digest: program.hash(),
        input: public_input.individual_tokens,
        output: public_output,
    };

    // The default parameters give a (conjectured) security level of 160 bits.
    let stark = Stark::default();

    // Generate the proof.
    let proof = stark.prove(&claim, &aet)?;

    Ok((stark, claim, proof))
}

/// A convenience function for proving a [`Claim`] and the program that claim corresponds to.
/// Method [`prove_program`] gives a simpler interface with less control.
pub fn prove(
    stark: Stark,
    claim: &Claim,
    program: &Program,
    non_determinism: NonDeterminism,
) -> Result<Proof, ProvingError> {
    let program_digest = program.hash();
    if program_digest != claim.program_digest {
        return Err(ProvingError::ProgramDigestMismatch);
    }
    let (aet, public_output) = program.trace_execution((&claim.input).into(), non_determinism)?;
    if public_output != claim.output {
        return Err(ProvingError::PublicOutputMismatch);
    }

    stark.prove(claim, &aet)
}

/// Verify a proof generated by [`prove`] or [`prove_program`].
///
/// Use [`Stark::verify`] for more verbose verification failures.
#[must_use]
pub fn verify(stark: Stark, claim: &Claim, proof: &Proof) -> bool {
    stark.verify(claim, proof).is_ok()
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;
    use twenty_first::prelude::tip5::Tip5;
    use twenty_first::prelude::*;
    use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

    use crate::instruction::LabelledInstruction;
    use crate::instruction::TypeHint;

    use super::*;

    /// The compiler automatically adds any applicable auto trait (all of which are
    /// marker traits) to self-defined types. This implies that these trait bounds
    /// might vanish if the necessary pre-conditions are no longer met. That'd be a
    /// breaking API change!
    ///
    /// To prevent _accidental_ removal of auto trait implementations, this method
    /// tests for their presence. If you are re-designing any of the types below
    /// and a test fails as a result, that might be fine. You are now definitely
    /// aware of a consequence you might not have known about otherwise. (If you
    /// were already aware you know how subtle this stuff can be and are hopefully
    /// fine with reading this comment.)
    ///
    /// Inspired by “Rust for Rustaceans” by Jon Gjengset.
    fn implements_auto_traits<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn public_types_implement_usual_auto_traits() {
        // re-exports
        implements_auto_traits::<BFieldElement>();
        implements_auto_traits::<Digest>();
        implements_auto_traits::<Tip5>();
        implements_auto_traits::<XFieldElement>();

        // prelude
        implements_auto_traits::<LabelledInstruction>();
        implements_auto_traits::<NonDeterminism>();
        implements_auto_traits::<Program>();
        implements_auto_traits::<PublicInput>();
        implements_auto_traits::<Claim>();
        implements_auto_traits::<Proof>();
        implements_auto_traits::<Stark>();
        implements_auto_traits::<VMState>();

        // errors
        implements_auto_traits::<error::VMError>();
        implements_auto_traits::<error::InstructionError>();
        implements_auto_traits::<error::ArithmeticDomainError>();
        implements_auto_traits::<error::ProofStreamError>();
        implements_auto_traits::<error::FriSetupError>();
        implements_auto_traits::<error::FriProvingError>();
        implements_auto_traits::<error::FriValidationError>();
        implements_auto_traits::<error::ProgramDecodingError>();
        implements_auto_traits::<error::ProvingError>();
        implements_auto_traits::<error::VerificationError>();
        implements_auto_traits::<error::OpStackElementError>();
        implements_auto_traits::<error::NumberOfWordsError>();

        // table things
        implements_auto_traits::<table::cascade_table::CascadeTable>();
        implements_auto_traits::<table::cascade_table::ExtCascadeTable>();
        implements_auto_traits::<table::challenges::ChallengeId>();
        implements_auto_traits::<table::challenges::Challenges>();
        implements_auto_traits::<table::constraint_circuit::BinOp>();
        implements_auto_traits::<table::constraint_circuit::SingleRowIndicator>();
        implements_auto_traits::<table::constraint_circuit::DualRowIndicator>();
        implements_auto_traits::<table::cross_table_argument::PermArg>();
        implements_auto_traits::<table::cross_table_argument::EvalArg>();
        implements_auto_traits::<table::cross_table_argument::LookupArg>();
        implements_auto_traits::<table::cross_table_argument::GrandCrossTableArg>();
        implements_auto_traits::<table::degree_lowering_table::DegreeLoweringBaseTableColumn>();
        implements_auto_traits::<table::degree_lowering_table::DegreeLoweringExtTableColumn>();
        implements_auto_traits::<table::degree_lowering_table::DegreeLoweringTable>();
        implements_auto_traits::<table::hash_table::HashTable>();
        implements_auto_traits::<table::hash_table::ExtHashTable>();
        implements_auto_traits::<table::hash_table::HashTableMode>();
        implements_auto_traits::<table::jump_stack_table::JumpStackTable>();
        implements_auto_traits::<table::jump_stack_table::ExtJumpStackTable>();
        implements_auto_traits::<table::lookup_table::LookupTable>();
        implements_auto_traits::<table::lookup_table::ExtLookupTable>();
        implements_auto_traits::<table::master_table::TableId>();
        implements_auto_traits::<table::master_table::MasterBaseTable>();
        implements_auto_traits::<table::master_table::MasterExtTable>();
        implements_auto_traits::<table::op_stack_table::OpStackTable>();
        implements_auto_traits::<table::op_stack_table::ExtOpStackTable>();
        implements_auto_traits::<table::op_stack_table::OpStackTableEntry>();
        implements_auto_traits::<table::processor_table::ProcessorTable>();
        implements_auto_traits::<table::processor_table::ExtProcessorTable>();
        implements_auto_traits::<table::program_table::ProgramTable>();
        implements_auto_traits::<table::program_table::ExtProgramTable>();
        implements_auto_traits::<table::ram_table::RamTableCall>();
        implements_auto_traits::<table::ram_table::RamTable>();
        implements_auto_traits::<table::ram_table::ExtRamTable>();
        implements_auto_traits::<table::table_column::ProgramBaseTableColumn>();
        implements_auto_traits::<table::table_column::ProgramExtTableColumn>();
        implements_auto_traits::<table::table_column::ProcessorBaseTableColumn>();
        implements_auto_traits::<table::table_column::ProcessorExtTableColumn>();
        implements_auto_traits::<table::table_column::OpStackBaseTableColumn>();
        implements_auto_traits::<table::table_column::OpStackExtTableColumn>();
        implements_auto_traits::<table::table_column::RamBaseTableColumn>();
        implements_auto_traits::<table::table_column::RamExtTableColumn>();
        implements_auto_traits::<table::table_column::JumpStackBaseTableColumn>();
        implements_auto_traits::<table::table_column::JumpStackExtTableColumn>();
        implements_auto_traits::<table::table_column::HashBaseTableColumn>();
        implements_auto_traits::<table::table_column::HashExtTableColumn>();
        implements_auto_traits::<table::table_column::CascadeBaseTableColumn>();
        implements_auto_traits::<table::table_column::CascadeExtTableColumn>();
        implements_auto_traits::<table::table_column::LookupBaseTableColumn>();
        implements_auto_traits::<table::table_column::LookupExtTableColumn>();
        implements_auto_traits::<table::table_column::U32BaseTableColumn>();
        implements_auto_traits::<table::table_column::U32ExtTableColumn>();
        implements_auto_traits::<table::u32_table::U32TableEntry>();
        implements_auto_traits::<table::u32_table::U32Table>();
        implements_auto_traits::<table::u32_table::ExtU32Table>();
        implements_auto_traits::<table::TasmConstraintEvaluationMemoryLayout>();

        // other
        implements_auto_traits::<aet::AlgebraicExecutionTrace>();
        implements_auto_traits::<aet::TableHeight>();
        implements_auto_traits::<arithmetic_domain::ArithmeticDomain>();
        implements_auto_traits::<fri::Fri<Tip5>>();
        implements_auto_traits::<TypeHint>();
        implements_auto_traits::<instruction::AnInstruction<usize>>();
        implements_auto_traits::<instruction::InstructionBit>();
        implements_auto_traits::<op_stack::OpStack>();
        implements_auto_traits::<op_stack::UnderflowIO>();
        implements_auto_traits::<op_stack::OpStackElement>();
        implements_auto_traits::<op_stack::NumberOfWords>();
        implements_auto_traits::<parser::ParseError>();
        implements_auto_traits::<parser::InstructionToken>();
        implements_auto_traits::<profiler::VMPerformanceProfile>();
        implements_auto_traits::<program::ExecutionTraceProfile>();
        implements_auto_traits::<program::InstructionIter>();
        implements_auto_traits::<program::ProfileLine>();
        implements_auto_traits::<program::VMTableHeights>();
        implements_auto_traits::<proof_item::FriResponse>();
        implements_auto_traits::<proof_item::ProofItem>();
        implements_auto_traits::<proof_stream::ProofStream>();
        implements_auto_traits::<vm::CoProcessorCall>();
    }

    #[proptest]
    fn prove_verify_knowledge_of_hash_preimage(
        #[strategy(arb())] hash_preimage: Digest,
        #[strategy(arb())] some_tie_to_an_outer_context: Digest,
    ) {
        let hash_digest = Tip5::hash_pair(hash_preimage, Digest::default()).values();

        let program = triton_program! {
            divine 5
            hash
            push {hash_digest[4]}
            push {hash_digest[3]}
            push {hash_digest[2]}
            push {hash_digest[1]}
            push {hash_digest[0]}
            assert_vector
            read_io 5
            halt
        };

        let public_input = PublicInput::from(some_tie_to_an_outer_context.reversed().values());
        let non_determinism = NonDeterminism::new(hash_preimage.reversed().values());
        let maybe_proof = prove_program(&program, public_input.clone(), non_determinism);
        let (stark, claim, proof) =
            maybe_proof.map_err(|err| TestCaseError::Fail(err.to_string().into()))?;
        prop_assert_eq!(Stark::default(), stark);

        let verdict = verify(stark, &claim, &proof);
        prop_assert!(verdict);

        prop_assert!(claim.output.is_empty());
        let expected_program_digest = program.hash();
        prop_assert_eq!(expected_program_digest, claim.program_digest);
        prop_assert_eq!(public_input.individual_tokens, claim.input);
    }

    #[test]
    fn lib_use_initial_ram() {
        let program = triton_program!(
            push 51 read_mem 1 pop 1
            push 42 read_mem 1 pop 1
            mul
            write_io 1 halt
        );

        let public_input = PublicInput::default();
        let initial_ram = [(42, 17), (51, 13)].map(|(address, v)| (bfe!(address), bfe!(v)));
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);
        let (stark, claim, proof) = prove_program(&program, public_input, non_determinism).unwrap();
        assert!(13 * 17 == claim.output[0].value());

        let verdict = verify(stark, &claim, &proof);
        assert!(verdict);
    }

    #[test]
    fn lib_prove_verify() {
        let program = triton_program!(push 1 assert halt);
        let claim = Claim::about_program(&program);

        let stark = Stark::default();
        let proof = prove(stark, &claim, &program, [].into()).unwrap();
        let verdict = verify(stark, &claim, &proof);
        assert!(verdict);
    }

    #[test]
    fn lib_prove_with_incorrect_program_digest_gives_appropriate_error() {
        let program = triton_program!(push 1 assert halt);
        let other_program = triton_program!(push 2 assert halt);
        let claim = Claim::about_program(&other_program);

        let stark = Stark::default();
        let_assert!(Err(err) = prove(stark, &claim, &program, [].into()));
        assert!(let ProvingError::ProgramDigestMismatch = err);
    }

    #[test]
    fn lib_prove_with_incorrect_public_output_gives_appropriate_error() {
        let program = triton_program! { read_io 1 push 2 mul write_io 1 halt };
        let claim = Claim::about_program(&program)
            .with_input(bfe_vec![2])
            .with_output(bfe_vec![5]);

        let stark = Stark::default();
        let_assert!(Err(err) = prove(stark, &claim, &program, [].into()));
        assert!(let ProvingError::PublicOutputMismatch = err);
    }

    #[test]
    fn nested_triton_asm_interpolation() {
        let double_write = triton_asm![write_io 1; 2];
        let quadruple_write = triton_asm!({&double_write} write_io 2);
        let snippet_0 = triton_asm!(push 7 nop call my_label);
        let snippet_1 = triton_asm!(pop 2 halt my_label: push 8 push 9 {&quadruple_write});
        let source_code = triton_asm!(push 6 {&snippet_0} {&snippet_1} halt);

        let program = triton_program!({ &source_code });
        let public_output = program.run([].into(), [].into()).unwrap();

        let expected_output = bfe_vec![9, 8, 7, 6];
        assert_eq!(expected_output, public_output);
    }

    #[test]
    fn triton_asm_interpolation_of_many_pops() {
        let push_25 = triton_asm![push 0; 25];
        let pop_25 = triton_asm![pop 5; 5];
        let program = triton_program! { push 1 { &push_25 } { &pop_25 } assert halt };
        program.run([].into(), [].into()).unwrap();
    }

    #[test]
    #[should_panic(expected = "IndexOutOfBounds(0)")]
    fn parsing_pop_with_illegal_argument_fails() {
        triton_instr!(pop 0);
    }

    #[test]
    fn triton_asm_macro_can_parse_type_hints() {
        let instructions = triton_asm!(
            hint name_0: Type0  = stack[0..8]
            hint name_1         = stack[1..9]
            hint name_2: Type2  = stack[2]
            hint name_3         = stack[3]
        );

        assert!(4 == instructions.len());
        let_assert!(LabelledInstruction::TypeHint(type_hint_0) = instructions[0].clone());
        let_assert!(LabelledInstruction::TypeHint(type_hint_1) = instructions[1].clone());
        let_assert!(LabelledInstruction::TypeHint(type_hint_2) = instructions[2].clone());
        let_assert!(LabelledInstruction::TypeHint(type_hint_3) = instructions[3].clone());

        let expected_type_hint_0 = TypeHint {
            starting_index: 0,
            length: 8,
            type_name: Some("Type0".to_string()),
            variable_name: "name_0".to_string(),
        };
        let expected_type_hint_1 = TypeHint {
            starting_index: 1,
            length: 8,
            type_name: None,
            variable_name: "name_1".to_string(),
        };
        let expected_type_hint_2 = TypeHint {
            starting_index: 2,
            length: 1,
            type_name: Some("Type2".to_string()),
            variable_name: "name_2".to_string(),
        };
        let expected_type_hint_3 = TypeHint {
            starting_index: 3,
            length: 1,
            type_name: None,
            variable_name: "name_3".to_string(),
        };

        assert!(expected_type_hint_0 == type_hint_0);
        assert!(expected_type_hint_1 == type_hint_1);
        assert!(expected_type_hint_2 == type_hint_2);
        assert!(expected_type_hint_3 == type_hint_3);
    }

    #[test]
    fn triton_program_macro_can_parse_type_hints() {
        let program = triton_program! {
            push 3 hint loop_counter = stack[0]
            call my_loop
            pop 1
            halt

            my_loop:
                dup 0 push 0 eq
                hint return_condition: bool = stack[0]
                skiz return
                divine 3
                swap 3
                hint magic_number: XFE = stack[1..4]
                hint fizzled_magic = stack[5..8]
                recurse
        };

        let expected_type_hint_address_02 = TypeHint {
            starting_index: 0,
            length: 1,
            type_name: None,
            variable_name: "loop_counter".to_string(),
        };
        let expected_type_hint_address_12 = TypeHint {
            starting_index: 0,
            length: 1,
            type_name: Some("bool".to_string()),
            variable_name: "return_condition".to_string(),
        };
        let expected_type_hint_address_18_0 = TypeHint {
            starting_index: 1,
            length: 3,
            type_name: Some("XFE".to_string()),
            variable_name: "magic_number".to_string(),
        };
        let expected_type_hint_address_18_1 = TypeHint {
            starting_index: 5,
            length: 3,
            type_name: None,
            variable_name: "fizzled_magic".to_string(),
        };

        assert!(vec![expected_type_hint_address_02] == program.type_hints_at(2));

        assert!(vec![expected_type_hint_address_12] == program.type_hints_at(12));

        let expected_type_hints_address_18 = vec![
            expected_type_hint_address_18_0,
            expected_type_hint_address_18_1,
        ];
        assert!(expected_type_hints_address_18 == program.type_hints_at(18));
    }
}
