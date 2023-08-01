//! Triton Virtual Machine is a Zero-Knowledge Proof System (ZKPS) for proving correct execution
//! of programs written in Triton assembly. The proof system is a zk-STARK, which is a
//! state-of-the-art ZKPS.

#![recursion_limit = "4096"]

use anyhow::bail;
use anyhow::Result;
pub use twenty_first::shared_math::b_field_element::BFieldElement;
pub use twenty_first::shared_math::tip5::Digest;

use crate::program::Program;
pub use crate::proof::Claim;
pub use crate::proof::Proof;
use crate::stark::Stark;
use crate::stark::StarkHasher;
pub use crate::stark::StarkParameters;

pub mod aet;
pub mod arithmetic_domain;
pub mod error;
pub mod example_programs;
pub mod fri;
pub mod instruction;
pub mod op_stack;
pub mod parser;
pub mod profiler;
pub mod program;
pub mod proof;
pub mod proof_item;
pub mod proof_stream;
mod shared_tests;
pub mod stark;
pub mod table;
pub mod vm;

/// Compile an entire program written in [Triton assembly][tasm].
/// The resulting [`Program`](crate::program::Program) can be
/// [run](crate::program::Program::run).
///
/// It is possible to use string-like interpolation to insert instructions, arguments, labels,
/// or other substrings into the program.
///
/// # Examples
///
/// ```
/// # use triton_vm::triton_program;
/// let program = triton_program!(
///     read_io push 5 mul
///     call check_eq_15
///     push 1 write_io
///     halt
///     // assert that the top of the stack is 15
///     check_eq_15:
///         push 15 eq assert
///         return
/// );
/// let output = program.run(vec![3_u64.into()], vec![]).unwrap();
/// assert_eq!(1, output[0].value());
/// ```
///
/// Any type with an appropriate [`Display`](std::fmt::Display) implementation can be
/// interpolated. This includes, for example, primitive types like `u64` and `&str`, but also
/// [`Instruction`](crate::instruction::Instruction)s,
/// [`BFieldElement`](crate::BFieldElement)s, and
/// [`Label`](crate::instruction::LabelledInstruction)s, among others.
///
/// ```
/// # use triton_vm::triton_program;
/// # use triton_vm::BFieldElement;
/// # use triton_vm::instruction::Instruction;
/// let element_0 = BFieldElement::new(0);
/// let label = "my_label";
/// let instruction_push = Instruction::Push(42_u64.into());
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
/// [from_code]: crate::program::Program::from_code
#[macro_export]
macro_rules! triton_program {
    {$($source_code:tt)*} => {{
        let labelled_instructions = $crate::triton_asm!($($source_code)*);
        $crate::program::Program::new(&labelled_instructions)
    }};
}

/// Compile [Triton assembly][tasm] into a list of labelled
/// [`Instruction`](crate::instruction::LabelledInstruction)s.
/// Similar to [`triton_program!`](crate::triton_program), it is possible to use string-like
/// interpolation to insert instructions, arguments, labels, or other expressions.
///
/// Similar to [`vec!], a single instruction can be repeated a specified number of times.
///
/// Furthermore, a list of LabelledInstructions can be inserted like so: `[&list]`.
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
/// # use triton_vm::instruction::AnInstruction::ReadIo;
/// let instructions = triton_asm![read_io; 3];
/// assert_eq!(3, instructions.len());
/// assert_eq!(LabelledInstruction::Instruction(ReadIo), instructions[0]);
/// assert_eq!(LabelledInstruction::Instruction(ReadIo), instructions[1]);
/// assert_eq!(LabelledInstruction::Instruction(ReadIo), instructions[2]);
/// ```
///
/// Inserting substring of labelled instructions:
///
/// ```
/// let insert_me = triton_asm!(
///     pop
///     pop
/// );
/// let wrapper = triton_asm!(
///     push 0
///     {&insert_me}
///     push 1
/// );
/// assert_eq!(LabelledInstruction::Instruction(Push(BFieldElement::new(0))), wrapper[0]);
/// assert_eq!(LabelledInstruction::Instruction(Pop), wrapper[1]);
/// assert_eq!(LabelledInstruction::Instruction(Pop), wrapper[2]);
/// assert_eq!(LabelledInstruction::Instruction(Push(BFieldElement::new(1))), wrapper[3]);
///```
///
/// # Panics
///
/// **Panics** if the instructions cannot be parsed.
/// For examples, see [`triton_program!`](crate::triton_program), with the exception that
/// labels are not checked for existence or uniqueness.
///
/// [tasm]: https://triton-vm.org/spec/instructions.html
#[macro_export]
macro_rules! triton_asm {
    // we have reached the end; quit parsing
    (@fmt $fmt:expr, $($args:expr,)*; ) => {
        format_args!($fmt $(,$args)*).to_string()
    };
    // label (e.g., "some_function:")
    (@fmt $fmt:expr, $($args:expr,)*; $instr:ident: $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " ", stringify!($instr), ": "), $($args,)*; $($tail)*
        )
    };
    // instruction
    (@fmt $fmt:expr, $($args:expr,)*; $instr:ident $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " ", stringify!($instr), " "), $($args,)*; $($tail)*
        )
    };
    // instruction argument
    (@fmt $fmt:expr, $($args:expr,)*; $arg:literal $($tail:tt)*) => {
        $crate::triton_asm!(@fmt
            concat!($fmt, " ", stringify!($arg), " "), $($args,)*; $($tail)*
        )
    };
    // an expression wrapped in curly braces followed by colon (e.g., "{entrypoint}:")
    (@fmt $fmt:expr, $($args:expr,)*; {$e:expr}: $($tail:tt)*) => {
        $crate::triton_asm!(@fmt concat!($fmt, "{}: "), $($args,)* $e,; $($tail)*)
    };
    // a reference expression wrapped in curly braces (e.g., "{&instructions_list}")
    (@fmt $fmt:expr, $($args:expr,)*; {&$e:expr} $($tail:tt)*) => {
        $crate::triton_asm!(@fmt concat!($fmt, "{} "), $($args,)* $crate::instruction::LabelledInstructions($e).to_string(),; $($tail)*)
    };
    // an expression wrapped in curly braces (e.g., "{entrypoint}")
    (@fmt $fmt:expr, $($args:expr,)*; {$e:expr} $($tail:tt)*) => {
        $crate::triton_asm!(@fmt concat!($fmt, "{} "), $($args,)* $e,; $($tail)*)
    };

    // repeated instruction, with or without argument
    [push $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(push $arg); $num ] };
    [dup $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(dup $arg); $num ] };
    [swap $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(swap $arg); $num ] };
    [call $arg:ident; $num:expr] => { vec![ $crate::triton_instr!(call $arg); $num ] };
    [$instr:ident; $num:expr] => { vec![ $crate::triton_instr!($instr); $num ] };

    // entry point for macro
    {$($source_code:tt)*} => {{
        let source_code = $crate::triton_asm!(@fmt "",; $($source_code)*);
        let (_, instructions) = $crate::parser::tokenize(&source_code).unwrap();
        $crate::parser::to_labelled_instructions(&instructions)
    }};
}

/// Compile a single [Triton assembly][tasm] instruction into a [`LabelledInstruction`].
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
    (push $arg:expr) => {{
        let argument = $crate::BFieldElement::new($arg);
        let instruction = $crate::instruction::AnInstruction::<String>::Push(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (dup $arg:literal) => {{
        let argument: $crate::op_stack::OpStackElement = u32::try_into($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Dup(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (swap $arg:literal) => {{
        assert_ne!(0_u32, $arg, "`swap 0` is illegal.");
        let argument: $crate::op_stack::OpStackElement = u32::try_into($arg).unwrap();
        let instruction = $crate::instruction::AnInstruction::<String>::Swap(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    (call $arg:ident) => {{
        let argument = stringify!($arg).to_string();
        let instruction = $crate::instruction::AnInstruction::<String>::Call(argument);
        $crate::instruction::LabelledInstruction::Instruction(instruction)
    }};
    ($instr:ident) => {{
        let (_, instructions) = $crate::parser::tokenize(stringify!($instr)).unwrap();
        instructions[0].to_labelled_instruction()
    }};
}

/// Like [`assert_eq!`], but returns a [`Result`] instead of panicking.
/// Can only be used in functions that return a [`Result`].
/// Thin wrapper around [`anyhow::ensure!`].
macro_rules! ensure_eq {
    ($left:expr, $right:expr) => {{
        anyhow::ensure!(
            $left == $right,
            "Expected `{}` to equal `{}`.\nleft: {:?}\nright: {:?}\n",
            stringify!($left),
            stringify!($right),
            $left,
            $right,
        )
    }};
}
pub(crate) use ensure_eq;

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
    public_input: &[u64],
    secret_input: &[u64],
) -> Result<(StarkParameters, Claim, Proof)> {
    let canonical_representation_error =
        "input must contain only elements in canonical representation, i.e., \
        elements smaller than the prime field's modulus 2^64 - 2^32 + 1.";
    if public_input.iter().any(|&e| e > BFieldElement::MAX) {
        bail!("Public {canonical_representation_error})");
    }
    if secret_input.iter().any(|&e| e > BFieldElement::MAX) {
        bail!("Secret {canonical_representation_error}");
    }

    // Convert the public and secret inputs to BFieldElements.
    let public_input = public_input
        .iter()
        .map(|&e| BFieldElement::new(e))
        .collect::<Vec<_>>();
    let secret_input = secret_input
        .iter()
        .map(|&e| BFieldElement::new(e))
        .collect::<Vec<_>>();

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
    let (aet, public_output) = program.trace_execution(public_input.clone(), secret_input)?;

    // Hash the program to obtain its digest.
    let program_digest = program.hash::<StarkHasher>();

    // The default parameters give a (conjectured) security level of 160 bits.
    let parameters = StarkParameters::default();

    // Set up the claim that is to be proven. The claim contains all public information. The
    // proof is zero-knowledge with respect to everything else.
    let claim = Claim {
        program_digest,
        input: public_input,
        output: public_output,
    };

    // Generate the proof.
    let proof = Stark::prove(&parameters, &claim, &aet, &mut None);

    Ok((parameters, claim, proof))
}

/// A convenience function for proving a [`Claim`] and the program that claim corresponds to.
/// Method [`prove_program`] gives a simpler interface with less control.
pub fn prove(
    parameters: &StarkParameters,
    claim: &Claim,
    program: &Program,
    secret_input: &[BFieldElement],
) -> Result<Proof> {
    let program_digest = program.hash::<StarkHasher>();
    ensure_eq!(program_digest, claim.program_digest);
    let (aet, public_output) =
        program.trace_execution(claim.input.clone(), secret_input.to_vec())?;
    ensure_eq!(public_output, claim.output);
    let proof = Stark::prove(parameters, claim, &aet, &mut None);
    Ok(proof)
}

/// Verify a proof generated by [`prove`] or [`prove_program`].
#[must_use]
pub fn verify(parameters: &StarkParameters, claim: &Claim, proof: &Proof) -> bool {
    Stark::verify(parameters, claim, proof, &mut None).unwrap_or(false)
}

#[cfg(test)]
mod public_interface_tests {
    use crate::shared_tests::create_proofs_directory;
    use crate::shared_tests::load_proof;
    use crate::shared_tests::proof_file_exists;
    use crate::shared_tests::save_proof;
    use crate::stark::StarkHasher;

    use super::*;

    #[test]
    pub fn lockscript_test() {
        // Program proves the knowledge of a hash preimage
        let program = triton_program!(
            divine divine divine divine divine
            hash pop pop pop pop pop
            push 09456474867485907852
            push 12765666850723567758
            push 08551752384389703074
            push 03612858832443241113
            push 12064501419749299924
            assert_vector
            read_io read_io read_io read_io read_io
            halt
        );

        let secret_input = vec![
            7534225252725590272,
            10242377928140984092,
            4934077665495234419,
            1344204945079929819,
            2308095244057597075,
        ];
        let public_input = vec![
            4541691341642414223,
            488727826369776966,
            18398227966153280881,
            6431838875748878863,
            17174585125955027015,
        ];

        let (parameters, claim, proof) =
            prove_program(&program, &public_input, &secret_input).unwrap();
        assert_eq!(
            StarkParameters::default(),
            parameters,
            "Prover must return default STARK parameters"
        );
        let expected_program_digest = program.hash::<StarkHasher>();
        assert_eq!(
            expected_program_digest, claim.program_digest,
            "program digest must match program"
        );
        assert_eq!(
            public_input,
            claim.public_input(),
            "Claimed input must match supplied input"
        );
        assert!(
            claim.output.is_empty(),
            "Output must be empty for program that doesn't write to output"
        );
        let verdict = verify(&parameters, &claim, &proof);
        assert!(verdict);
    }

    #[test]
    fn lib_prove_verify() {
        let parameters = StarkParameters::default();
        let program = triton_program!(push 1 assert halt);
        let claim = Claim {
            program_digest: program.hash::<StarkHasher>(),
            input: vec![],
            output: vec![],
        };

        let proof = prove(&parameters, &claim, &program, &[]).unwrap();
        let verdict = verify(&parameters, &claim, &proof);
        assert!(verdict);
    }

    #[test]
    fn save_proof_to_and_load_from_disk_test() {
        let filename = "nop_halt.tsp";
        if !proof_file_exists(filename) {
            create_proofs_directory().unwrap();
        }

        let program = triton_program!(nop halt);
        let (_, _, proof) = prove_program(&program, &[], &[]).unwrap();

        save_proof(filename, proof.clone()).unwrap();
        let loaded_proof = load_proof(filename).unwrap();

        assert_eq!(proof, loaded_proof);
    }

    /// Invocations of the `ensure_eq!` macro for testing purposes must be wrapped in their own
    /// function due to the return type requirements, which _must_ be
    /// - `Result<_>` for any method invoking the `ensure_eq!` macro, and
    /// - `()` for any method annotated with `#[test]`.
    fn method_with_failing_ensure_eq_macro() -> Result<()> {
        ensure_eq!("a", "a");
        let left_hand_side = 2;
        let right_hand_side = 1;
        ensure_eq!(left_hand_side, right_hand_side);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Expected `left_hand_side` to equal `right_hand_side`.")]
    fn ensure_eq_macro() {
        method_with_failing_ensure_eq_macro().unwrap()
    }
}
