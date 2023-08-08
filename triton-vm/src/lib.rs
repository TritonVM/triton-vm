//! Triton Virtual Machine is a Zero-Knowledge Proof System (ZKPS) for proving correct execution
//! of programs written in Triton assembly. The proof system is a zk-STARK, which is a
//! state-of-the-art ZKPS.

#![recursion_limit = "4096"]

use anyhow::bail;
use anyhow::Result;
pub use twenty_first::shared_math::b_field_element::BFieldElement;
pub use twenty_first::shared_math::tip5::Digest;

pub use crate::program::NonDeterminism;
pub use crate::program::Program;
pub use crate::program::PublicInput;
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
/// let output = program.run(vec![3].into(), [].into()).unwrap();
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
/// Similar to [`vec!`], a single instruction can be repeated a specified number of times.
///
/// Furthermore, a list of [`LabelledInstruction`](crate::instruction::LabelledInstruction)s
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
/// # use triton_vm::BFieldElement;
/// # use triton_vm::triton_asm;
/// # use triton_vm::instruction::LabelledInstruction;
/// # use triton_vm::instruction::AnInstruction::Push;
/// # use triton_vm::instruction::AnInstruction::Pop;
/// let insert_me = triton_asm!(
///     pop
///     pop
/// );
/// let surrounding_code = triton_asm!(
///     push 0
///     {&insert_me}
///     push 1
/// );
/// # let zero = BFieldElement::new(0);
/// # assert_eq!(LabelledInstruction::Instruction(Push(zero)), surrounding_code[0]);
/// assert_eq!(LabelledInstruction::Instruction(Pop), surrounding_code[1]);
/// assert_eq!(LabelledInstruction::Instruction(Pop), surrounding_code[2]);
/// # let one = BFieldElement::new(1);
/// # assert_eq!(LabelledInstruction::Instruction(Push(one)), surrounding_code[3]);
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
    (@fmt $fmt:expr, $($args:expr,)*; ) => {
        format_args!($fmt $(,$args)*).to_string()
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
    [push $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(push $arg); $num ] };
    [dup $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(dup $arg); $num ] };
    [swap $arg:literal; $num:expr] => { vec![ $crate::triton_instr!(swap $arg); $num ] };
    [call $arg:ident; $num:expr] => { vec![ $crate::triton_instr!(call $arg); $num ] };
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
    non_determinism: &NonDeterminism<u64>,
) -> Result<(StarkParameters, Claim, Proof)> {
    input_elements_have_unique_representation(public_input, non_determinism)?;

    // Convert public and secret inputs to BFieldElements.
    let public_input: PublicInput = public_input.to_owned().into();
    let non_determinism = non_determinism.into();

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

    // Hash the program to obtain its digest.
    let program_digest = program.hash::<StarkHasher>();

    // The default parameters give a (conjectured) security level of 160 bits.
    let parameters = StarkParameters::default();

    // Set up the claim that is to be proven. The claim contains all public information. The
    // proof is zero-knowledge with respect to everything else.
    let claim = Claim {
        program_digest,
        input: public_input.individual_tokens,
        output: public_output,
    };

    // Generate the proof.
    let proof = Stark::prove(&parameters, &claim, &aet, &mut None);

    Ok((parameters, claim, proof))
}

fn input_elements_have_unique_representation(
    public_input: &[u64],
    non_determinism: &NonDeterminism<u64>,
) -> Result<()> {
    let max = BFieldElement::MAX;
    let canonical_representation_error =
        "must contain only elements in canonical representation, i.e., \
        elements smaller than the prime field's modulus 2^64 - 2^32 + 1.";
    if public_input.iter().any(|&e| e > max) {
        bail!("Public input {canonical_representation_error})");
    }
    if non_determinism.individual_tokens.iter().any(|&e| e > max) {
        bail!("Secret input {canonical_representation_error}");
    }
    if non_determinism.ram.keys().any(|&e| e > max) {
        bail!("RAM addresses {canonical_representation_error}");
    }
    if non_determinism.ram.values().any(|&e| e > max) {
        bail!("RAM values {canonical_representation_error}");
    }
    Ok(())
}

/// A convenience function for proving a [`Claim`] and the program that claim corresponds to.
/// Method [`prove_program`] gives a simpler interface with less control.
pub fn prove(
    parameters: &StarkParameters,
    claim: &Claim,
    program: &Program,
    non_determinism: NonDeterminism<BFieldElement>,
) -> Result<Proof> {
    let program_digest = program.hash::<StarkHasher>();
    ensure_eq!(program_digest, claim.program_digest);
    let (aet, public_output) = program.trace_execution((&claim.input).into(), non_determinism)?;
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
    use rand::thread_rng;
    use rand::Rng;

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

        let non_determinism = NonDeterminism::new(secret_input);
        let (parameters, claim, proof) =
            prove_program(&program, &public_input, &non_determinism).unwrap();
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
    fn lib_use_initial_ram() {
        let program = triton_program!(
            push 51 read_mem
            push 42 read_mem
            swap 1 swap 2 mul
            write_io halt
        );

        let initial_ram = [(42, 17), (51, 13)].into();
        let non_determinism = NonDeterminism::new(vec![]).with_ram(initial_ram);
        let (parameters, claim, proof) = prove_program(&program, &[], &non_determinism).unwrap();
        assert_eq!(13 * 17, claim.output[0].value());

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

        let proof = prove(&parameters, &claim, &program, [].into()).unwrap();
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
        let (_, _, proof) = prove_program(&program, &[], &[].into()).unwrap();

        save_proof(filename, proof.clone()).unwrap();
        let loaded_proof = load_proof(filename).unwrap();

        assert_eq!(proof, loaded_proof);
    }

    #[test]
    fn canonical_representation_failures() {
        let valid_public_input = thread_rng()
            .gen::<[BFieldElement; 10]>()
            .map(|bfe| bfe.value());
        let invalid_public_input = [thread_rng().gen_range(BFieldElement::MAX..=u64::MAX)];

        let valid_secret_input = thread_rng()
            .gen::<[BFieldElement; 10]>()
            .map(|bfe| bfe.value());
        let invalid_secret_input = [thread_rng().gen_range(BFieldElement::MAX..=u64::MAX)];

        let valid_initial_ram = thread_rng()
            .gen::<[(BFieldElement, BFieldElement); 10]>()
            .map(|(key, val)| (key.value(), val.value()));
        let invalid_key_initial_ram = [(
            thread_rng().gen_range(BFieldElement::MAX..=u64::MAX),
            thread_rng().gen::<BFieldElement>().value(),
        )];
        let invalid_val_initial_ram = [(
            thread_rng().gen::<BFieldElement>().value(),
            thread_rng().gen_range(BFieldElement::MAX..=u64::MAX),
        )];

        let valid_non_determinism =
            NonDeterminism::new(valid_secret_input.into()).with_ram(valid_initial_ram.into());
        let invalid_secret_input_non_determinism =
            NonDeterminism::new(invalid_secret_input.into()).with_ram(valid_initial_ram.into());
        let invalid_key_initial_ram_non_determinism =
            NonDeterminism::new(valid_secret_input.into()).with_ram(invalid_key_initial_ram.into());
        let invalid_val_initial_ram_non_determinism =
            NonDeterminism::new(valid_secret_input.into()).with_ram(invalid_val_initial_ram.into());

        let public_input_error = input_elements_have_unique_representation(
            &invalid_public_input,
            &valid_non_determinism,
        )
        .unwrap_err();
        assert!(public_input_error.to_string().contains("Public input"));

        let secret_input_error = input_elements_have_unique_representation(
            &valid_public_input,
            &invalid_secret_input_non_determinism,
        )
        .unwrap_err();
        assert!(secret_input_error.to_string().contains("Secret input"));

        let initial_ram_key_error = input_elements_have_unique_representation(
            &valid_public_input,
            &invalid_key_initial_ram_non_determinism,
        )
        .unwrap_err();
        assert!(initial_ram_key_error.to_string().contains("RAM addresses"));

        let initial_ram_val_error = input_elements_have_unique_representation(
            &valid_public_input,
            &invalid_val_initial_ram_non_determinism,
        )
        .unwrap_err();
        assert!(initial_ram_val_error.to_string().contains("RAM values"));
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

    #[test]
    fn nested_triton_asm_interpolation() {
        let triple_write = triton_asm![write_io; 3];
        let snippet_0 = triton_asm!(push 7 nop call my_label);
        let snippet_1 = triton_asm!(pop pop halt my_label: push 8 {&triple_write});
        let source_code = triton_asm!(push 6 {&snippet_0} {&snippet_1} halt);

        let program = triton_program!({ &source_code });
        let public_output = program.run([].into(), [].into()).unwrap();

        let expected_output = [8, 7, 6].map(BFieldElement::new).to_vec();
        assert_eq!(expected_output, public_output);
    }
}
