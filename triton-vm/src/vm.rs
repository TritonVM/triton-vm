use std::error::Error;
use std::fmt::Display;
use std::io::Cursor;

use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::instruction;
use crate::instruction::{parse, Instruction, LabelledInstruction};
use crate::state::{VMOutput, VMState};
use crate::stdio::{InputStream, OutputStream, VecStream};
use crate::table::base_matrix::AlgebraicExecutionTrace;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut stream = self.instructions.iter();
        while let Some(instruction) = stream.next() {
            writeln!(f, "{}", instruction)?;

            // Skip duplicate placeholder used for aligning instructions and instruction_pointer in VM.
            for _ in 1..instruction.size() {
                stream.next();
            }
        }
        Ok(())
    }
}

pub struct SkippyIter {
    cursor: Cursor<Vec<Instruction>>,
}

impl Iterator for SkippyIter {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.cursor.position() as usize;
        let instructions = self.cursor.get_ref();
        let instruction = *instructions.get(pos)?;
        self.cursor.set_position((pos + instruction.size()) as u64);

        Some(instruction)
    }
}

impl IntoIterator for Program {
    type Item = Instruction;

    type IntoIter = SkippyIter;

    fn into_iter(self) -> Self::IntoIter {
        let cursor = Cursor::new(self.instructions);
        SkippyIter { cursor }
    }
}

/// A `Program` is a `Vec<Instruction>` that contains duplicate elements for
/// instructions with a size of 2. This means that the index in the vector
/// corresponds to the VM's `instruction_pointer`. These duplicate values
/// should most often be skipped/ignored, e.g. when pretty-printing.
impl Program {
    /// Create a `Program` from a slice of `Instruction`.
    ///
    /// All valid programs terminate with `Halt`.
    ///
    /// `new()` will append `Halt` if not present.
    pub fn new(input: &[LabelledInstruction]) -> Self {
        let instructions = instruction::convert_labels(input)
            .iter()
            .flat_map(|instr| vec![*instr; instr.size()])
            .collect::<Vec<_>>();

        Program { instructions }
    }

    /// Create a `Program` by parsing source code.
    ///
    /// All valid programs terminate with `Halt`.
    ///
    /// `from_code()` will append `Halt` if not present.
    pub fn from_code(code: &str) -> Result<Self, Box<dyn Error>> {
        let instructions = parse(code)?;
        Ok(Program::new(&instructions))
    }

    /// Convert a `Program` to a `Vec<BFieldElement>`.
    ///
    /// Every single-word instruction is converted to a single word.
    ///
    /// Every double-word instruction is converted to two words.
    pub fn to_bwords(&self) -> Vec<BFieldElement> {
        self.clone()
            .into_iter()
            .map(|instruction| {
                let opcode = instruction.opcode_b();
                if let Some(arg) = instruction.arg() {
                    vec![opcode, arg]
                } else {
                    vec![opcode]
                }
            })
            .concat()
    }

    /// Simulate (execute) a `Program` and record every state transition. Returns an
    /// `AlgebraicExecutionTrace` recording every intermediate state of the processor and all co-
    /// processors.
    ///
    /// On premature termination of the VM, returns the `AlgebraicExecutionTrace` for the execution
    /// up to the point of failure.
    pub fn simulate<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
    ) -> (AlgebraicExecutionTrace, Option<Box<dyn Error>>)
    where
        In: InputStream,
        Out: OutputStream,
    {
        let mut aet = AlgebraicExecutionTrace::default();
        let mut state = VMState::new(self);
        // record initial state
        aet.processor_matrix.push(state.to_processor_row());

        while !state.is_complete() {
            let vm_output = match state.step_mut(stdin, secret_in) {
                Err(err) => return (aet, Some(err)),
                Ok(vm_output) => vm_output,
            };

            match vm_output {
                Some(VMOutput::XlixTrace(mut hash_trace)) => {
                    aet.hash_matrix.append(&mut hash_trace)
                }
                Some(VMOutput::WriteOutputSymbol(written_word)) => {
                    if let Err(error) = stdout.write_elem(written_word) {
                        return (aet, Some(Box::new(error)));
                    }
                }
                None => (),
            }
            // Record next, to be executed state. If `Halt`,
            aet.processor_matrix.push(state.to_processor_row());
        }

        (aet, None)
    }

    /// Wrapper around `.simulate()` for easier i/o handling. Behaviour is that of `.simulate()`,
    /// except the executed program's output is returned instead of modified in-place. Consequently,
    /// takes as arguments only `input` and `secret_input`.
    pub fn simulate_with_input(
        &self,
        input: &[BFieldElement],
        secret_input: &[BFieldElement],
    ) -> (
        AlgebraicExecutionTrace,
        Option<Box<dyn Error>>,
        Vec<BFieldElement>,
    ) {
        let mut stdin = VecStream::new(input);
        let mut secret_in = VecStream::new(secret_input);
        let mut stdout = VecStream::new(&[]);

        let (aet, vm_error) = self.simulate(&mut stdin, &mut secret_in, &mut stdout);
        (aet, vm_error, stdout.to_bword_vec())
    }

    pub fn run<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
    ) -> (Vec<VMState>, Option<Box<dyn Error>>)
    where
        In: InputStream,
        Out: OutputStream,
    {
        let mut states = vec![VMState::new(self)];
        let mut current_state = states.last().unwrap();

        while !current_state.is_complete() {
            let step = current_state.step(stdin, secret_in);
            let (next_state, vm_output) = match step {
                Err(err) => {
                    println!("Encountered an error when running VM.");
                    return (states, Some(err));
                }
                Ok((next_state, vm_output)) => (next_state, vm_output),
            };

            if let Some(VMOutput::WriteOutputSymbol(written_word)) = vm_output {
                if let Err(error) = stdout.write_elem(written_word) {
                    return (states, Some(Box::new(error)));
                }
            }

            states.push(next_state);
            current_state = states.last().unwrap();
        }

        (states, None)
    }

    pub fn run_with_input(
        &self,
        input: &[BFieldElement],
        secret_input: &[BFieldElement],
    ) -> (Vec<VMState>, Vec<BFieldElement>, Option<Box<dyn Error>>) {
        let input_bytes = input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let secret_input_bytes = secret_input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let mut stdin = VecStream::new_from_bytes(&input_bytes);
        let mut secret_in = VecStream::new_from_bytes(&secret_input_bytes);
        let mut stdout = VecStream::new_from_bytes(&[]);

        let (trace, err) = self.run(&mut stdin, &mut secret_in, &mut stdout);

        let out = stdout.to_bword_vec();

        (trace, out, err)
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[cfg(test)]
pub mod triton_vm_tests {
    use std::iter::zip;
    use std::ops::{BitAnd, BitXor};

    use num_traits::{One, Zero};
    use rand::rngs::ThreadRng;
    use rand::{Rng, RngCore};
    use twenty_first::shared_math::mpolynomial::MPolynomial;
    use twenty_first::shared_math::other;
    use twenty_first::shared_math::other::roundup_npo2;
    use twenty_first::shared_math::rescue_prime_regular::{RescuePrimeRegular, NUM_ROUNDS};
    use twenty_first::timing_reporter::TimingReporter;

    use crate::instruction::{sample_programs, AnInstruction};
    use crate::table::base_matrix::{BaseMatrices, ProcessorMatrixRow};
    use crate::table::base_table::{Extendable, InheritsFromTable};
    use crate::table::challenges::AllChallenges;
    use crate::table::extension_table::Evaluable;
    use crate::table::processor_table::{self, ProcessorTable};
    use crate::table::table_collection::interpolant_degree;
    use crate::table::table_column::ProcessorBaseTableColumn;

    use super::*;

    #[test]
    fn initialise_table_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        let stdin = vec![BFieldElement::new(42), BFieldElement::new(56)];
        let secret_in = vec![];

        let (base_matrices, err, stdout) = program.simulate_with_input(&stdin, &secret_in);

        println!(
            "VM output: [{}]",
            stdout
                .iter()
                .map(|s| format!("{s}"))
                .collect_vec()
                .join(", ")
        );

        if let Some(e) = err {
            panic!("Execution failed: {e}");
        }
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    #[test]
    fn initialise_table_42_test() {
        // 1. Execute program
        let code = "
        push 5
        push 18446744069414584320
        add
        halt
    ";
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);

        let (base_matrices, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    #[test]
    fn simulate_tvm_gcd_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        let stdin = vec![42_u64.into(), 56_u64.into()];
        let secret_in = vec![];

        let (_, err, stdout) = program.simulate_with_input(&stdin, &secret_in);

        println!(
            "VM output: [{}]",
            stdout
                .iter()
                .map(|s| format!("{s}"))
                .collect_vec()
                .join(", ")
        );

        if let Some(e) = err {
            panic!("Execution failed: {e}");
        }

        let expected_symbol = BFieldElement::new(14);
        let computed_symbol = stdout[0];

        assert_eq!(expected_symbol, computed_symbol);
    }

    #[test]
    fn hello_world() {
        // 1. Execute program
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);

        let (aet, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout);
        let base_matrices = BaseMatrices::new(aet, &program);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix.clone() {
            println!("{}", ProcessorMatrixRow { row });
        }

        // 1. Check `output_matrix`.
        {
            let expecteds = vec![
                10, 33, 100, 108, 114, 111, 87, 32, 44, 111, 108, 108, 101, 72,
            ]
            .into_iter()
            .rev()
            .map(BFieldElement::new);
            let actuals = stdout.to_bword_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each `hash` operation result in 8 rows.
        {
            let hash_instruction_count = 0;
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert!(hash_instruction_count <= 8 * prc_rows_count)
        }

        //3. noRows(jmpstack_tabel) == noRows(processor_table)
        {
            let jmp_rows_count = base_matrices.jump_stack_matrix.len();
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert_eq!(jmp_rows_count, prc_rows_count)
        }

        // "4. "READIO; WRITEIO" -> noRows(inputable) + noRows(outputtable) == noReadIO +
        // noWriteIO"

        {
            // Input
            let expected_input_count = 0;

            let actual_input_count = stdin.to_bword_vec().len();

            assert_eq!(expected_input_count, actual_input_count);

            // Output
            let expected_output_count = 14;
            //let actual = base_matrices.ram_matrix.len();

            let actual_output_count = stdout.to_bword_vec().len();

            assert_eq!(expected_output_count, actual_output_count);
        }
    }

    #[test]
    fn hash_hash_hash_test() {
        // 1. Execute program
        let code = sample_programs::HASH_HASH_HASH_HALT;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);

        let (aet, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout);
        let base_matrices = BaseMatrices::new(aet, &program);

        let jmp_rows_count = base_matrices.jump_stack_matrix.len();
        let prc_rows_count = base_matrices.processor_matrix.len();

        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
        println!("Errors: {:?}", err);

        // 1. Each of three `hash` instructions result in NUM_ROUNDS+1 rows.
        assert_eq!(3 * (NUM_ROUNDS + 1), base_matrices.hash_matrix.len());

        // 2. noRows(jmpstack_table) == noRows(processor_table)
        assert_eq!(jmp_rows_count, prc_rows_count);

        // 3. The number of input_table rows is equivalent to the number of read_io instructions.
        assert_eq!(0, stdin.to_bword_vec().len());

        // 4. The number of output_table rows is equivalent to the number of write_io instructions.
        assert_eq!(0, stdout.to_bword_vec().len());
    }

    /// Source code and associated input. Primarily for testing of the VM's instructions.
    pub struct SourceCodeAndInput {
        pub source_code: String,
        pub input: Vec<BFieldElement>,
        pub secret_input: Vec<BFieldElement>,
    }

    impl SourceCodeAndInput {
        fn without_input(source_code: &str) -> Self {
            Self {
                source_code: source_code.to_string(),
                input: vec![],
                secret_input: vec![],
            }
        }

        fn run(&self) -> Vec<BFieldElement> {
            let program =
                Program::from_code(&self.source_code).expect("Could not load source code");
            let (_, output, err) = program.run_with_input(&self.input, &self.secret_input);
            if let Some(e) = err {
                panic!("Running the program failed: {}", e)
            }
            output
        }

        fn simulate(
            &self,
        ) -> (
            AlgebraicExecutionTrace,
            Option<Box<dyn Error>>,
            Vec<BFieldElement>,
        ) {
            let program =
                Program::from_code(&self.source_code).expect("Could not load source code.");
            program.simulate_with_input(&self.input, &self.secret_input)
        }
    }

    pub fn test_hash_nop_nop_lt() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("hash nop hash nop nop hash push 3 push 2 lt assert halt")
    }

    pub fn test_program_for_push_pop_dup_swap_nop() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 1 push 2 pop assert \
            push 1 dup0 assert assert \
            push 1 push 2 swap1 assert pop \
            nop nop nop halt",
        )
    }

    pub fn test_program_for_divine() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "divine assert halt".to_string(),
            input: vec![],
            secret_input: vec![BFieldElement::one()],
        }
    }

    pub fn test_program_for_skiz() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 1 skiz push 0 skiz assert push 1 skiz halt")
    }

    pub fn test_program_for_call_recurse_return() -> SourceCodeAndInput {
        let source_code = "push 2 call label halt label: push -1 add dup0 skiz recurse return";
        SourceCodeAndInput::without_input(source_code)
    }

    pub fn test_program_for_write_mem_read_mem() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2 push 1 write_mem pop push 0 read_mem assert halt")
    }

    pub fn test_program_for_hash() -> SourceCodeAndInput {
        let source_code =
            "push 0 push 0 push 0 push 1 push 2 push 3 hash pop pop pop pop pop read_io eq assert halt";
        let mut hash_input = [BFieldElement::zero(); 10];
        hash_input[0] = BFieldElement::new(3);
        hash_input[1] = BFieldElement::new(2);
        hash_input[2] = BFieldElement::new(1);
        let digest = RescuePrimeRegular::hash_10(&hash_input);
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![digest.to_vec()[0]],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_divine_sibling_noswitch() -> SourceCodeAndInput {
        let source_code = "
            push 3 \
            push 4 push 2 push 2 push 2 push 1 \
            push 5679457 push 1337 push 345887 push -234578456 push 23657565 \
            divine_sibling \
            push 1 add assert assert assert assert assert \
            assert \
            push -1 add assert \
            push -1 add assert \
            push -1 add assert \
            push -3 add assert \
            assert halt ";
        let one = BFieldElement::one();
        let zero = BFieldElement::zero();
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![one, one, one, one, zero],
        }
    }

    pub fn test_program_for_divine_sibling_switch() -> SourceCodeAndInput {
        let source_code = "
            push 2 \
            push 4 push 2 push 2 push 2 push 1 \
            push 5679457 push 1337 push 345887 push -234578456 push 23657565 \
            divine_sibling \
            assert \
            push -1 add assert \
            push -1 add assert \
            push -1 add assert \
            push -3 add assert \
            push 1 add assert assert assert assert assert \
            assert halt ";
        let one = BFieldElement::one();
        let zero = BFieldElement::zero();
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![one, one, one, one, zero],
        }
    }

    pub fn test_program_for_assert_vector() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 1 push 2 push 3 push 4 push 5 \
             push 1 push 2 push 3 push 4 push 5 \
             assert_vector halt",
        )
    }

    pub fn property_based_test_program_for_assert_vector() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.gen_range(0..BFieldElement::QUOTIENT);
        let st1 = rng.gen_range(0..BFieldElement::QUOTIENT);
        let st2 = rng.gen_range(0..BFieldElement::QUOTIENT);
        let st3 = rng.gen_range(0..BFieldElement::QUOTIENT);
        let st4 = rng.gen_range(0..BFieldElement::QUOTIENT);

        let source_code = format!(
            "push {} push {} push {} push {} push {} \
            read_io read_io read_io read_io read_io \
            assert_vector halt",
            st4, st3, st2, st1, st0,
        );

        SourceCodeAndInput {
            source_code,
            input: vec![st4.into(), st3.into(), st2.into(), st1.into(), st0.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_add_mul_invert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 2 push -1 add assert \
            push -1 push -1 mul assert \
            push 3 dup0 invert mul assert \
            halt",
        )
    }

    pub fn test_program_for_instruction_split() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push -1 split swap1 lt assert halt ")
    }

    pub fn property_based_test_program_for_split() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::QUOTIENT;
        let hi = st0 >> 32;
        let lo = st0 & 0xffff_ffff;

        let source_code = format!(
            "push {} split read_io eq assert read_io eq assert halt",
            st0
        );

        SourceCodeAndInput {
            source_code,
            input: vec![hi.into(), lo.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_eq() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "read_io divine eq assert halt".to_string(),
            input: vec![BFieldElement::new(42)],
            secret_input: vec![BFieldElement::new(42)],
        }
    }

    pub fn property_based_test_program_for_eq() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u64() % BFieldElement::QUOTIENT;

        let source_code = format!(
            "push {} dup0 read_io eq assert dup0 divine eq assert halt",
            st0
        );

        SourceCodeAndInput {
            source_code,
            input: vec![st0.into()],
            secret_input: vec![st0.into()],
        }
    }

    pub fn test_program_for_lsb() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 3 lsb assert assert halt")
    }

    pub fn property_based_test_program_for_lsb() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let lsb = st0 % 2;
        let st0_shift_right = st0 >> 1;

        let source_code = format!("push {} lsb read_io eq assert read_io eq assert halt", st0);

        SourceCodeAndInput {
            source_code,
            input: vec![lsb.into(), st0_shift_right.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_lt() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 2 lt assert halt")
    }

    pub fn property_based_test_program_for_lt() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st1 = rng.next_u32();
        let st0 = rng.next_u32();
        let result = if st0 < st1 {
            1_u64.into()
        } else {
            0_u64.into()
        };

        let source_code = format!("push {} push {} lt read_io eq assert halt", st1, st0);

        SourceCodeAndInput {
            source_code,
            input: vec![result],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_and() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 3 and assert halt")
    }

    pub fn property_based_test_program_for_and() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st1 = rng.next_u32();
        let st0 = rng.next_u32();
        let result = st0.bitand(st1);

        let source_code = format!("push {} push {} and read_io eq assert halt", st1, st0);

        SourceCodeAndInput {
            source_code,
            input: vec![result.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_xor() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 7 push 6 xor assert halt")
    }

    pub fn property_based_test_program_for_xor() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st1 = rng.next_u32();
        let st0 = rng.next_u32();
        let result = st0.bitxor(st1);

        let source_code = format!("push {} push {} xor read_io eq assert halt", st1, st0);

        SourceCodeAndInput {
            source_code,
            input: vec![result.into()],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_reverse() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2147483648 reverse assert halt")
    }

    pub fn property_based_test_program_for_reverse() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();
        let st0_rev = st0.reverse_bits().into();

        let source_code = format!("push {} reverse read_io eq assert halt", st0);

        SourceCodeAndInput {
            source_code,
            input: vec![st0_rev],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_lte() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 2 lte assert halt")
    }

    pub fn property_based_test_program_for_lte() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st1 = rng.next_u32();
        let st0 = rng.next_u32();
        let result = if st0 <= st1 {
            1_u64.into()
        } else {
            0_u64.into()
        };

        let source_code = format!("push {} push {} lte read_io eq assert halt", st1, st0);

        SourceCodeAndInput {
            source_code,
            input: vec![result],
            secret_input: vec![],
        }
    }

    pub fn test_program_for_div() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2 push 3 div assert assert halt")
    }

    pub fn property_based_test_program_for_div() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let denominator = rng.next_u32();
        let numerator = rng.next_u32();
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;

        let source_code = format!(
            "push {} push {} div read_io eq assert read_io eq assert halt",
            denominator, numerator
        );

        SourceCodeAndInput {
            source_code,
            input: vec![remainder.into(), quotient.into()],
            secret_input: vec![],
        }
    }

    pub fn property_based_test_program_for_is_u32() -> SourceCodeAndInput {
        let mut rng = ThreadRng::default();
        let st0 = rng.next_u32();

        let source_code = format!("push {} is_u32 halt", st0);

        SourceCodeAndInput::without_input(&source_code)
    }

    #[test]
    #[should_panic(expected = "st0 must be 1.")]
    pub fn negative_property_is_u32_test() {
        let mut rng = ThreadRng::default();
        let st0 = (rng.next_u32() as u64) << 32;

        let source_code = format!("push {} is_u32 halt", st0);
        let program = SourceCodeAndInput::without_input(&source_code);
        let _ = program.run();
    }

    pub fn test_program_for_split() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push -2 split push 4294967294 eq assert push 4294967295 eq assert \
             push -1 split push 4294967295 eq assert push 0 eq assert \
             push  0 split push 0 eq assert push 0 eq assert \
             push  1 split push 0 eq assert push 1 eq assert \
             push  2 split push 0 eq assert push 2 eq assert \
             push 4294967297 split assert assert \
             halt",
        )
    }

    pub fn test_program_for_split_assert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push -2 split_assert push 4294967294 eq assert push 4294967295 eq assert \
             push -1 split_assert push 4294967295 eq assert push 0 eq assert \
             push  0 split_assert push 0 eq assert push 0 eq assert \
             push  1 split_assert push 0 eq assert push 1 eq assert \
             push  2 split_assert push 0 eq assert push 2 eq assert \
             push 4294967297 split_assert assert assert \
             halt",
        )
    }

    pub fn test_program_for_xxadd() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 push 9 push 10 xxadd halt")
    }

    pub fn test_program_for_xxmul() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 push 9 push 10 xxmul halt")
    }

    pub fn test_program_for_xinvert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 xinvert halt")
    }

    pub fn test_program_for_xbmul() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 xbmul halt")
    }

    pub fn test_program_for_read_io_write_io() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "read_io assert read_io read_io dup1 dup1 add write_io mul write_io halt"
                .to_string(),
            input: vec![1_u64.into(), 3_u64.into(), 14_u64.into()],
            secret_input: vec![],
        }
    }

    pub fn small_tasm_test_programs() -> Vec<SourceCodeAndInput> {
        vec![
            test_program_for_push_pop_dup_swap_nop(),
            test_program_for_divine(),
            test_program_for_skiz(),
            test_program_for_call_recurse_return(),
            test_program_for_write_mem_read_mem(),
            test_program_for_hash(),
            test_program_for_divine_sibling_noswitch(),
            test_program_for_divine_sibling_switch(),
            test_program_for_assert_vector(),
            test_program_for_add_mul_invert(),
            test_program_for_eq(),
            test_program_for_lsb(),
            test_program_for_split(),
            test_program_for_xxadd(),
            test_program_for_xxmul(),
            test_program_for_xinvert(),
            test_program_for_xbmul(),
            test_program_for_read_io_write_io(),
        ]
    }

    pub fn property_based_test_programs() -> Vec<SourceCodeAndInput> {
        vec![
            property_based_test_program_for_assert_vector(),
            property_based_test_program_for_split(),
            property_based_test_program_for_eq(),
            property_based_test_program_for_lsb(),
            property_based_test_program_for_lt(),
            property_based_test_program_for_and(),
            property_based_test_program_for_xor(),
            property_based_test_program_for_reverse(),
            property_based_test_program_for_lte(),
            property_based_test_program_for_div(),
            property_based_test_program_for_is_u32(),
        ]
    }

    /// programs with a cycle count of 150 and upwards
    pub fn bigger_tasm_test_programs() -> Vec<SourceCodeAndInput> {
        vec![
            test_hash_nop_nop_lt(),
            test_program_for_instruction_split(),
            test_program_for_lt(),
            test_program_for_and(),
            test_program_for_xor(),
            test_program_for_reverse(),
            test_program_for_lte(),
            test_program_for_div(),
            test_program_for_split_assert(),
        ]
    }

    #[test]
    fn processor_table_constraints_evaluate_to_zero_for_small_tasm_programs_test() {
        processor_table_constraints_evaluate_to_zero(&small_tasm_test_programs())
    }

    #[test]
    fn processor_table_constraints_evaluate_to_zero_for_property_based_tasm_programs_test() {
        processor_table_constraints_evaluate_to_zero(&property_based_test_programs())
    }

    #[test]
    #[ignore = "too slow"]
    fn processor_table_constraints_evaluate_to_zero_for_bigger_tasm_programs_test() {
        processor_table_constraints_evaluate_to_zero(&bigger_tasm_test_programs())
    }

    fn processor_table_constraints_evaluate_to_zero(all_programs: &[SourceCodeAndInput]) {
        let mut timer = TimingReporter::start();
        for (code_idx, program) in all_programs.iter().enumerate() {
            let (aet, err, output) = program.simulate();

            println!("\nChecking transition constraints for program number {code_idx}");
            println!(
                "VM output: [{}]",
                output
                    .iter()
                    .map(|s| format!("{s}"))
                    .collect_vec()
                    .join(", ")
            );
            if let Some(e) = err {
                panic!("The VM is not happy: {}", e);
            }

            let processor_matrix = aet
                .processor_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect_vec();
            let num_cycles = processor_matrix.len();

            let mut processor_table = ProcessorTable::new_prover(processor_matrix);
            let padded_height = roundup_npo2(processor_table.data().len() as u64) as usize;
            processor_table.pad(padded_height);

            assert!(
                other::is_power_of_two(processor_table.data().len()),
                "Matrix length must be power of 2 after padding"
            );

            let num_trace_randomizers = 2;
            let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
            let ext_processor_table = processor_table.extend(
                &AllChallenges::placeholder().processor_table_challenges,
                interpolant_degree,
            );

            for (row_idx, (row, next_row)) in ext_processor_table
                .data()
                .iter()
                .tuple_windows()
                .enumerate()
            {
                let evaluation_point = vec![row.clone(), next_row.clone()].concat();

                for (tc_idx, tc_evaluation_result) in ext_processor_table
                    .evaluate_transition_constraints(&evaluation_point)
                    .iter()
                    .enumerate()
                {
                    if !tc_evaluation_result.is_zero() {
                        panic!(
                            "In row {row_idx}, the constraint with index {tc_idx} evaluates to \
                            {tc_evaluation_result} but must be 0.\n\
                            Instruction: {:?} – opcode: {:?}\n\
                            Evaluation Point, first half:   {:?}\n\
                            Evaluation Point, second half:  {:?}",
                            AnInstruction::<BFieldElement>::try_from(
                                evaluation_point[ProcessorBaseTableColumn::CI as usize]
                                    .coefficients[0]
                                    .value(),
                            )
                            .unwrap(),
                            evaluation_point[ProcessorBaseTableColumn::CI as usize].coefficients[0]
                                .value(),
                            evaluation_point[..processor_table::FULL_WIDTH].to_vec(),
                            evaluation_point[processor_table::FULL_WIDTH..].to_vec()
                        );
                    }
                }
            }
            timer.elapsed(
                format!("Program number {code_idx:>2} (cycles: {num_cycles:>4})").as_str(),
            );
        }
        println!("{}", timer.finish());
    }

    fn _assert_air_constraints_on_matrix(
        table_data: &[Vec<BFieldElement>],
        air_constraints: &[MPolynomial<BFieldElement>],
    ) {
        for step in 0..table_data.len() - 1 {
            let register: Vec<BFieldElement> = table_data[step].clone();
            let next_register: Vec<BFieldElement> = table_data[step + 1].clone();
            let point: Vec<BFieldElement> = vec![register, next_register].concat();

            for air_constraint in air_constraints.iter() {
                assert!(air_constraint.evaluate(&point).is_zero());
            }
        }
    }

    #[test]
    fn xxadd_test() {
        let stdin_words = vec![
            BFieldElement::new(2),
            BFieldElement::new(3),
            BFieldElement::new(5),
            BFieldElement::new(7),
            BFieldElement::new(11),
            BFieldElement::new(13),
        ];
        let xxadd_code = "
            read_io read_io read_io
            read_io read_io read_io
            xxadd
            swap2
            write_io write_io write_io
            halt
        ";
        let program = SourceCodeAndInput {
            source_code: xxadd_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let actual_stdout = program.run();
        let expected_stdout = VecStream::new(&[
            BFieldElement::new(9),
            BFieldElement::new(14),
            BFieldElement::new(18),
        ]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn xxmul_test() {
        let stdin_words = vec![
            BFieldElement::new(2),
            BFieldElement::new(3),
            BFieldElement::new(5),
            BFieldElement::new(7),
            BFieldElement::new(11),
            BFieldElement::new(13),
        ];
        let xxmul_code = "
            read_io read_io read_io
            read_io read_io read_io
            xxmul
            swap2
            write_io write_io write_io
            halt
        ";
        let program = SourceCodeAndInput {
            source_code: xxmul_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let actual_stdout = program.run();
        let expected_stdout = VecStream::new(&[
            BFieldElement::new(108),
            BFieldElement::new(123),
            BFieldElement::new(22),
        ]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn xinv_test() {
        let stdin_words = vec![
            BFieldElement::new(2),
            BFieldElement::new(3),
            BFieldElement::new(5),
        ];
        let xinv_code = "
            read_io read_io read_io
            dup2 dup2 dup2
            dup2 dup2 dup2
            xinvert xxmul
            swap2
            write_io write_io write_io
            xinvert
            swap2
            write_io write_io write_io
            halt";
        let program = SourceCodeAndInput {
            source_code: xinv_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let actual_stdout = program.run();
        let expected_stdout = VecStream::new(&[
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::one(),
            BFieldElement::new(16360893149904808002),
            BFieldElement::new(14209859389160351173),
            BFieldElement::new(4432433203958274678),
        ]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn xbmul_test() {
        let stdin_words = vec![
            BFieldElement::new(2),
            BFieldElement::new(3),
            BFieldElement::new(5),
            BFieldElement::new(7),
        ];
        let xbmul_code: &str = "
            read_io read_io read_io
            read_io
            xbmul
            swap2
            write_io write_io write_io
            halt";
        let program = SourceCodeAndInput {
            source_code: xbmul_code.to_string(),
            input: stdin_words,
            secret_input: vec![],
        };

        let actual_stdout = program.run();
        let expected_stdout = VecStream::new(&[
            BFieldElement::new(14),
            BFieldElement::new(21),
            BFieldElement::new(35),
        ]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn pseudo_sub_test() {
        let actual_stdout =
            SourceCodeAndInput::without_input("push 7 push 19 sub write_io halt").run();
        let expected_stdout = VecStream::new(&[BFieldElement::new(12)]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }
}
