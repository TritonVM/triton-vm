use crate::instruction;
use crate::instruction::{parse, Instruction, LabelledInstruction};
use crate::state::{VMOutput, VMState, STATE_REGISTER_COUNT};
use crate::stdio::{InputStream, OutputStream, VecStream};
use crate::table::base_matrix::BaseMatrices;
use itertools::Itertools;
use std::error::Error;
use std::fmt::Display;
use std::io::Cursor;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_xlix::{neptune_params, RescuePrimeXlix};

type BWord = BFieldElement;

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

    /// Convert a `Program` to a `Vec<BWord>`.
    ///
    /// Every single-word instruction is converted to a single word.
    ///
    /// Every double-word instruction is converted to two words.
    pub fn to_bwords(&self) -> Vec<BWord> {
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

    /// Simulate (execute) a `Program` and record every state transition.
    ///
    /// Returns a `BaseMatrices` that records the execution.
    ///
    /// Optionally returns errors on premature termination, but returns a
    /// `BaseMatrices` for the execution up to the point of failure.
    pub fn simulate<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<{ STATE_REGISTER_COUNT }>,
    ) -> (BaseMatrices, Option<Box<dyn Error>>)
    where
        In: InputStream,
        Out: OutputStream,
    {
        let mut base_matrices = BaseMatrices::default();
        base_matrices.initialize(self);

        let mut state = VMState::new(self);
        let initial_instruction = state.current_instruction().unwrap();

        base_matrices.append(&state, None, initial_instruction);

        while !state.is_complete() {
            let vm_output = match state.step_mut(stdin, secret_in, rescue_prime) {
                Err(err) => return (base_matrices, Some(err)),
                Ok(vm_output) => vm_output,
            };
            let current_instruction = state.current_instruction().unwrap_or(Instruction::Halt);

            if let Some(VMOutput::WriteOutputSymbol(written_word)) = vm_output {
                if let Err(error) = stdout.write_elem(written_word) {
                    return (base_matrices, Some(Box::new(error)));
                }
            }

            base_matrices.append(&state, vm_output, current_instruction);
        }

        base_matrices.sort_instruction_matrix();
        base_matrices.sort_op_stack_matrix();
        base_matrices.sort_ram_matrix();
        base_matrices.sort_jump_stack_matrix();

        base_matrices.set_ram_matrix_inverse_of_ramp_diff();

        (base_matrices, None)
    }

    /// Wrapper around `.simulate()` for easier i/o handling. Behaviour is that of `.simulate()`,
    /// except the executed program's output is returned instead of modified in-place. Consequently,
    /// takes as arguments only `input` and `secret_input`.
    pub fn simulate_with_input(
        &self,
        input: &[BFieldElement],
        secret_input: &[BFieldElement],
    ) -> (BaseMatrices, Option<Box<dyn Error>>, Vec<BFieldElement>) {
        let mut stdin = VecStream::new_bwords(input);
        let mut secret_in = VecStream::new_bwords(secret_input);
        let mut stdout = VecStream::new_bytes(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, vm_error) =
            self.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);
        (base_matrices, vm_error, stdout.to_bword_vec())
    }

    pub fn run<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<{ STATE_REGISTER_COUNT }>,
    ) -> (Vec<VMState>, Option<Box<dyn Error>>)
    where
        In: InputStream,
        Out: OutputStream,
    {
        let mut states = vec![VMState::new(self)];
        let mut current_state = states.last().unwrap();

        while !current_state.is_complete() {
            let step = current_state.step(stdin, secret_in, rescue_prime);
            let (next_state, vm_output) = match step {
                Err(err) => return (states, Some(err)),
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
    ) -> (Vec<VMState>, Vec<BWord>, Option<Box<dyn Error>>) {
        let input_bytes = input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let secret_input_bytes = secret_input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let mut stdin = VecStream::new_bytes(&input_bytes);
        let mut secret_in = VecStream::new_bytes(&secret_input_bytes);
        let mut stdout = VecStream::new_bytes(&[]);
        let rescue_prime = neptune_params();

        let (trace, err) = self.run(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

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
mod triton_vm_tests {
    use std::iter::zip;

    use super::*;
    use crate::instruction::sample_programs;
    use crate::table::base_matrix::ProcessorMatrixRow;
    use crate::table::base_table::{Extendable, InheritsFromTable};
    use crate::table::challenges_endpoints::{AllChallenges, AllEndpoints};
    use crate::table::extension_table::Evaluable;
    use crate::table::processor_table::ProcessorTable;
    use twenty_first::shared_math::mpolynomial::MPolynomial;
    use twenty_first::shared_math::other;
    use twenty_first::shared_math::traits::IdentityValues;

    #[test]
    fn initialise_table_test() {
        // 1. Parse program
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[42.into(), 56.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        // 2. Execute program, convert to base matrices
        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("Err: {:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }

        println!("Output: {:?}", stdout.to_bword_vec());

        // 3. Extract constraints
        // 4. Check constraints
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

        let mut stdin = VecStream::new_bwords(&[]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    #[test]
    fn simulate_gcd_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[42.into(), 56.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (_, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        assert!(err.is_none());
        let expected = BWord::new(14);
        let actual = stdout.to_bword_vec()[0];

        assert_eq!(expected, actual);
    }

    #[test]
    fn hello_world() {
        // 1. Execute program
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

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
            .map(|x| BWord::new(x));
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

        let mut stdin = VecStream::new_bwords(&[]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let jmp_rows_count = base_matrices.jump_stack_matrix.len();
        let prc_rows_count = base_matrices.processor_matrix.len();

        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
        println!("Errors: {:?}", err);

        // 1. Each of three `hash` instructions result in 8 rows.
        assert_eq!(24, base_matrices.hash_matrix.len());

        // 2. noRows(jmpstack_table) == noRows(processor_table)
        assert_eq!(jmp_rows_count, prc_rows_count);

        // 3. The number of input_table rows is equivalent to the number of read_io instructions.
        assert_eq!(0, stdin.to_bword_vec().len());

        // 4. The number of output_table rows is equivalent to the number of write_io instructions.
        assert_eq!(0, stdout.to_bword_vec().len());
    }

    fn _check_base_matrices(
        base_matrices: &BaseMatrices,
        input_symbols: &[BFieldElement],
        output_symbols: &[BFieldElement],
        expected_input_rows: usize,
        expected_output_rows: usize,
        hash_instruction_count: usize,
    ) {
        // 1. Check `output_matrix`.
        {
            let expecteds = vec![].into_iter().rev().map(|x| BWord::new(x));
            let actuals = output_symbols.to_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each `hash` operation result in 8 rows in the state matrix.
        {
            let hash_state_rows_count = base_matrices.hash_matrix.len();
            assert_eq!(hash_instruction_count * 8, hash_state_rows_count)
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
            let actual_input_rows = input_symbols.len();
            assert_eq!(expected_input_rows, actual_input_rows);

            // Output
            let actual_output_rows = output_symbols.len();

            assert_eq!(expected_output_rows, actual_output_rows);
        }
    }

    /// Source code and associated input. Primarily for testing of the VM's instructions.
    struct SourceCodeAndInput {
        source_code: String,
        input: Vec<BFieldElement>,
        secret_input: Vec<BFieldElement>,
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

        fn simulate(&self) -> (BaseMatrices, Option<Box<dyn Error>>, Vec<BFieldElement>) {
            let program =
                Program::from_code(&self.source_code).expect("Could not load source code.");
            program.simulate_with_input(&self.input, &self.secret_input)
        }
    }

    fn test_program_for_push_pop_dup_swap_nop() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 1 push 2 pop assert \
            push 1 dup0 assert assert \
            push 1 push 2 swap1 assert pop \
            nop nop nop halt",
        )
    }

    fn test_program_for_divine() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "divine assert halt".to_string(),
            input: vec![],
            secret_input: vec![1.into()],
        }
    }

    fn test_program_for_skiz() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 1 skiz push 0 skiz assert push 1 skiz halt")
    }

    fn test_program_for_call_recurse_return() -> SourceCodeAndInput {
        let source_code = "push 2 call label halt label: push -1 add dup0 skiz recurse return";
        SourceCodeAndInput::without_input(source_code)
    }

    fn test_program_for_write_mem_read_mem() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2 push 1 write_mem pop push 0 read_mem assert halt")
    }

    fn test_program_for_hash() -> SourceCodeAndInput {
        let source_code = "push 1 push 2 push 3 hash read_io eq assert halt";
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![BFieldElement::new(17994241411625465269)],
            secret_input: vec![],
        }
    }

    fn test_program_for_divine_sibling() -> SourceCodeAndInput {
        let source_code = "
            push 3 swap12 divine_sibling \
            dup0 assert dup1 assert dup2 assert \
            dup3 assert dup4 assert dup5 assert \
            halt";
        let one = BFieldElement::ring_one();
        SourceCodeAndInput {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![one, one, one, one, one, one],
        }
    }

    fn test_program_for_assert_vector() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 1 push 2 push 3 push 4 push 5 push 6 \
             push 1 push 2 push 3 push 4 push 5 push 6 \
             assert_vector halt",
        )
    }

    fn test_program_for_add_mul_invert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input(
            "push 2 push -1 add assert \
            push -1 push -1 mul assert \
            push 3 dup0 invert mul assert \
            halt",
        )
    }

    fn test_program_for_instruction_split() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push -1 split swap1 lt assert halt ")
    }

    fn test_program_for_eq() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "read_io divine eq assert halt".to_string(),
            input: vec![42.into()],
            secret_input: vec![42.into()],
        }
    }

    fn test_program_for_lt() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 3 push 2 lt assert halt")
    }

    fn test_program_for_and() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 3 and assert halt")
    }

    fn test_program_for_xor() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 7 push 6 xor assert halt")
    }

    fn test_program_for_reverse() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 2147483648 reverse assert halt")
    }

    fn test_program_for_div() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 3 push 2 div assert assert halt")
    }

    fn test_program_for_xxadd() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 push 9 push 10 xxadd halt")
    }

    fn test_program_for_xxmul() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 push 9 push 10 xxmul halt")
    }

    fn test_program_for_xinvert() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 xinvert halt")
    }

    fn test_program_for_xbmul() -> SourceCodeAndInput {
        SourceCodeAndInput::without_input("push 5 push 6 push 7 push 8 xbmul halt")
    }

    fn test_program_for_read_io() -> SourceCodeAndInput {
        SourceCodeAndInput {
            source_code: "read_io assert halt".to_string(),
            input: vec![1.into()],
            secret_input: vec![],
        }
    }

    #[test]
    fn processor_table_constraints_evaluate_to_zero_test() {
        let all_programs = vec![
            test_program_for_push_pop_dup_swap_nop(),
            test_program_for_divine(),
            test_program_for_skiz(),
            test_program_for_call_recurse_return(),
            test_program_for_write_mem_read_mem(),
            test_program_for_hash(),
            test_program_for_divine_sibling(),
            test_program_for_assert_vector(),
            test_program_for_add_mul_invert(),
            test_program_for_instruction_split(),
            test_program_for_eq(),
            test_program_for_lt(),
            test_program_for_and(),
            test_program_for_xor(),
            test_program_for_reverse(),
            test_program_for_div(),
            test_program_for_xxadd(),
            test_program_for_xxmul(),
            test_program_for_xinvert(),
            test_program_for_xbmul(),
            test_program_for_read_io(),
        ];
        for program in all_programs.into_iter() {
            println!(
                "\n\nChecking transition constraints for program: \"{}\"",
                &program.source_code
            );
            let (base_matrices, err, _) = program.simulate();

            if let Some(e) = err {
                panic!("The VM is not happy: {}", e);
            }

            let num_trace_randomizers = 2;
            let processor_matrix = base_matrices
                .processor_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect_vec();

            let mut processor_table =
                ProcessorTable::new_prover(num_trace_randomizers, processor_matrix);
            processor_table.pad();

            assert!(
                other::is_power_of_two(processor_table.data().len()),
                "Matrix length must be power of 2 after padding"
            );

            let (ext_processor_table, _) = processor_table.extend(
                &AllChallenges::dummy().processor_table_challenges,
                &AllEndpoints::dummy().processor_table_endpoints,
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
                            Evaluation Point:   {:?}",
                            evaluation_point,
                        );
                    }
                }
            }
        }
    }

    fn _assert_air_constraints_on_matrix(
        table_data: &[Vec<BFieldElement>],
        air_constraints: &[MPolynomial<BFieldElement>],
    ) {
        for step in 0..table_data.len() - 1 {
            let register: Vec<BFieldElement> = table_data[step].clone().into();
            let next_register: Vec<BFieldElement> = table_data[step + 1].clone().into();
            let point: Vec<BFieldElement> = vec![register, next_register].concat();

            for air_constraint in air_constraints.iter() {
                assert!(air_constraint.evaluate(&point).is_zero());
            }
        }
    }

    #[test]
    fn xxadd_test() {
        let stdin_words = vec![2.into(), 3.into(), 5.into(), 7.into(), 11.into(), 13.into()];
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
        let expected_stdout = VecStream::new_bwords(&[9.into(), 14.into(), 18.into()]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn xxmul_test() {
        let stdin_words = vec![2.into(), 3.into(), 5.into(), 7.into(), 11.into(), 13.into()];
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
        let expected_stdout = VecStream::new_bwords(&[108.into(), 123.into(), 22.into()]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn xinv_test() {
        let stdin_words = vec![2.into(), 3.into(), 5.into()];
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
        let expected_stdout = VecStream::new_bwords(&[
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_one(),
            BFieldElement::new(16360893149904808002),
            BFieldElement::new(14209859389160351173),
            BFieldElement::new(4432433203958274678),
        ]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn xbmul_test() {
        let stdin_words = vec![2.into(), 3.into(), 5.into(), 7.into()];
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
        let expected_stdout = VecStream::new_bwords(&[14.into(), 21.into(), 35.into()]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }

    #[test]
    fn pseudo_sub_test() {
        let actual_stdout =
            SourceCodeAndInput::without_input("push 7 push 19 sub write_io halt").run();
        let expected_stdout = VecStream::new_bwords(&[12.into()]);

        assert_eq!(expected_stdout.to_bword_vec(), actual_stdout);
    }
}
