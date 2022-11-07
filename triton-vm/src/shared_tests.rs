use std::error::Error;

use twenty_first::shared_math::b_field_element::BFieldElement;

use triton_profiler::triton_profiler::TritonProfiler;
use triton_profiler::{prof_start, prof_stop};

use crate::proof::{Claim, Proof};
use crate::stark::{Stark, StarkParameters};
use crate::stdio::VecStream;
use crate::table::base_matrix::AlgebraicExecutionTrace;
use crate::table::base_matrix::BaseMatrices;
use crate::table::table_collection::BaseTableCollection;
use crate::vm::Program;

pub fn parse_setup_simulate(
    code: &str,
    input_symbols: &[BFieldElement],
    secret_input_symbols: &[BFieldElement],
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (AlgebraicExecutionTrace, VecStream, Program) {
    let program = Program::from_code(code);

    assert!(program.is_ok(), "program parses correctly");
    let program = program.unwrap();

    let mut stdin = VecStream::new(input_symbols);
    let mut secret_in = VecStream::new(secret_input_symbols);
    let mut stdout = VecStream::new(&[]);

    if let Some(profiler) = maybe_profiler.as_mut() {
        profiler.start("simulate")
    }
    let (aet, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {}", error);
    }

    if let Some(profiler) = maybe_profiler.as_mut() {
        profiler.stop("simulate")
    }
    (aet, stdout, program)
}

pub fn parse_simulate_prove(
    code: &str,
    input_symbols: &[BFieldElement],
    secret_input_symbols: &[BFieldElement],
    output_symbols: &[BFieldElement],
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (Stark, Proof) {
    let (aet, _, program) =
        parse_setup_simulate(code, input_symbols, secret_input_symbols, maybe_profiler);
    let base_matrices = BaseMatrices::new(aet.clone(), &program.to_bwords());

    prof_start!(maybe_profiler, "padding");
    let log_expansion_factor = 2;
    let security_level = 32;
    let padded_height = BaseTableCollection::padded_height(&base_matrices);
    prof_stop!(maybe_profiler, "padding");

    prof_start!(maybe_profiler, "prove");
    let parameters = StarkParameters::new(security_level, 1 << log_expansion_factor);
    let program = Program::from_code(code);
    let program = match program {
        Ok(p) => p.to_bwords(),
        Err(e) => panic!(
            "Could not convert program from code to vector of BFieldElements: {}",
            e
        ),
    };
    let claim = Claim {
        input: input_symbols.to_vec(),
        program,
        output: output_symbols.to_vec(),
        padded_height,
    };
    let stark = Stark::new(claim, parameters);

    let proof = stark.prove(aet, maybe_profiler);
    prof_stop!(maybe_profiler, "prove");

    (stark, proof)
}

/// Source code and associated input. Primarily for testing of the VM's instructions.
pub struct SourceCodeAndInput {
    pub source_code: String,
    pub input: Vec<BFieldElement>,
    pub secret_input: Vec<BFieldElement>,
}

impl SourceCodeAndInput {
    pub fn without_input(source_code: &str) -> Self {
        Self {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![],
        }
    }

    pub fn run(&self) -> Vec<BFieldElement> {
        let program = Program::from_code(&self.source_code).expect("Could not load source code");
        let (_, output, err) = program.run_with_input(&self.input, &self.secret_input);
        if let Some(e) = err {
            panic!("Running the program failed: {}", e)
        }
        output
    }

    pub fn simulate(
        &self,
    ) -> (
        AlgebraicExecutionTrace,
        Option<Box<dyn Error>>,
        Vec<BFieldElement>,
    ) {
        let program = Program::from_code(&self.source_code).expect("Could not load source code.");
        program.simulate_with_input(&self.input, &self.secret_input)
    }
}

pub fn test_hash_nop_nop_lt() -> SourceCodeAndInput {
    SourceCodeAndInput::without_input("hash nop hash nop nop hash push 3 push 2 lt assert halt")
}

pub fn test_halt() -> SourceCodeAndInput {
    SourceCodeAndInput::without_input("halt")
}
