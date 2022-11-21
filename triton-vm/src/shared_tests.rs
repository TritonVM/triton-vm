use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Error, Result};
use twenty_first::shared_math::b_field_element::BFieldElement;

use triton_profiler::triton_profiler::TritonProfiler;
use triton_profiler::{prof_start, prof_stop};

use crate::proof::{Claim, Proof};
use crate::stark::{Stark, StarkParameters};
use crate::table::base_matrix::AlgebraicExecutionTrace;
use crate::table::base_matrix::BaseMatrices;
use crate::table::table_collection::BaseTableCollection;
use crate::vm::Program;

pub fn parse_setup_simulate(
    code: &str,
    input_symbols: Vec<BFieldElement>,
    secret_input_symbols: Vec<BFieldElement>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (AlgebraicExecutionTrace, Vec<BFieldElement>, Program) {
    let program = Program::from_code(code);

    assert!(program.is_ok(), "program parses correctly");
    let program = program.unwrap();

    prof_start!(maybe_profiler, "simulate");
    let (aet, stdout, err) = program.simulate(input_symbols, secret_input_symbols);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {}", error);
    }
    prof_stop!(maybe_profiler, "simulate");

    (aet, stdout, program)
}

pub fn parse_simulate_prove(
    code: &str,
    input_symbols: Vec<BFieldElement>,
    secret_input_symbols: Vec<BFieldElement>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (Stark, Proof) {
    let (aet, output_symbols, program) = parse_setup_simulate(
        code,
        input_symbols.clone(),
        secret_input_symbols,
        maybe_profiler,
    );
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
        input: input_symbols,
        program,
        output: output_symbols,
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
        let (_, output, err) = program.run(self.input.clone(), self.secret_input.clone());
        if let Some(e) = err {
            panic!("Running the program failed: {}", e)
        }
        output
    }

    pub fn simulate(&self) -> (AlgebraicExecutionTrace, Vec<BFieldElement>, Option<Error>) {
        let program = Program::from_code(&self.source_code).expect("Could not load source code.");
        program.simulate(self.input.clone(), self.secret_input.clone())
    }
}

pub fn test_hash_nop_nop_lt() -> SourceCodeAndInput {
    SourceCodeAndInput::without_input("hash nop hash nop nop hash push 3 push 2 lt assert halt")
}

pub fn test_halt() -> SourceCodeAndInput {
    SourceCodeAndInput::without_input("halt")
}

pub fn proofs_directory() -> String {
    "proofs/".to_owned()
}

pub fn create_proofs_directory() -> Result<()> {
    match create_dir_all(proofs_directory()) {
        Ok(ay) => Ok(ay),
        Err(e) => Err(Error::new(e)),
    }
}

pub fn proofs_directory_exists() -> bool {
    Path::new(&proofs_directory()).is_dir()
}

pub fn proof_file_exists(filename: &str) -> bool {
    if !Path::new(&proofs_directory()).is_dir() {
        return false;
    }
    let full_filename = format!("{}{}", proofs_directory(), filename);
    if File::open(full_filename).is_err() {
        return false;
    }
    true
}

pub fn load_proof(filename: &str) -> Result<Proof> {
    let full_filename = format!("{}{}", proofs_directory(), filename);
    let mut contents: Vec<u8> = vec![];
    let mut file_handle = File::open(full_filename)?;
    let i = file_handle.read_to_end(&mut contents)?;
    println!("Read {} bytes of proof data from disk.", i);
    let proof: Proof = bincode::deserialize(&contents).expect("Cannot deserialize proof.");

    Ok(proof)
}

pub fn save_proof(filename: &str, proof: Proof) -> Result<()> {
    if !proofs_directory_exists() {
        create_proofs_directory()?;
    }
    let full_filename = format!("{}{}", proofs_directory(), filename);
    let mut file_handle = match File::create(full_filename.clone()) {
        Ok(fh) => fh,
        Err(e) => panic!("Cannot write proof to disk at {}: {:?}", full_filename, e),
    };
    let binary = match bincode::serialize(&proof) {
        Ok(b) => b,
        Err(e) => panic!("Cannot serialize proof: {:?}", e),
    };
    let amount = file_handle.write(&binary)?;
    println!("Wrote {} bytes of proof data to disk.", amount);
    Ok(())
}
