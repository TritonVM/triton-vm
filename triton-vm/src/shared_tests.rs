use std::fs::create_dir_all;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use anyhow::Error;
use anyhow::Result;
use triton_opcodes::program::Program;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::proof::Claim;
use crate::proof::Proof;
use crate::stark::Stark;
use crate::stark::StarkParameters;
use crate::table::master_table::MasterBaseTable;
use crate::vm::run;
use crate::vm::simulate;
use crate::vm::AlgebraicExecutionTrace;

pub fn parse_setup_simulate(
    code: &str,
    input_symbols: Vec<BFieldElement>,
    secret_input_symbols: Vec<BFieldElement>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (AlgebraicExecutionTrace, Vec<BFieldElement>) {
    let program = Program::from_code(code);

    let program = program.expect("Program must parse.");

    prof_start!(maybe_profiler, "simulate");
    let (aet, stdout, err) = simulate(&program, input_symbols, secret_input_symbols);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {error}");
    }
    prof_stop!(maybe_profiler, "simulate");

    (aet, stdout)
}

pub fn parse_simulate_prove(
    code: &str,
    input_symbols: Vec<BFieldElement>,
    secret_input_symbols: Vec<BFieldElement>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (Stark, Proof) {
    let (aet, output_symbols) = parse_setup_simulate(
        code,
        input_symbols.clone(),
        secret_input_symbols,
        maybe_profiler,
    );

    let padded_height = MasterBaseTable::padded_height(&aet);
    let claim = Claim {
        input: input_symbols,
        program: aet.program.to_bwords(),
        output: output_symbols,
        padded_height,
    };
    let log_expansion_factor = 2;
    let security_level = 32;
    let parameters = StarkParameters::new(security_level, 1 << log_expansion_factor);
    let stark = Stark::new(claim, parameters);

    prof_start!(maybe_profiler, "prove");
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
        let (_, output, err) = run(&program, self.input.clone(), self.secret_input.clone());
        if let Some(e) = err {
            panic!("Running the program failed: {e}")
        }
        output
    }

    pub fn simulate(&self) -> (AlgebraicExecutionTrace, Vec<BFieldElement>, Option<Error>) {
        let program = Program::from_code(&self.source_code).expect("Could not load source code.");
        simulate(&program, self.input.clone(), self.secret_input.clone())
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
    let full_filename = format!("{}{filename}", proofs_directory());
    if File::open(full_filename).is_err() {
        return false;
    }
    true
}

pub fn load_proof(filename: &str) -> Result<Proof> {
    let full_filename = format!("{}{filename}", proofs_directory());
    let mut contents: Vec<u8> = vec![];
    let mut file_handle = File::open(full_filename)?;
    let i = file_handle.read_to_end(&mut contents)?;
    println!("Read {i} bytes of proof data from disk.");
    let proof: Proof = bincode::deserialize(&contents).expect("Cannot deserialize proof.");

    Ok(proof)
}

pub fn save_proof(filename: &str, proof: Proof) -> Result<()> {
    if !proofs_directory_exists() {
        create_proofs_directory()?;
    }
    let full_filename = format!("{}{filename}", proofs_directory());
    let mut file_handle = match File::create(full_filename.clone()) {
        Ok(fh) => fh,
        Err(e) => panic!("Cannot write proof to disk at {full_filename}: {e:?}"),
    };
    let binary = match bincode::serialize(&proof) {
        Ok(b) => b,
        Err(e) => panic!("Cannot serialize proof: {e:?}"),
    };
    let amount = file_handle.write(&binary)?;
    println!("Wrote {amount} bytes of proof data to disk.");
    Ok(())
}

pub const FIBONACCI_SEQUENCE: &str = "
    // initialize stack: ⊥ 0 1 i
    push 0
    push 1
    read_io

    // is any looping necessary?
    dup0
    skiz
    call fib-loop

    // pop zero, write result
    pop
    write_io
    halt

    // before: ⊥ 0 1 i
    // after:  ⊥ fib(i-1) fib(i) 0
    fib-loop:
        push -1   // ⊥ a b j -1
        add       // ⊥ a b (j-1)
        swap2     // ⊥ (j-1) b a
        dup1      // ⊥ (j-1) b a b
        add       // ⊥ (j-1) b (a+b)
        swap1     // ⊥ (j-1) (a+b) b
        swap2     // ⊥ b (a+b) (j-1)
        dup0      // ⊥ b (a+b) (j-1) (j-1)
        skiz      // ⊥ b (a+b) (j-1)
        recurse
        return
    ";

pub const MANY_U32_INSTRUCTIONS: &str = "
    push 1311768464867721216 split
    push 13387 push 78810 lt
    push     5 push     7 pow
    push 69584 push  6796 xor
    push 64972 push  3915 and
    push 98668 push 15787 div
    push 15787 push 98668 div
    push 98141 push  7397 and
    push 67749 push 60797 lt
    push 49528 split
    push 53483 call lsb
    push 79655 call is_u32
    push 60615 log_2_floor
    push    13 push     5 pow
    push 86323 push 37607 xor
    push 32374 push 20636 pow
    push 97416 log_2_floor
    push 14392 push 31589 div
    halt
    lsb:
        push 2 swap1 div return
    is_u32:
        split pop push 0 eq return";
