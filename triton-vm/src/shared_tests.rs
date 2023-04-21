use std::fs::create_dir_all;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use anyhow::Error;
use anyhow::Result;
use triton_opcodes::program::Program;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;

use crate::proof::Claim;
use crate::proof::Proof;
use crate::stark::Stark;
use crate::stark::StarkParameters;
use crate::table::master_table::MasterBaseTable;
use crate::vm::simulate;
use crate::vm::AlgebraicExecutionTrace;

pub fn parse_setup_simulate(
    code: &str,
    input_symbols: Vec<u64>,
    secret_input_symbols: Vec<u64>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (AlgebraicExecutionTrace, Vec<u64>) {
    let program = Program::from_code(code);

    let program = program.expect("Program must parse.");
    let public_input = input_symbols.into_iter().map(BFieldElement::new).collect();
    let secret_input = secret_input_symbols
        .into_iter()
        .map(BFieldElement::new)
        .collect();

    prof_start!(maybe_profiler, "simulate");
    let (aet, stdout, err) = simulate(&program, public_input, secret_input);
    if let Some(error) = err {
        panic!("The VM encountered the following problem: {error}");
    }
    prof_stop!(maybe_profiler, "simulate");

    let stdout = stdout.into_iter().map(|e| e.value()).collect();
    (aet, stdout)
}

pub fn parse_simulate_prove(
    code: &str,
    input_symbols: Vec<u64>,
    secret_input_symbols: Vec<u64>,
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
        program_digest: Tip5::hash(&aet.program),
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
    pub input: Vec<u64>,
    pub secret_input: Vec<u64>,
}

impl SourceCodeAndInput {
    pub fn without_input(source_code: &str) -> Self {
        Self {
            source_code: source_code.to_string(),
            input: vec![],
            secret_input: vec![],
        }
    }

    pub fn public_input(&self) -> Vec<BFieldElement> {
        self.input.iter().map(|&x| BFieldElement::new(x)).collect()
    }

    pub fn secret_input(&self) -> Vec<BFieldElement> {
        self.secret_input
            .iter()
            .map(|&x| BFieldElement::new(x))
            .collect()
    }

    #[deprecated(since = "0.19.0", note = "use `simulate` instead")]
    pub fn run(&self) -> Vec<BFieldElement> {
        let program = Program::from_code(&self.source_code).expect("Could not load source code");
        let (_, output, err) = simulate(&program, self.public_input(), self.secret_input());
        if let Some(e) = err {
            panic!("Running the program failed: {e}")
        }
        output
    }

    pub fn simulate(&self) -> (AlgebraicExecutionTrace, Vec<BFieldElement>, Option<Error>) {
        let program = Program::from_code(&self.source_code).expect("Could not load source code.");
        simulate(&program, self.public_input(), self.secret_input())
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
    dup 0
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
        swap 2    // ⊥ (j-1) b a
        dup 1     // ⊥ (j-1) b a b
        add       // ⊥ (j-1) b (a+b)
        swap 1    // ⊥ (j-1) (a+b) b
        swap 2    // ⊥ b (a+b) (j-1)
        dup 0     // ⊥ b (a+b) (j-1) (j-1)
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
        push 2 swap 1 div return
    is_u32:
        split pop push 0 eq return";

pub const VERIFY_SUDOKU: &str = "
    call initialize_primes
    call read_sudoku
    call initialize_flag
    call write_sudoku_and_check_rows
    call check_columns
    call check_squares
    push 0
    read_mem
    assert
    halt

    // For mapping legal Sudoku digits to distinct primes. Helps with checking consistency of
    // rows, columns, and boxes.
    initialize_primes:
        push 1 push  2 write_mem pop
        push 2 push  3 write_mem pop
        push 3 push  5 write_mem pop
        push 4 push  7 write_mem pop
        push 5 push 11 write_mem pop
        push 6 push 13 write_mem pop
        push 7 push 17 write_mem pop
        push 8 push 19 write_mem pop
        push 9 push 23 write_mem pop
        return

    read_sudoku:
        call read9
        call read9
        call read9
        call read9
        call read9
        call read9
        call read9
        call read9
        call read9
        return

    read9:
        call read1
        call read1
        call read1
        call read1
        call read1
        call read1
        call read1
        call read1
        call read1
        return

    // Applies the mapping from legal Sudoku digits to distinct primes.
    read1:                            // _
        read_io                       // _ d
        read_mem                      // _ d p
        swap 1                        // _ p d
        pop                           // _ p
        return

    initialize_flag:
        push 0
        push 1
        write_mem
        pop
        return

    write_sudoku_and_check_rows:      // row0 row1 row2 row3 row4 row5 row6 row7 row8
        push 9                        // row0 row1 row2 row3 row4 row5 row6 row7 row8 9
        call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 row6 row7 18
        call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 row6 27
        call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 36
        call write_and_check_one_row  // row0 row1 row2 row3 row4 45
        call write_and_check_one_row  // row0 row1 row2 row3 54
        call write_and_check_one_row  // row0 row1 row2 63
        call write_and_check_one_row  // row0 row1 72
        call write_and_check_one_row  // row0 81
        call write_and_check_one_row  // 90
        pop                           // ⊥
        return

    write_and_check_one_row:          // s0 s1 s2 s3 s4 s5 s6 s7 s8 mem_addr
        push 1                        // s0 s1 s2 s3 s4 s5 s6 s7 s8 mem_addr 1
        call multiply_and_write       // s0 s1 s2 s3 s4 s5 s6 s7 (mem_addr+1) s8
        call multiply_and_write       // s0 s1 s2 s3 s4 s5 s6 (mem_addr+2) (s8·s7)
        call multiply_and_write       // s0 s1 s2 s3 s4 s5 (mem_addr+3) (s8·s7·s6)
        call multiply_and_write       // s0 s1 s2 s3 s4 (mem_addr+4) (s8·s7·s6·s5)
        call multiply_and_write       // s0 s1 s2 s3 (mem_addr+5) (s8·s7·s6·s5·s4)
        call multiply_and_write       // s0 s1 s2 (mem_addr+6) (s8·s7·s6·s5·s4·s3)
        call multiply_and_write       // s0 s1 (mem_addr+7) (s8·s7·s6·s5·s4·s3·s2)
        call multiply_and_write       // s0 (mem_addr+8) (s8·s7·s6·s5·s4·s3·s2·s1)
        call multiply_and_write       // (mem_addr+9) (s8·s7·s6·s5·s4·s3·s2·s1·s0)
        push 223092870                // (mem_addr+9) (s8·s7·s6·s5·s4·s3·s2·s1·s0) 223092870
        eq                            // (mem_addr+9) (s8·s7·s6·s5·s4·s3·s2·s1·s0==223092870)
        skiz                          // (mem_addr+9)
        return
        push 0                        // (mem_addr+9) 0
        push 0                        // (mem_addr+9) 0 0
        write_mem                     // (mem_addr+9) 0
        pop                           // (mem_addr+9)
        return

    multiply_and_write:               // s mem_addr acc
        dup 2                         // s mem_addr acc s 
        mul                           // s mem_addr (acc·s)
        swap 1                        // s (acc·s) mem_addr
        push 1                        // s (acc·s) mem_addr 1
        add                           // s (acc·s) (mem_addr+1)
        swap 1                        // s (mem_addr+1) (acc·s)
        swap 2                        // (acc·s) (mem_addr+1) s
        write_mem                     // (acc·s) (mem_addr+1)
        swap 1                        // (mem_addr+1) (acc·s)
        return

    check_columns:
        push 1
        call check_one_column
        push 2
        call check_one_column
        push 3
        call check_one_column
        push 4
        call check_one_column
        push 5
        call check_one_column
        push 6
        call check_one_column
        push 7
        call check_one_column
        push 8
        call check_one_column
        push 9
        call check_one_column
        return

    check_one_column:
        call get_column_element
        call get_column_element
        call get_column_element
        call get_column_element
        call get_column_element
        call get_column_element
        call get_column_element
        call get_column_element
        call get_column_element
        pop
        call check_9_numbers
        return

    get_column_element:
        push 9
        add
        read_mem
        swap 1
        return

    check_squares:
        push 10
        call check_one_square
        push 13
        call check_one_square
        push 16
        call check_one_square
        push 37
        call check_one_square
        push 40
        call check_one_square
        push 43
        call check_one_square
        push 64
        call check_one_square
        push 67
        call check_one_square
        push 70
        call check_one_square
        return

    check_one_square:
        read_mem
        swap 1
        push 1
        add
        read_mem
        swap 1
        push 1
        add
        read_mem
        swap 1
        push 7
        add
        read_mem
        swap 1
        push 1
        add
        read_mem
        swap 1
        push 1
        add
        read_mem
        swap 1
        push 7
        add
        read_mem
        swap 1
        push 1
        add
        read_mem
        swap 1
        push 1
        add
        read_mem
        swap 1
        pop
        call check_9_numbers
        return

    check_9_numbers:
        mul
        mul
        mul
        mul
        mul
        mul
        mul
        mul
        // 223092870 = 2·3·5·7·11·13·17·19·23
        push 223092870
        eq
        skiz
        return
        push 0
        push 0
        write_mem
        pop
        return
    ";
