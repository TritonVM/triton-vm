use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

use triton_vm::prelude::*;

#[derive(StructOpt, Debug)]
enum Opts {
    Prove {
        #[structopt(long, parse(from_os_str))]
        dst: PathBuf,
    },
    Verify {
        src: PathBuf,
    },
}

fn main() {
    // read command-line args
    match Opts::from_args() {
        Opts::Prove { dst } => prove(dst),
        Opts::Verify { src } => verify(src),
    }
}

fn prove(output_path: PathBuf) {
    let factorial_program = triton_program!(
                            //op stack:
        read_io 1           // n
        push 1              // n accumulator
        call factorial      // 0 accumulator!
        write_io 1          // 0
        halt

        factorial:          // n acc
        // if n == 0: return
            dup 1           // n acc n
            push 0 eq       // n acc n==0
            skiz            // n acc
                return      // 0 acc
            // else: multiply accumulator with n and recurse
            dup 1           // n acc n
            mul             // n acc·n
            swap 1          // acc·n n
            push -1 add     // acc·n n-1
            swap 1          // n-1 acc·n
            recurse
    );

    let public_input = PublicInput::from([bfe!(1_000)]);
    let non_determinism = NonDeterminism::default();

    let now = Instant::now();
    let (stark, claim, proof) =
        triton_vm::prove_program(&factorial_program, public_input, non_determinism).unwrap();
    println!("Proof generated in: {:.0?}", now.elapsed());

    let proof_bytes = serde_json::to_vec(&proof).unwrap();
    let stark_bytes = serde_json::to_vec(&stark).unwrap();
    let claim_bytes = serde_json::to_vec(&claim).unwrap();

    println!("Proof size: {:?}KB", proof_bytes.len() / 1024);

    // dump Proof, Stark and Claim into file followed by the size
    let mut f = File::create(&output_path).unwrap();
    f.write(&(proof_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write(&proof_bytes).unwrap();
    f.write(&(stark_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write(&stark_bytes).unwrap();
    f.write(&(claim_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write(&claim_bytes).unwrap();
    f.flush().unwrap();

    println!(
        "Stark, claim and proof written to {}",
        output_path.as_path().display()
    );
}

fn verify(output_path: PathBuf) {
    let file = File::open(output_path).unwrap();
    let mut reader = BufReader::new(file);

    let mut len_buf = [0u8; 8];

    // Read proof length
    reader.read_exact(&mut len_buf).unwrap();
    let proof_len = u64::from_le_bytes(len_buf) as usize;
    println!("proof length: {}", proof_len);

    // Read proof
    let mut proof_buf = vec![0u8; proof_len];
    reader.read_exact(&mut proof_buf).unwrap();
    let proof: Proof = serde_json::from_slice(&proof_buf).unwrap();

    // Read Stark len
    reader.read_exact(&mut len_buf).unwrap();
    let stark_len = u64::from_le_bytes(len_buf) as usize;
    println!("stark length: {}", stark_len);

    // Read Stark
    let mut stark_buf = vec![0u8; stark_len];
    reader.read_exact(&mut stark_buf).unwrap();
    let stark: Stark = serde_json::from_slice(&stark_buf).unwrap();

    // Read Claim length
    reader.read_exact(&mut len_buf).unwrap();
    let claim_len = u64::from_le_bytes(len_buf) as usize;
    println!("claim length: {}", claim_len);

    // Read Claim
    let mut claim_buf = vec![0u8; claim_len];
    reader.read_exact(&mut claim_buf).unwrap();
    let claim: Claim = serde_json::from_slice(&claim_buf).unwrap();

    // Verify proof
    let now = Instant::now();
    let verdict = triton_vm::verify(stark, &claim, &proof);
    assert!(verdict);

    println!("Proof verified in: {:?}", now.elapsed());
}
