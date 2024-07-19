use std::fmt::Write as _;
use std::fs;
use std::io::Write;

use anyhow::Context;
use anyhow::Result;
use clap::command;
use clap::Parser;
use serde::Deserialize;
use serde::Serialize;
use tip5::DIGEST_LENGTH;
use triton_vm::prelude::*;

#[derive(Debug, Parser)]
#[command(name = "triton-cli")]
#[command(about = "Compile, prove and verify Triton assembly programs", long_about = None)]
enum CliArg {
    #[command(arg_required_else_help = true)]
    Prove {
        asm_path: String,
        proof_out_path: String,
        #[clap(long)]
        /// Public inputs to pass to the proof, comma separated
        public_inputs: Option<String>,
        #[clap(long)]
        /// Private inputs to pass to the proof, comma separated
        private_inputs: Option<String>,
    },
    #[command(arg_required_else_help = true)]
    Verify { proof_path: String },
}

fn main() -> Result<()> {
    let arg = CliArg::parse();
    match arg {
        CliArg::Prove {
            asm_path,
            proof_out_path,
            public_inputs,
            private_inputs,
        } => prove(&asm_path, &proof_out_path, public_inputs, private_inputs)?,
        CliArg::Verify { proof_path } => verify(&proof_path)?,
    }
    Ok(())
}

fn digest_to_str(d: Digest) -> String {
    let mut hex = String::new();
    d.0.iter()
        .for_each(|v| write!(&mut hex, "{:16x}", u64::from(v)).unwrap());
    format!("0x{hex}")
}

fn verify(proof_path: &str) -> Result<()> {
    let (stark, claim, proof) = read_proof(proof_path).expect("Failed to load proof");

    stark
        .verify(&claim, &proof, &mut None)
        .with_context(|| "Stark proof verification failed")?;

    println!("proof is valid!");
    println!("program digest: {}", digest_to_str(claim.program_digest));
    println!("=================");
    println!("security level: {} bits", stark.security_level);
    println!("FRI expansion factor: {}", stark.fri_expansion_factor);
    println!("trace randomizers: {}", stark.num_trace_randomizers);
    println!("colinearity checks: {}", stark.num_collinearity_checks);
    println!("codeword checks: {}", stark.num_combination_codeword_checks);
    println!("=================");
    println!("public inputs:");
    if claim.input.is_empty() {
        println!("(none)");
    }
    for v in claim.input {
        println!("{v}");
    }
    println!("public outputs:");
    if claim.output.is_empty() {
        println!("(none)");
    }
    for v in claim.output {
        println!("{v}");
    }
    Ok(())
}

fn parse_inputs(inputs: Option<String>) -> Vec<BFieldElement> {
    inputs
        .unwrap_or_default()
        .split(',')
        .filter(|v| !v.is_empty())
        .map(|v| v.parse().unwrap())
        .collect()
}

fn prove(
    asm_filepath: &str,
    out: &str,
    public_inputs: Option<String>,
    private_inputs: Option<String>,
) -> Result<()> {
    if std::path::Path::new(out).exists() {
        anyhow::bail!("output file already exists: {out}");
    }
    let asm = fs::read_to_string(asm_filepath)
        .with_context(|| "Failed to read Triton assembly from file")?;
    let program = triton_program!({ asm });

    let public_input = PublicInput::from(parse_inputs(public_inputs));
    let non_determinism = NonDeterminism::from(parse_inputs(private_inputs));
    println!("proving...");
    let data = triton_vm::prove_program(&program, public_input, non_determinism)
        .with_context(|| "Triton VM errored during program execution")?;
    println!("success!");

    write_proof(data, out).with_context(|| "Failed to write proof to file")?;
    println!("proof written to: {out}");
    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializedProof {
    version: u8,
    security_level: u64,
    fri_expansion_factor: u64,
    num_trace_randomizers: u64,
    num_colinearity_checks: u64,
    num_combination_codeword_checks: u64,
    program_digest: Vec<u64>,
    public_inputs: Vec<u64>,
    public_outputs: Vec<u64>,
    proof: Vec<u64>,
}

fn read_proof(proof_path: &str) -> Result<(Stark, Claim, Proof)> {
    let proof_bytes = fs::read(proof_path)?;
    let serialized_proof: SerializedProof = bincode::deserialize(&proof_bytes)?;
    let stark = Stark {
        security_level: usize::try_from(serialized_proof.security_level)?,
        fri_expansion_factor: usize::try_from(serialized_proof.fri_expansion_factor)?,
        num_trace_randomizers: usize::try_from(serialized_proof.num_trace_randomizers)?,
        num_collinearity_checks: usize::try_from(serialized_proof.num_colinearity_checks)?,
        num_combination_codeword_checks: usize::try_from(
            serialized_proof.num_combination_codeword_checks,
        )?,
    };
    let mut digest = Digest::default();
    assert_eq!(
        DIGEST_LENGTH,
        serialized_proof.program_digest.len(),
        "digest length mismatch!"
    );
    serialized_proof
        .program_digest
        .iter()
        .enumerate()
        .for_each(|(i, x)| digest.0[i] = BFieldElement::new(*x));
    let claim = Claim {
        program_digest: digest,
        input: serialized_proof
            .public_inputs
            .iter()
            .map(|v| BFieldElement::new(*v))
            .collect(),
        output: serialized_proof
            .public_outputs
            .iter()
            .map(|v| BFieldElement::new(*v))
            .collect(),
    };
    let proof_vec = serialized_proof
        .proof
        .iter()
        .map(|v| BFieldElement::new(*v))
        .collect();
    Ok((stark, claim, Proof(proof_vec)))
}

fn write_proof(data: (Stark, Claim, Proof), out: &str) -> Result<()> {
    let (stark, claim, proof) = data;
    let serialized = SerializedProof {
        version: 1,
        security_level: u64::try_from(stark.security_level)?,
        fri_expansion_factor: u64::try_from(stark.fri_expansion_factor)?,
        num_trace_randomizers: u64::try_from(stark.num_trace_randomizers)?,
        num_colinearity_checks: u64::try_from(stark.num_collinearity_checks)?,
        num_combination_codeword_checks: u64::try_from(stark.num_combination_codeword_checks)?,
        program_digest: claim.program_digest.0.iter().map(u64::from).collect(),
        public_inputs: claim.input.iter().map(u64::from).collect(),
        public_outputs: claim.output.iter().map(u64::from).collect(),
        proof: proof.0.iter().map(u64::from).collect(),
    };
    let mut file = fs::File::create_new(out)?;
    let proof_bytes = bincode::serialize(&serialized)?;
    file.write_all(&proof_bytes)?;
    Ok(())
}

#[test]
fn test_serialization() -> Result<()> {
    let asm = "./test-vectors/simple.tasm";
    let proof = "./test-vectors/simple.proof";
    prove(asm, proof, None, None)?;
    verify(&proof)?;
    fs::remove_file(proof)?;
    Ok(())
}
