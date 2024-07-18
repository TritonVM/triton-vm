use std::fmt::Write as _;
use std::fs;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;

use anyhow::Context;
use anyhow::Result;
use clap::command;
use clap::Parser;
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
        CliArg::Verify { proof_path } => verify(&proof_path),
    }
    Ok(())
}

fn digest_to_str(d: Digest) -> String {
    let mut hex = String::new();
    d.0.iter()
        .for_each(|v| write!(&mut hex, "{:16x}", u64::from(v)).unwrap());
    format!("0x{hex}")
}

fn verify(proof_path: &str) {
    let (stark, claim, proof) = read_proof(proof_path).expect("Failed to load proof");

    let verdict = triton_vm::verify(stark, &claim, &proof);
    if !verdict {
        println!("Proof is not valid!");
        std::process::exit(1);
    }
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
        println!("output file already exists: {out}");
        std::process::exit(1);
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

    // write the proof in an arbitrary binary format
    //
    // version: u8
    //
    // security_level: u64
    // fri_expansion_factor: u64
    // num_trace_randomizers: u64
    // num_colinearity_checks: u64
    // num_combination_codeword_checks: u64
    //
    // digest_length: u8
    // public_input_length: u64
    // public_output_length: u64
    // proof_length: u64
    //
    // program_digest: [u64; digest_length]
    //
    // public_inputs: u64[public_input_length]
    // public_outputs: u64[public_output_length]
    // proof: u64[proof_length]
    write_proof(data, out).with_context(|| "Failed to write proof to file")?;
    println!("proof written to: {out}");
    Ok(())
}

fn read_proof(proof_path: &str) -> Result<(Stark, Claim, Proof)> {
    let file = fs::File::open(proof_path)?;
    let mut reader = BufReader::new(file);
    let version = read_u8(&mut reader)?;
    assert!(version == 1, "wrong proof file version!");
    let stark = Stark {
        security_level: read_usize(&mut reader)?,
        fri_expansion_factor: read_usize(&mut reader)?,
        num_trace_randomizers: read_usize(&mut reader)?,
        num_collinearity_checks: read_usize(&mut reader)?,
        num_combination_codeword_checks: read_usize(&mut reader)?,
    };
    let digest_len = read_u8(&mut reader)?;
    let input_len = read_u64(&mut reader)?;
    let output_len = read_u64(&mut reader)?;
    let proof_len = read_u64(&mut reader)?;
    let mut d = Digest::default();
    assert_eq!(
        d.0.len(),
        usize::from(digest_len),
        "digest length mismatch!"
    );
    read_u64_vec(&mut reader, digest_len.into())?
        .iter()
        .enumerate()
        .for_each(|(i, x)| d.0[i] = BFieldElement::new(*x));
    let claim = Claim {
        program_digest: d,
        input: read_u64_vec(&mut reader, input_len)?
            .iter()
            .map(|v| BFieldElement::new(*v))
            .collect(),
        output: read_u64_vec(&mut reader, output_len)?
            .iter()
            .map(|v| BFieldElement::new(*v))
            .collect(),
    };
    let proof_vec = read_u64_vec(&mut reader, proof_len)?
        .iter()
        .map(|v| BFieldElement::new(*v))
        .collect();
    Ok((stark, claim, Proof(proof_vec)))
}

fn write_proof(data: (Stark, Claim, Proof), out: &str) -> Result<()> {
    let (stark, claim, proof) = data;
    let mut file = fs::File::create_new(out)?;
    file.write_all(&[1])?; // write the version
                           // fails on systems with usize > 64 bits
    file.write_all(&u64::try_from(stark.security_level)?.to_le_bytes())?;
    file.write_all(&u64::try_from(stark.fri_expansion_factor)?.to_le_bytes())?;
    file.write_all(&u64::try_from(stark.num_trace_randomizers)?.to_le_bytes())?;
    file.write_all(&u64::try_from(stark.num_collinearity_checks)?.to_le_bytes())?;
    file.write_all(&u64::try_from(stark.num_combination_codeword_checks)?.to_le_bytes())?;

    file.write_all(&[u8::try_from(claim.program_digest.0.len())?])?;
    file.write_all(&u64::try_from(claim.input.len())?.to_le_bytes())?;
    file.write_all(&u64::try_from(claim.output.len())?.to_le_bytes())?;
    file.write_all(&u64::try_from(proof.0.len())?.to_le_bytes())?;
    for v in claim.program_digest.0 {
        file.write_all(&u64::from(&v).to_le_bytes())?;
    }
    for v in claim.input {
        file.write_all(&u64::from(&v).to_le_bytes())?;
    }
    for v in claim.output {
        file.write_all(&u64::from(&v).to_le_bytes())?;
    }
    for v in proof.0 {
        file.write_all(&u64::from(&v).to_le_bytes())?;
    }

    Ok(())
}

fn read_u64_vec<T: Read>(reader: &mut T, len: u64) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for _ in 0..len {
        out.push(read_u64(reader)?);
    }
    Ok(out)
}

fn read_u64<T: Read>(reader: &mut T) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u8<T: Read>(reader: &mut T) -> Result<u8> {
    let mut bytes = [0];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_usize<T: Read>(reader: &mut T) -> Result<usize> {
    Ok(usize::try_from(read_u64(reader)?)?)
}

#[test]
fn test_serialization() -> Result<()> {
    let asm = "./test-vectors/simple.tasm".to_string();
    let proof = "./test-vectors/simple.proof".to_string();
    prove(&asm, &proof, None, None)?;
    verify(&proof);
    fs::remove_file(proof).unwrap();
    Ok(())
}
