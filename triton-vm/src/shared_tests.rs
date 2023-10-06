#![cfg(test)]

use std::fs::create_dir_all;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use anyhow::anyhow;
use anyhow::Result;
use proptest::collection::vec;
use proptest::prelude::*;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::shared_math::x_field_element::EXTENSION_DEGREE;

use crate::aet::AlgebraicExecutionTrace;
use crate::profiler::prof_start;
use crate::profiler::prof_stop;
use crate::profiler::TritonProfiler;
use crate::program::Program;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::stark::Stark;
use crate::stark::StarkHasher;
use crate::stark::StarkParameters;
use crate::table::master_table::MasterBaseTable;
use crate::NonDeterminism;
use crate::PublicInput;

prop_compose! {
    pub(crate) fn arbitrary_bfield_element()(value in 0..BFieldElement::P) -> BFieldElement {
        BFieldElement::new(value)
    }
}

prop_compose! {
    pub(crate) fn arbitrary_non_zero_bfield_element()(
        value in 1..BFieldElement::P
    ) -> BFieldElement {
        BFieldElement::new(value)
    }
}

prop_compose! {
    pub(crate) fn arbitrary_xfield_element()(
        coefficients in vec(arbitrary_bfield_element(), EXTENSION_DEGREE)
    ) -> XFieldElement {
        XFieldElement::new(coefficients.try_into().unwrap())
    }
}

prop_compose! {
    pub(crate) fn arbitrary_non_zero_xfield_element()(
        coefficient_0 in arbitrary_non_zero_bfield_element(),
        coefficient_1 in arbitrary_bfield_element(),
        coefficient_2 in arbitrary_bfield_element(),
    ) -> XFieldElement {
        let coefficients = [coefficient_0, coefficient_1, coefficient_2];
        XFieldElement::new(coefficients)
    }
}

prop_compose! {
    pub(crate) fn arbitrary_polynomial()(
        degree in -1_i64..1 << 10,
    )(
        polynomial in arbitrary_polynomial_of_degree(degree),
    ) -> Polynomial<XFieldElement> {
        polynomial
    }
}

prop_compose! {
    pub(crate) fn arbitrary_polynomial_of_degree(degree: i64)(
        leading_coefficient in arbitrary_non_zero_xfield_element(),
        other_coefficients in vec(arbitrary_xfield_element(), degree.try_into().unwrap_or(0)),
    ) -> Polynomial<XFieldElement> {
        let coefficients = match degree >= 0 {
            true => [other_coefficients, vec![leading_coefficient]].concat(),
            false => vec![],
        };
        let polynomial = Polynomial::new(coefficients);
        assert_eq!(degree, polynomial.degree() as i64);
        polynomial
    }
}

/// Prove correct execution of the given program.
/// Return the used parameters and the generated claim & proof.
pub(crate) fn prove_with_low_security_level(
    program: &Program,
    public_input: PublicInput,
    non_determinism: NonDeterminism<BFieldElement>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (StarkParameters, Claim, Proof) {
    prof_start!(maybe_profiler, "trace program");
    let (aet, public_output) = program
        .trace_execution(public_input.clone(), non_determinism)
        .unwrap();
    prof_stop!(maybe_profiler, "trace program");

    let parameters = stark_parameters_with_low_security_level();
    let claim = construct_claim(&aet, public_input.individual_tokens, public_output);

    prof_start!(maybe_profiler, "prove");
    let proof = Stark::prove(parameters, &claim, &aet, maybe_profiler);
    prof_stop!(maybe_profiler, "prove");

    (parameters, claim, proof)
}

pub(crate) fn construct_claim(
    aet: &AlgebraicExecutionTrace,
    public_input: Vec<BFieldElement>,
    public_output: Vec<BFieldElement>,
) -> Claim {
    Claim {
        program_digest: aet.program.hash::<StarkHasher>(),
        input: public_input,
        output: public_output,
    }
}

/// Generate STARK parameters with a low security level.
pub(crate) fn stark_parameters_with_low_security_level() -> StarkParameters {
    let security_level = 32;
    let log_expansion_factor = 2;
    StarkParameters::new(security_level, log_expansion_factor)
}

pub(crate) fn construct_master_base_table(
    parameters: StarkParameters,
    aet: &AlgebraicExecutionTrace,
) -> MasterBaseTable {
    let padded_height = MasterBaseTable::padded_height(aet);
    let fri = Stark::derive_fri(parameters, padded_height);
    let max_degree = Stark::derive_max_degree(padded_height, parameters.num_trace_randomizers);
    let quotient_domain = Stark::quotient_domain(fri.domain, max_degree);
    MasterBaseTable::new(
        aet,
        parameters.num_trace_randomizers,
        quotient_domain,
        fri.domain,
    )
}

/// Program and associated inputs.
pub(crate) struct ProgramAndInput {
    pub program: Program,
    pub public_input: Vec<u64>,
    pub non_determinism: NonDeterminism<u64>,
}

impl ProgramAndInput {
    pub fn without_input(program: Program) -> Self {
        Self {
            program,
            public_input: vec![],
            non_determinism: [].into(),
        }
    }

    pub fn public_input(&self) -> PublicInput {
        self.public_input.clone().into()
    }

    pub fn non_determinism(&self) -> NonDeterminism<BFieldElement> {
        (&self.non_determinism).into()
    }

    /// A thin wrapper around [`Program::run`].
    pub fn run(&self) -> Result<Vec<BFieldElement>> {
        self.program
            .run(self.public_input(), self.non_determinism())
    }
}

pub fn proofs_directory() -> String {
    "proofs/".to_string()
}

pub fn create_proofs_directory() -> Result<()> {
    create_dir_all(proofs_directory()).map_err(|e| anyhow!(e))
}

pub fn proofs_directory_exists() -> bool {
    Path::new(&proofs_directory()).is_dir()
}

pub fn proof_file_exists(filename: &str) -> bool {
    if !proofs_directory_exists() {
        return false;
    }
    let full_filename = format!("{}{filename}", proofs_directory());
    File::open(full_filename).is_ok()
}

pub fn load_proof(filename: &str) -> Result<Proof> {
    let full_filename = format!("{}{filename}", proofs_directory());
    let mut file_content = vec![];
    let mut file_handle = File::open(full_filename)?;
    let num_bytes_read = file_handle.read_to_end(&mut file_content)?;
    println!("Read {num_bytes_read} bytes of proof data from disk.");
    let proof: Proof = bincode::deserialize(&file_content).expect("Cannot deserialize proof.");
    Ok(proof)
}

pub fn save_proof(filename: &str, proof: Proof) -> Result<()> {
    if !proofs_directory_exists() {
        create_proofs_directory()?;
    }
    let full_filename = format!("{}{filename}", proofs_directory());
    let mut file_handle = File::create(full_filename)?;
    let binary = bincode::serialize(&proof)?;
    let amount = file_handle.write(&binary)?;
    println!("Wrote {amount} bytes of proof data to disk.");
    Ok(())
}
