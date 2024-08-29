use assert2::assert;
use assert2::let_assert;
use isa::program::Program;
use num_traits::Zero;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest_arbitrary_interop::arb;
use test_strategy::Arbitrary;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::error::VMError;
use crate::fri::AuthenticationStructure;
use crate::prelude::*;
use crate::profiler::profiler;
use crate::proof_item::FriResponse;
use crate::table::master_table::MasterBaseTable;

pub(crate) const DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS: usize = 2;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Arbitrary)]
#[filter(!#self.0.is_zero())]
struct NonZeroXFieldElement(#[strategy(arb())] XFieldElement);

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
        leading_coefficient: NonZeroXFieldElement,
        other_coefficients in vec(arb(), degree.try_into().unwrap_or(0)),
    ) -> Polynomial<XFieldElement> {
        let leading_coefficient = leading_coefficient.0;
        let coefficients = match degree >= 0 {
            true => [other_coefficients, vec![leading_coefficient]].concat(),
            false => vec![],
        };
        let polynomial = Polynomial::new(coefficients);
        assert!(degree== polynomial.degree() as i64);
        polynomial
    }
}

#[derive(Debug, Clone, test_strategy::Arbitrary)]
pub(crate) struct LeavedMerkleTreeTestData {
    #[strategy(1..=10_usize)]
    pub _tree_height: usize,

    #[strategy(vec(arb(), 1 << #_tree_height))]
    pub leaves: Vec<XFieldElement>,

    #[strategy(vec(0..#leaves.len(), 1..=#leaves.len()))]
    pub revealed_indices: Vec<usize>,

    #[strategy(Just(#leaves.iter().map(|&x| x.into()).collect()))]
    pub leaves_as_digests: Vec<Digest>,

    #[strategy(Just(MerkleTree::new::<CpuParallel>(&#leaves_as_digests).unwrap()))]
    pub merkle_tree: MerkleTree,

    #[strategy(Just(#revealed_indices.iter().map(|&i| #leaves[i]).collect()))]
    pub revealed_leaves: Vec<XFieldElement>,

    #[strategy(Just(#merkle_tree.authentication_structure(&#revealed_indices).unwrap()))]
    pub auth_structure: AuthenticationStructure,
}

impl LeavedMerkleTreeTestData {
    pub fn root(&self) -> Digest {
        self.merkle_tree.root()
    }

    pub fn leaves(&self) -> &[XFieldElement] {
        &self.leaves
    }

    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn into_fri_response(self) -> FriResponse {
        FriResponse {
            auth_structure: self.auth_structure,
            revealed_leaves: self.revealed_leaves,
        }
    }
}

/// Prove correct execution of the supplied program, then verify said proof.
/// Also print the [`VMPerformanceProfile`][profile] for both proving and
/// verification to standard out.
///
/// [profile]: crate::profiler::VMPerformanceProfile
pub(crate) fn prove_and_verify(
    program_and_input: ProgramAndInput,
    log_2_fri_expansion_factor: usize,
) {
    let ProgramAndInput {
        program,
        public_input,
        non_determinism,
    } = program_and_input;

    profiler!(start "Pre-flight");
    let (aet, public_output) =
        VM::trace_execution(&program, public_input.clone(), non_determinism.clone()).unwrap();

    let claim = Claim::about_program(&program)
        .with_input(public_input.individual_tokens.clone())
        .with_output(public_output);
    let stark = low_security_stark(log_2_fri_expansion_factor);
    profiler!(stop "Pre-flight");

    profiler!(start "Prove");
    let proof = stark.prove(&claim, &aet).unwrap();
    profiler!(stop "Prove");

    profiler!(start "Verify");
    assert!(let Ok(()) = stark.verify(&claim, &proof));
    profiler!(stop "Verify");
    let profile = crate::profiler::finish();

    let_assert!(Ok(padded_height) = proof.padded_height());
    let fri = stark.derive_fri(padded_height).unwrap();
    let profile = profile
        .with_padded_height(padded_height)
        .with_fri_domain_len(fri.domain.length);
    println!("{profile}");
}

pub(crate) fn low_security_stark(log_expansion_factor: usize) -> Stark {
    let security_level = 32;
    Stark::new(security_level, log_expansion_factor)
}

pub(crate) fn construct_master_base_table(
    stark: Stark,
    aet: &AlgebraicExecutionTrace,
) -> MasterBaseTable {
    let padded_height = aet.padded_height();
    let fri = stark.derive_fri(padded_height).unwrap();
    let max_degree = stark.derive_max_degree(padded_height);
    let quotient_domain = Stark::quotient_domain(fri.domain, max_degree).unwrap();
    MasterBaseTable::new(
        aet,
        stark.num_trace_randomizers,
        quotient_domain,
        fri.domain,
    )
}

/// Program and associated inputs.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ProgramAndInput {
    pub program: Program,
    pub public_input: PublicInput,
    pub non_determinism: NonDeterminism,
}

impl ProgramAndInput {
    pub fn new(program: Program) -> Self {
        Self {
            program,
            public_input: PublicInput::default(),
            non_determinism: NonDeterminism::default(),
        }
    }

    #[must_use]
    pub fn with_input<PI: Into<PublicInput>>(mut self, public_input: PI) -> Self {
        self.public_input = public_input.into();
        self
    }

    #[must_use]
    pub fn with_non_determinism<ND: Into<NonDeterminism>>(mut self, non_determinism: ND) -> Self {
        self.non_determinism = non_determinism.into();
        self
    }

    pub fn public_input(&self) -> PublicInput {
        self.public_input.clone()
    }

    pub fn non_determinism(&self) -> NonDeterminism {
        self.non_determinism.clone()
    }

    /// A thin wrapper around [`VM::run`].
    pub fn run(&self) -> Result<Vec<BFieldElement>, VMError> {
        VM::run(&self.program, self.public_input(), self.non_determinism())
    }
}
