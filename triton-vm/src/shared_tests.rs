use num_traits::Zero;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest_arbitrary_interop::arb;
use test_strategy::Arbitrary;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::error::VMError;
use crate::fri::AuthenticationStructure;
use crate::profiler::prof_start;
use crate::profiler::prof_stop;
use crate::profiler::TritonProfiler;
use crate::program::Program;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::proof_item::FriResponse;
use crate::stark::Stark;
use crate::table::master_table::MasterBaseTable;
use crate::NonDeterminism;
use crate::PublicInput;

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
        assert_eq!(degree, polynomial.degree() as i64);
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

    #[strategy(Just(CpuParallel::from_digests(&#leaves_as_digests).unwrap()))]
    pub merkle_tree: MerkleTree<Tip5>,

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

/// Convenience function to prove correct execution of the given program.
pub(crate) fn prove_with_low_security_level(
    program: &Program,
    public_input: PublicInput,
    non_determinism: NonDeterminism<BFieldElement>,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> (Stark, Claim, Proof) {
    prof_start!(maybe_profiler, "trace program");
    let (aet, public_output) = program
        .trace_execution(public_input.clone(), non_determinism)
        .unwrap();
    prof_stop!(maybe_profiler, "trace program");

    let claim = Claim::about_program(&aet.program)
        .with_input(public_input.individual_tokens)
        .with_output(public_output);

    prof_start!(maybe_profiler, "prove");
    let stark = low_security_stark();
    let proof = stark.prove(&claim, &aet, maybe_profiler).unwrap();
    prof_stop!(maybe_profiler, "prove");

    (stark, claim, proof)
}

pub(crate) fn low_security_stark() -> Stark {
    let security_level = 32;
    let log_expansion_factor = 2;
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
    pub fn run(&self) -> Result<Vec<BFieldElement>, VMError> {
        self.program
            .run(self.public_input(), self.non_determinism())
    }
}
