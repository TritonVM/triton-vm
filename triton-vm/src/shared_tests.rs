use assert2::assert;
use assert2::let_assert;
use isa::program::Program;
use itertools::Itertools;
use num_traits::Zero;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest_arbitrary_adapter::arb;
use rand::Rng;
use rand::SeedableRng;
use rand::prelude::StdRng;
use test_strategy::Arbitrary;
use twenty_first::prelude::*;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::challenges::Challenges;
use crate::error::VMError;
use crate::low_degree_test::stir::StirResponse;
use crate::prelude::*;
use crate::profiler::profiler;
use crate::proof_item::AuthenticationStructure;
use crate::stark::ProverDomains;
use crate::table::master_table::MasterAuxTable;
use crate::table::master_table::MasterMainTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Arbitrary)]
#[filter(!#self.0.is_zero())]
struct NonZeroXFieldElement(#[strategy(arb())] XFieldElement);

prop_compose! {
    pub(crate) fn arbitrary_polynomial()(
        degree in -1_i64..1 << 10,
    )(
        polynomial in arbitrary_polynomial_of_degree(degree),
    ) -> Polynomial<'static, XFieldElement> {
        polynomial
    }
}

prop_compose! {
    pub(crate) fn arbitrary_polynomial_of_degree(degree: i64)(
        leading_coefficient: NonZeroXFieldElement,
        other_coefficients in vec(arb(), degree.try_into().unwrap_or(0)),
    ) -> Polynomial<'static, XFieldElement> {
        let mut coefficients = other_coefficients;
        if degree >= 0 {
            coefficients.push(leading_coefficient.0);
        }
        let polynomial = Polynomial::new(coefficients);
        assert!(degree == polynomial.degree() as i64);

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

    #[strategy(Just(MerkleTree::par_new(&#leaves_as_digests).unwrap()))]
    pub merkle_tree: MerkleTree,

    #[strategy(Just(#merkle_tree.authentication_structure(&#revealed_indices).unwrap()))]
    pub auth_structure: AuthenticationStructure,
}

impl LeavedMerkleTreeTestData {
    pub fn root(&self) -> Digest {
        self.merkle_tree.root()
    }

    /// A polynomial interpolating all the tree's leaves.
    pub fn interpolated_leaves(&self) -> Polynomial<'static, XFieldElement> {
        let domain = ArithmeticDomain::of_length(self.num_leaves()).unwrap();

        domain.interpolate(&self.leaves)
    }

    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn into_stir_response(self) -> StirResponse {
        // use an implicit folding factor of 1, which is illegal in STIR but
        // doesn't matter for the testing purposes here
        let queried_leafs = self
            .revealed_indices
            .iter()
            .map(|&i| vec![self.leaves[i]])
            .collect();

        StirResponse {
            queried_leafs,
            auth_structure: self.auth_structure,
        }
    }
}

/// Program and associated inputs, as well as parameters with which to prove
/// correct execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct TestableProgram {
    pub program: Program,
    pub public_input: PublicInput,
    pub non_determinism: NonDeterminism,
    pub stark: Stark,
}

impl TestableProgram {
    pub fn new(program: Program) -> Self {
        Self {
            program,
            public_input: PublicInput::default(),
            non_determinism: NonDeterminism::default(),
            stark: Stark::low_security(),
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

    #[must_use]
    pub fn use_stark(mut self, stark: Stark) -> Self {
        self.stark = stark;
        self
    }

    pub fn public_input(&self) -> PublicInput {
        self.public_input.clone()
    }

    pub fn non_determinism(&self) -> NonDeterminism {
        self.non_determinism.clone()
    }

    /// A thin wrapper around [`VM::run`].
    pub fn run(self) -> Result<Vec<BFieldElement>, VMError> {
        let public_input = self.public_input();
        let non_determinism = self.non_determinism();
        VM::run(self.program, public_input, non_determinism)
    }

    /// Prove correct execution of the program, then verify said proof.
    ///
    /// Also print the [`VMPerformanceProfile`][profile] for both proving and
    /// verification to standard out.
    ///
    /// [profile]: crate::profiler::VMPerformanceProfile
    pub fn prove_and_verify(self) {
        let Self {
            program,
            public_input,
            non_determinism,
            stark,
        } = self;

        crate::profiler::start("");
        profiler!(start "Pre-flight");
        let claim = Claim::about_program(&program).with_input(public_input.clone());
        let (aet, public_output) =
            VM::trace_execution(program, public_input, non_determinism).unwrap();
        let claim = claim.with_output(public_output);
        profiler!(stop "Pre-flight");

        profiler!(start "Prove");
        let proof = stark.prove(&claim, &aet).unwrap();
        profiler!(stop "Prove");

        profiler!(start "Verify");
        assert!(let Ok(()) = stark.verify(&claim, &proof));
        profiler!(stop "Verify");
        let profile = crate::profiler::finish();

        let_assert!(Ok(padded_height) = proof.padded_height());
        assert!(aet.padded_height() == padded_height);

        let ldt = stark.ldt(padded_height).unwrap();
        let profile = profile
            .with_cycle_count(aet.height_of_table(TableId::Processor))
            .with_padded_height(padded_height)
            .with_ldt_domain_len(ldt.initial_domain().len());
        println!("{profile}");
    }

    pub fn generate_proof_artifacts(self) -> ProofArtifacts {
        let Self {
            program,
            public_input,
            non_determinism,
            stark,
        } = self;

        let claim = Claim::about_program(&program).with_input(public_input.clone());
        let (aet, stdout) = VM::trace_execution(program, public_input, non_determinism).unwrap();
        let claim = claim.with_output(stdout);

        // construct master main table
        let padded_height = aet.padded_height();
        let ldt = stark.ldt(padded_height).unwrap();
        let num_trace_randomizers = Stark::num_trace_randomizers(&*ldt);
        let ldt_domain = ldt.initial_domain();
        let max_degree = stark.max_degree(padded_height, num_trace_randomizers);
        let domains =
            ProverDomains::derive(padded_height, num_trace_randomizers, ldt_domain, max_degree);

        let mut master_main_table = MasterMainTable::new(
            &aet,
            domains,
            num_trace_randomizers,
            StdRng::seed_from_u64(6718321586953195571).random(),
        );
        master_main_table.pad();
        let master_main_table = master_main_table;

        let challenges = Challenges::placeholder(&claim);
        let master_aux_table = master_main_table.extend(&challenges);

        ProofArtifacts {
            claim,
            master_main_table,
            master_aux_table,
            challenges,
        }
    }
}

/// Various intermediate artifacts required for proof generation.
#[must_use]
pub struct ProofArtifacts {
    pub claim: Claim,
    pub master_main_table: MasterMainTable,
    pub master_aux_table: MasterAuxTable,
    pub challenges: Challenges,
}

/// Test helper struct for corrupting digests. Primarily used for negative tests.
#[derive(Debug, Clone, PartialEq, Eq, test_strategy::Arbitrary)]
pub(crate) struct DigestCorruptor {
    #[strategy(vec(0..Digest::LEN, 1..=Digest::LEN))]
    #[filter(#corrupt_indices.iter().all_unique())]
    corrupt_indices: Vec<usize>,

    #[strategy(vec(arb(), #corrupt_indices.len()))]
    corrupt_elements: Vec<BFieldElement>,
}

impl DigestCorruptor {
    pub fn corrupt(&self, digest: &mut Digest) -> Result<(), TestCaseError> {
        let original_digest = *digest;
        for (&i, &element) in self.corrupt_indices.iter().zip(&self.corrupt_elements) {
            digest.0[i] = element;
        }
        if *digest == original_digest {
            let reject_reason = "corruption must change digest".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        Ok(())
    }
}
