use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::Add;

use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::ArrayView2;
use ndarray::Axis;
use num_traits::One;
use rayon::prelude::*;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::roundup_npo2;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use triton_profiler::prof_itr0;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::polynomial::Polynomial;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::fri::Fri;
use crate::fri::FriValidationError;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::challenges::Challenges;
use crate::table::master_table::*;
use crate::vm::AlgebraicExecutionTrace;

pub type StarkHasher = Tip5;
pub type Maker = CpuParallel;
pub type StarkProofStream = ProofStream<ProofItem, StarkHasher>;

pub struct StarkParameters {
    pub security_level: usize,
    pub fri_expansion_factor: usize,
    pub num_trace_randomizers: usize,
    pub num_randomizer_polynomials: usize,
    pub num_colinearity_checks: usize,
    pub num_combination_codeword_checks: usize,
}

impl StarkParameters {
    pub fn new(security_level: usize, log2_of_fri_expansion_factor: usize) -> Self {
        assert_ne!(
            0, log2_of_fri_expansion_factor,
            "FRI expansion factor must be greater than one."
        );

        let num_randomizer_polynomials = 1; // over the XField
        let fri_expansion_factor = 1 << log2_of_fri_expansion_factor;
        let num_colinearity_checks = security_level / log2_of_fri_expansion_factor;
        let num_trace_randomizers = num_colinearity_checks * 2;
        let num_combination_codeword_checks = security_level;

        StarkParameters {
            security_level,
            fri_expansion_factor,
            num_trace_randomizers,
            num_randomizer_polynomials,
            num_colinearity_checks,
            num_combination_codeword_checks,
        }
    }
}

impl Default for StarkParameters {
    fn default() -> Self {
        let log_2_of_fri_expansion_factor = 2;
        let security_level = 160;

        Self::new(security_level, log_2_of_fri_expansion_factor)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum StarkValidationError {
    QuotientElementInequality,
    CombinationLeafInequality,
    PaddedHeightInequality,
    FriValidationError(FriValidationError),
}

impl Error for StarkValidationError {}

impl fmt::Display for StarkValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "STARK error: {self:?}")
    }
}

pub struct Stark {
    pub parameters: StarkParameters,
    pub claim: Claim,
    pub max_degree: Degree,
    pub interpolant_degree: Degree,
    pub fri: Fri<StarkHasher>,
}

impl Stark {
    pub fn new(claim: Claim, parameters: StarkParameters) -> Self {
        let interpolant_degree =
            interpolant_degree(claim.padded_height, parameters.num_trace_randomizers);
        let max_degree_with_origin =
            max_degree_with_origin(interpolant_degree, claim.padded_height);
        let max_degree = (roundup_npo2(max_degree_with_origin.degree as u64) - 1) as Degree;
        let fri_domain_length = parameters.fri_expansion_factor * (max_degree as usize + 1);
        let coset_offset = BFieldElement::generator();
        let fri = Fri::new(
            coset_offset,
            fri_domain_length,
            parameters.fri_expansion_factor,
            parameters.num_colinearity_checks,
        );
        Self {
            parameters,
            claim,
            max_degree,
            interpolant_degree,
            fri,
        }
    }

    pub fn prove(
        &self,
        aet: AlgebraicExecutionTrace,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Proof {
        prof_start!(maybe_profiler, "base tables");
        prof_start!(maybe_profiler, "create");
        let mut master_base_table =
            MasterBaseTable::new(aet, self.parameters.num_trace_randomizers, self.fri.domain);
        prof_stop!(maybe_profiler, "create");

        prof_start!(maybe_profiler, "pad");
        master_base_table.pad();
        prof_stop!(maybe_profiler, "pad");

        prof_start!(maybe_profiler, "LDE");
        master_base_table.randomize_trace();
        let (fri_domain_master_base_table, base_interpolation_polys) =
            master_base_table.to_fri_domain_table();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree");
        let base_merkle_tree = fri_domain_master_base_table.merkle_tree(maybe_profiler);
        let base_merkle_tree_root = base_merkle_tree.get_root();
        prof_stop!(maybe_profiler, "Merkle tree");

        prof_start!(maybe_profiler, "Fiat-Shamir");
        let padded_height = BFieldElement::new(master_base_table.padded_height as u64);
        let mut proof_stream = StarkProofStream::new();
        proof_stream.enqueue(&ProofItem::PaddedHeight(padded_height));
        proof_stream.enqueue(&ProofItem::MerkleRoot(base_merkle_tree_root));
        let extension_weights = proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let extension_challenges =
            Challenges::new(extension_weights, &self.claim.input, &self.claim.output);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "extend");
        let mut master_ext_table = master_base_table.extend(
            &extension_challenges,
            self.parameters.num_randomizer_polynomials,
        );
        prof_stop!(maybe_profiler, "extend");
        prof_stop!(maybe_profiler, "base tables");

        prof_start!(maybe_profiler, "ext tables");
        prof_start!(maybe_profiler, "LDE");
        master_ext_table.randomize_trace();
        let (fri_domain_ext_master_table, ext_interpolation_polys) =
            master_ext_table.to_fri_domain_table();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree");
        let ext_merkle_tree = fri_domain_ext_master_table.merkle_tree(maybe_profiler);
        let ext_merkle_tree_root = ext_merkle_tree.get_root();
        proof_stream.enqueue(&ProofItem::MerkleRoot(ext_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "ext tables");

        prof_start!(maybe_profiler, "out-of-domain rows");
        let trace_domain_generator = derive_domain_generator(padded_height.value());
        let out_of_domain_curr_row_index = proof_stream.sample_scalars(1)[0];
        let out_of_domain_next_row_index = trace_domain_generator * out_of_domain_curr_row_index;

        prof_start!(maybe_profiler, "lift base polys");
        let base_interpolation_polys = base_interpolation_polys
            .map(|poly| Polynomial::new(poly.coefficients.iter().map(|b| b.lift()).collect_vec()));
        // ignore randomizer codewords / polynomials
        let ext_interpolation_polys = ext_interpolation_polys.slice(s![..NUM_EXT_COLUMNS]);
        prof_stop!(maybe_profiler, "lift base polys");
        let out_of_domain_curr_base_row =
            base_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_curr_row_index));
        let out_of_domain_next_base_row =
            base_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_next_row_index));
        let out_of_domain_curr_ext_row =
            ext_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_curr_row_index));
        let out_of_domain_next_ext_row =
            ext_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_next_row_index));
        proof_stream.enqueue(&ProofItem::OutOfDomainBaseRow(
            out_of_domain_curr_base_row.to_vec(),
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainExtRow(
            out_of_domain_curr_ext_row.to_vec(),
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainBaseRow(
            out_of_domain_next_base_row.to_vec(),
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainExtRow(
            out_of_domain_next_ext_row.to_vec(),
        ));
        // todo: actually use the out-of-domain rows
        prof_stop!(maybe_profiler, "out-of-domain rows");

        prof_start!(maybe_profiler, "quotient-domain codewords");
        let trace_domain = ArithmeticDomain::new_no_offset(master_base_table.padded_height);
        let quotient_domain = self.quotient_domain();
        let unit_distance = self.fri.domain.length / quotient_domain.length;
        let base_quotient_domain_codewords = fri_domain_master_base_table
            .master_base_matrix
            .slice(s![..; unit_distance, ..]);
        let extension_quotient_domain_codewords = fri_domain_ext_master_table
            .master_ext_matrix
            .slice(s![..; unit_distance, ..]);
        prof_stop!(maybe_profiler, "quotient-domain codewords");

        prof_start!(maybe_profiler, "quotient codewords");
        let master_quotient_table = all_quotients(
            base_quotient_domain_codewords,
            extension_quotient_domain_codewords,
            trace_domain,
            quotient_domain,
            &extension_challenges,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "quotient codewords");

        prof_start!(maybe_profiler, "linearly combine quotient codewords");
        // Create quotient codeword. This is a part of the combination codeword. To reduce the
        // amount of hashing necessary, the quotient codeword is linearly summed instead of
        // hashed prior to committing to it.
        let quotient_combination_weights =
            Array1::from(proof_stream.sample_scalars(num_all_table_quotients()));
        let quotient_codeword = self.create_quotient_codeword(
            quotient_domain,
            master_quotient_table,
            quotient_combination_weights,
        );
        prof_stop!(maybe_profiler, "linearly combine quotient codewords");
        debug_assert_eq!(quotient_domain.length, quotient_codeword.len());

        prof_start!(maybe_profiler, "commit to quotient codeword");
        prof_start!(maybe_profiler, "LDE");
        let quotient_interpolation_poly = quotient_domain.interpolate(&quotient_codeword);
        let fri_quotient_codeword =
            Array1::from(self.fri.domain.evaluate(&quotient_interpolation_poly));
        prof_stop!(maybe_profiler, "LDE");
        let fri_quotient_codeword_digests = fri_quotient_codeword
            .iter()
            .map(|&x| x.into())
            .collect_vec();
        let quot_merkle_tree: MerkleTree<StarkHasher, _> =
            Maker::from_digests(&fri_quotient_codeword_digests);
        let quot_merkle_tree_root = quot_merkle_tree.get_root();
        proof_stream.enqueue(&ProofItem::MerkleRoot(quot_merkle_tree_root));

        let out_of_domain_quotient_element =
            quotient_interpolation_poly.evaluate(&out_of_domain_curr_row_index);
        proof_stream.enqueue(&ProofItem::OutOfDomainElement(
            out_of_domain_quotient_element,
        ));
        prof_stop!(maybe_profiler, "commit to quotient codeword");
        debug_assert_eq!(self.fri.domain.length, quot_merkle_tree.get_leaf_count());

        // Get weights for remainder of the combination codeword.
        prof_start!(maybe_profiler, "Fiat-Shamir");
        let num_base_and_ext_codeword_weights = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS;
        let base_and_ext_codeword_weights =
            proof_stream.sample_scalars(num_base_and_ext_codeword_weights);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "create base&ext codeword");
        prof_start!(maybe_profiler, "linear combination");
        let base_and_ext_codeword = self.create_base_and_ext_codeword(
            quotient_domain,
            base_quotient_domain_codewords,
            extension_quotient_domain_codewords.slice(s![.., ..NUM_EXT_COLUMNS]),
            base_and_ext_codeword_weights,
        );
        prof_stop!(maybe_profiler, "linear combination");

        prof_start!(maybe_profiler, "LDE 3");
        let base_and_ext_interpolation_poly = quotient_domain.interpolate(&base_and_ext_codeword);
        let fri_base_and_ext_codeword =
            Array1::from(self.fri.domain.evaluate(&base_and_ext_interpolation_poly));

        let out_of_domain_curr_row_base_and_ext_element =
            base_and_ext_interpolation_poly.evaluate(&out_of_domain_curr_row_index);
        let out_of_domain_next_row_base_and_ext_element =
            base_and_ext_interpolation_poly.evaluate(&out_of_domain_next_row_index);
        proof_stream.enqueue(&ProofItem::OutOfDomainElement(
            out_of_domain_curr_row_base_and_ext_element,
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainElement(
            out_of_domain_next_row_base_and_ext_element,
        ));
        prof_stop!(maybe_profiler, "LDE 3");
        prof_stop!(maybe_profiler, "create base&ext codeword");

        prof_start!(maybe_profiler, "combination codeword");
        // todo: apply DEEP trick, correctly build codeword to feed to FRI
        let fri_combination_codeword = fri_base_and_ext_codeword + fri_quotient_codeword.clone();
        debug_assert_eq!(self.fri.domain.length, fri_combination_codeword.len());
        let fri_combination_codeword = fri_domain_ext_master_table
            .randomizer_polynomials()
            .into_iter()
            .fold(fri_combination_codeword, ArrayBase::add)
            .to_vec();
        prof_stop!(maybe_profiler, "combination codeword");

        prof_start!(maybe_profiler, "FRI");
        let (revealed_current_row_indices, _) =
            self.fri.prove(&fri_combination_codeword, &mut proof_stream);
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "open trace leafs");
        // the relation between the FRI domain and the trace domain
        let unit_distance = self.fri.domain.length / master_base_table.padded_height;
        // Open leafs of zipped codewords at indicated positions
        let revealed_current_and_next_row_indices = self
            .revealed_current_and_next_row_indices(unit_distance, &revealed_current_row_indices);

        let revealed_base_elems = Self::get_revealed_elements(
            fri_domain_master_base_table.master_base_matrix.view(),
            &revealed_current_and_next_row_indices,
        );
        let auth_paths_base =
            base_merkle_tree.get_authentication_structure(&revealed_current_and_next_row_indices);
        proof_stream.enqueue(&ProofItem::MasterBaseTableRows(revealed_base_elems));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_base));

        let revealed_ext_elems = Self::get_revealed_elements(
            fri_domain_ext_master_table.master_ext_matrix.view(),
            &revealed_current_and_next_row_indices,
        );
        let auth_paths_ext =
            ext_merkle_tree.get_authentication_structure(&revealed_current_and_next_row_indices);
        proof_stream.enqueue(&ProofItem::MasterExtTableRows(revealed_ext_elems));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_ext));

        // Open quotient & combination codewords at the same positions as base & ext codewords.
        // Use `revealed_current_row_indices`, not `revealed_current_and_next_row_indices`, as
        // the latter is only needed to check transition constraints.
        let revealed_quotient_elements = revealed_current_row_indices
            .iter()
            .map(|&i| fri_quotient_codeword[i])
            .collect_vec();
        let revealed_quotient_auth_paths =
            quot_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(&ProofItem::RevealedCombinationElements(
            revealed_quotient_elements,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(
            revealed_quotient_auth_paths,
        ));
        prof_stop!(maybe_profiler, "open trace leafs");

        #[cfg(debug_assertions)]
        {
            let transcript_length = proof_stream.transcript_length();
            let kib = (transcript_length * 8 / 1024) + 1;
            println!("Created proof containing {transcript_length} B-field elements ({kib} kiB).");
        }

        proof_stream.to_proof()
    }

    fn quotient_domain(&self) -> ArithmeticDomain {
        // When debugging, it is useful to check the degree of some intermediate polynomials.
        // The quotient domain is chosen to be _just_ large enough to perform all the necessary
        // computations on polynomials. Concretely, the maximal degree of a polynomial over the
        // quotient domain is at most only slightly larger than the maximal degree allowed in the
        // STARK proof, and could be equal. This makes computation for the prover much faster.
        // However, it can also make it impossible to check if some operation (e.g., dividing out
        // the zerofier) has (erroneously) increased the polynomial's degree beyond the allowed
        // maximum.
        if cfg!(debug_assertions) {
            self.fri.domain
        } else {
            let offset = self.fri.domain.offset;
            let length = roundup_npo2(self.max_degree as u64);
            ArithmeticDomain::new(offset, length as usize)
        }
    }

    fn revealed_current_and_next_row_indices(
        &self,
        unit_distance: usize,
        revealed_current_rows_indices: &[usize],
    ) -> Vec<usize> {
        let mut indices = vec![];
        for &index in revealed_current_rows_indices.iter() {
            indices.push(index);
            indices.push((index + unit_distance) % self.fri.domain.length);
        }
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    fn get_revealed_elements<FF: FiniteField>(
        master_matrix: ArrayView2<FF>,
        revealed_indices: &[usize],
    ) -> Vec<Vec<FF>> {
        revealed_indices
            .iter()
            .map(|&idx| master_matrix.row(idx).to_vec())
            .collect_vec()
    }

    fn create_quotient_codeword(
        &self,
        quotient_domain: ArithmeticDomain,
        quotient_codewords: Array2<XFieldElement>,
        quotient_weights: Array1<XFieldElement>,
    ) -> Vec<XFieldElement> {
        assert_eq!(quotient_weights.len(), quotient_codewords.ncols());

        let weighted_codewords = quotient_codewords * quotient_weights;
        let quotient_codeword = weighted_codewords.sum_axis(Axis(1)).to_vec();

        assert_eq!(quotient_domain.length, quotient_codeword.len());
        #[cfg(debug_assertions)]
        self.debug_check_degree(&quotient_codeword, quotient_domain);
        quotient_codeword
    }

    fn create_base_and_ext_codeword(
        &self,
        quotient_domain: ArithmeticDomain,
        base_codewords: ArrayView2<BFieldElement>,
        extension_codewords: ArrayView2<XFieldElement>,
        weights: Vec<XFieldElement>,
    ) -> Vec<XFieldElement> {
        let (base_weights, ext_weights) = weights.split_at(NUM_BASE_COLUMNS);
        let base_weights = Array1::from(base_weights.to_vec());
        let ext_weights = Array1::from(ext_weights.to_vec());

        assert_eq!(base_weights.len(), base_codewords.ncols());
        assert_eq!(ext_weights.len(), extension_codewords.ncols());

        let weighted_base_codewords = &base_codewords * base_weights;
        let weighted_ext_codewords = &extension_codewords * &ext_weights;

        let base_and_ext_codeword =
            weighted_base_codewords.sum_axis(Axis(1)) + weighted_ext_codewords.sum_axis(Axis(1));
        let base_and_ext_codeword = base_and_ext_codeword.to_vec();

        assert_eq!(quotient_domain.length, base_and_ext_codeword.len());
        #[cfg(debug_assertions)]
        self.debug_check_degree(&base_and_ext_codeword, quotient_domain);
        base_and_ext_codeword
    }

    #[cfg(debug_assertions)]
    fn debug_check_degree(
        &self,
        combination_codeword: &[XFieldElement],
        quotient_domain: ArithmeticDomain,
    ) {
        let max_degree = self.max_degree as isize;
        let degree = quotient_domain.interpolate(combination_codeword).degree();
        let maybe_excl_mark = match degree > max_degree {
            true => "!",
            false => " ",
        };
        println!(
            "{maybe_excl_mark} Combination codeword has degree {degree}. \
            Must be of maximal degree {max_degree}."
        );
    }

    pub fn verify(
        &self,
        proof: Proof,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<bool> {
        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = StarkProofStream::from_proof(&proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir 1");
        let padded_height = proof_stream.dequeue()?.as_padded_heights()?.value() as usize;
        if self.claim.padded_height != padded_height {
            bail!(StarkValidationError::PaddedHeightInequality);
        }
        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;

        let extension_challenge_weights =
            proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let challenges = Challenges::new(
            extension_challenge_weights,
            &self.claim.input,
            &self.claim.output,
        );
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue");
        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        prof_stop!(maybe_profiler, "dequeue");

        prof_start!(maybe_profiler, "get out-of-domain index and rows");
        let trace_domain_generator = derive_domain_generator(padded_height as u64);
        let out_of_domain_curr_row_index = proof_stream.sample_scalars(1)[0];
        let _out_of_domain_next_row_index = trace_domain_generator * out_of_domain_curr_row_index;
        // todo: actually use the ood index – apply the DEEP trick

        let _out_of_domain_curr_base_row = proof_stream.dequeue()?.as_out_of_domain_base_row()?;
        let _out_of_domain_curr_ext_row = proof_stream.dequeue()?.as_out_of_domain_ext_row()?;
        let _out_of_domain_next_base_row = proof_stream.dequeue()?.as_out_of_domain_base_row()?;
        let _out_of_domain_next_ext_row = proof_stream.dequeue()?.as_out_of_domain_ext_row()?;
        // todo: actually use the ood rows – apply the DEEP trick
        prof_stop!(maybe_profiler, "get out-of-domain index and rows");

        prof_start!(maybe_profiler, "sample quotient codeword weights");
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights =
            Array1::from(proof_stream.sample_scalars(num_all_table_quotients()));
        prof_stop!(maybe_profiler, "sample quotient codeword weights");

        prof_start!(maybe_profiler, "get quotient codeword root");
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        let _out_of_domain_quotient_element = proof_stream.dequeue()?.as_out_of_domain_element()?;
        // todo: actually use the ood element – verify it was computed correctly
        prof_stop!(maybe_profiler, "get quotient codeword root");

        prof_start!(maybe_profiler, "Fiat-Shamir 2");
        let num_base_and_ext_codeword_weights = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS;
        let base_and_ext_codeword_weights =
            Array1::from(proof_stream.sample_scalars(num_base_and_ext_codeword_weights));
        prof_stop!(maybe_profiler, "Fiat-Shamir 2");

        prof_start!(maybe_profiler, "out-of-domain base&ext codeword element");
        let _out_of_domain_curr_row_base_and_ext_element =
            proof_stream.dequeue()?.as_out_of_domain_element()?;
        let _out_of_domain_next_row_base_and_ext_element =
            proof_stream.dequeue()?.as_out_of_domain_element()?;
        // todo: actually use the ood element – verify it was computed correctly and DEEP trick
        prof_stop!(maybe_profiler, "out-of-domain base&ext codeword element");

        // verify low degree of combination polynomial with FRI
        prof_start!(maybe_profiler, "FRI");
        let revealed_fri_indices_and_elements =
            self.fri.verify(&mut proof_stream, maybe_profiler)?;
        let (revealed_current_row_indices, revealed_fri_elements): (Vec<_>, Vec<_>) =
            revealed_fri_indices_and_elements.into_iter().unzip();
        // todo: actually use the revealed FRI elements – verify they were computed
        //   correctly, using the DEEP update
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "check leafs");
        prof_start!(maybe_profiler, "get indices");
        // the relation between the FRI domain and the trace domain
        let unit_distance = self.fri.domain.length / padded_height;
        let revealed_current_and_next_row_indices = self
            .revealed_current_and_next_row_indices(unit_distance, &revealed_current_row_indices);
        prof_stop!(maybe_profiler, "get indices");

        prof_start!(maybe_profiler, "dequeue base elements");
        let base_table_rows = proof_stream.dequeue()?.as_master_base_table_rows()?;
        let base_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_base: Vec<_> = base_table_rows
            .par_iter()
            .map(|revealed_base_elem| StarkHasher::hash_varlen(revealed_base_elem))
            .collect();
        prof_stop!(maybe_profiler, "dequeue base elements");

        prof_start!(maybe_profiler, "Merkle verify (base tree)");
        if !MerkleTree::<StarkHasher, Maker>::verify_authentication_structure_from_leaves(
            base_merkle_tree_root,
            &revealed_current_and_next_row_indices,
            &leaf_digests_base,
            &base_auth_paths,
        ) {
            bail!("Failed to verify authentication path for base codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (base tree)");

        prof_start!(maybe_profiler, "dequeue extension elements");
        let ext_table_rows = proof_stream.dequeue()?.as_master_ext_table_rows()?;
        let auth_paths_ext = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_ext = ext_table_rows
            .par_iter()
            .map(|xvalues| {
                let bvalues = xvalues
                    .iter()
                    .flat_map(|xfe| xfe.coefficients.to_vec())
                    .collect_vec();
                StarkHasher::hash_varlen(&bvalues)
            })
            .collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "dequeue extension elements");

        prof_start!(maybe_profiler, "Merkle verify (extension tree)");
        if !MerkleTree::<StarkHasher, Maker>::verify_authentication_structure_from_leaves(
            extension_tree_merkle_root,
            &revealed_current_and_next_row_indices,
            &leaf_digests_ext,
            &auth_paths_ext,
        ) {
            bail!("Failed to verify authentication path for extension codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (extension tree)");

        prof_start!(maybe_profiler, "Merkle verify (combined quotient)");
        let revealed_quotient_elements =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        let revealed_quotient_digests = revealed_quotient_elements
            .par_iter()
            .map(|&x| x.into())
            .collect::<Vec<_>>();
        let revealed_quotient_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        if !MerkleTree::<StarkHasher, Maker>::verify_authentication_structure_from_leaves(
            quotient_codeword_merkle_root,
            &revealed_current_row_indices,
            &revealed_quotient_digests,
            &revealed_quotient_auth_paths,
        ) {
            bail!("Failed to verify authentication path for combined quotient codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (combined quotient)");
        prof_stop!(maybe_profiler, "check leafs");

        prof_start!(maybe_profiler, "linear combination");
        prof_start!(maybe_profiler, "index");
        let (indexed_base_table_rows, indexed_ext_table_rows, indexed_randomizer_rows) =
            Self::index_revealed_rows(
                revealed_current_and_next_row_indices,
                base_table_rows,
                ext_table_rows,
            );
        prof_stop!(maybe_profiler, "index");

        // verify linear combination
        prof_start!(maybe_profiler, "main loop");
        let trace_domain_generator = derive_domain_generator(padded_height as u64);
        let trace_domain_generator_inverse = trace_domain_generator.inverse();
        for ((current_row_idx, revealed_quotient_element), revealed_fri_element) in
            revealed_current_row_indices
                .into_iter()
                .zip_eq(revealed_quotient_elements)
                .zip_eq(revealed_fri_elements)
        {
            prof_itr0!(maybe_profiler, "main loop");
            let next_row_idx = (current_row_idx + unit_distance) % self.fri.domain.length;
            let current_base_row = indexed_base_table_rows[&current_row_idx].view();
            let current_ext_row = indexed_ext_table_rows[&current_row_idx].view();
            let next_base_row = indexed_base_table_rows[&next_row_idx].view();
            let next_ext_row = indexed_ext_table_rows[&next_row_idx].view();

            prof_start!(maybe_profiler, "zerofiers");
            let one = BFieldElement::one();
            let current_fri_domain_value = self.fri.domain.domain_value(current_row_idx as u32);
            let initial_zerofier_inv = (current_fri_domain_value - one).inverse();
            let consistency_zerofier_inv =
                (current_fri_domain_value.mod_pow_u32(padded_height as u32) - one).inverse();
            let except_last_row = current_fri_domain_value - trace_domain_generator_inverse;
            let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
            let terminal_zerofier_inv = except_last_row.inverse();
            // i.e., only last row
            prof_stop!(maybe_profiler, "zerofiers");

            prof_start!(maybe_profiler, "evaluate AIR");
            let evaluated_initial_constraints =
                evaluate_all_initial_constraints(current_base_row, current_ext_row, &challenges);
            let evaluated_consistency_constraints = evaluate_all_consistency_constraints(
                current_base_row,
                current_ext_row,
                &challenges,
            );
            let evaluated_transition_constraints = evaluate_all_transition_constraints(
                current_base_row,
                current_ext_row,
                next_base_row,
                next_ext_row,
                &challenges,
            );
            let evaluated_terminal_constraints =
                evaluate_all_terminal_constraints(current_base_row, current_ext_row, &challenges);
            prof_stop!(maybe_profiler, "evaluate AIR");

            prof_start!(maybe_profiler, "base & ext elements");
            prof_start!(maybe_profiler, "collect");
            let mut base_and_ext_summands = current_base_row.map(|&element| element.lift());
            base_and_ext_summands
                .append(Axis(0), current_ext_row)
                .unwrap();
            prof_stop!(maybe_profiler, "collect");
            prof_start!(maybe_profiler, "inner product");
            let base_and_ext_element =
                (&base_and_ext_codeword_weights * base_and_ext_summands).sum();
            prof_stop!(maybe_profiler, "inner product");
            prof_stop!(maybe_profiler, "base & ext elements");

            prof_start!(maybe_profiler, "quotient elements");
            prof_start!(maybe_profiler, "divide");
            let mut quotient_summands = Vec::with_capacity(quot_codeword_weights.len());
            for (evaluated_constraints_category, zerofier_inverse) in [
                (evaluated_initial_constraints, initial_zerofier_inv),
                (evaluated_consistency_constraints, consistency_zerofier_inv),
                (evaluated_transition_constraints, transition_zerofier_inv),
                (evaluated_terminal_constraints, terminal_zerofier_inv),
            ] {
                let category_quotients = evaluated_constraints_category
                    .into_iter()
                    .map(|evaluated_constraint| evaluated_constraint * zerofier_inverse)
                    .collect_vec();
                quotient_summands.extend(category_quotients);
            }
            prof_stop!(maybe_profiler, "divide");
            prof_start!(maybe_profiler, "inner product");
            let quotient_element =
                (&quot_codeword_weights * &Array1::from(quotient_summands)).sum();
            prof_stop!(maybe_profiler, "inner product");
            if revealed_quotient_element != quotient_element {
                bail!(StarkValidationError::QuotientElementInequality);
            }
            prof_stop!(maybe_profiler, "quotient elements");

            prof_start!(maybe_profiler, "combination codeword equality");
            let randomizer_codewords_contribution = indexed_randomizer_rows[&current_row_idx].sum();
            // todo: DEEP update, check against revealed elements from FRI
            if revealed_fri_element
                != base_and_ext_element + quotient_element + randomizer_codewords_contribution
            {
                bail!(StarkValidationError::CombinationLeafInequality);
            }
            prof_stop!(maybe_profiler, "combination codeword equality");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "linear combination");
        Ok(true)
    }

    /// Hash-maps for base, extension, and randomizer rows that allow doing
    /// `indexed_revealed_base_rows[revealed_index]` instead of
    /// `revealed_base_rows[revealed_indices.iter().position(|&i| i == revealed_index).unwrap()]`.
    #[allow(clippy::type_complexity)]
    fn index_revealed_rows(
        revealed_indices: Vec<usize>,
        revealed_base_rows: Vec<Vec<BFieldElement>>,
        revealed_ext_rows: Vec<Vec<XFieldElement>>,
    ) -> (
        HashMap<usize, Array1<BFieldElement>>,
        HashMap<usize, Array1<XFieldElement>>,
        HashMap<usize, Array1<XFieldElement>>,
    ) {
        assert_eq!(revealed_indices.len(), revealed_base_rows.len());
        assert_eq!(revealed_indices.len(), revealed_ext_rows.len());

        let mut indexed_revealed_base_rows: HashMap<usize, Array1<BFieldElement>> = HashMap::new();
        let mut indexed_revealed_ext_rows: HashMap<usize, Array1<XFieldElement>> = HashMap::new();
        let mut indexed_revealed_rand_rows: HashMap<usize, Array1<XFieldElement>> = HashMap::new();

        for (i, &idx) in revealed_indices.iter().enumerate() {
            let base_row = Array1::from(revealed_base_rows[i].to_vec());
            let ext_row = Array1::from(revealed_ext_rows[i][..NUM_EXT_COLUMNS].to_vec());
            let rand_row = Array1::from(revealed_ext_rows[i][NUM_EXT_COLUMNS..].to_vec());

            indexed_revealed_base_rows.insert(idx, base_row);
            indexed_revealed_ext_rows.insert(idx, ext_row);
            indexed_revealed_rand_rows.insert(idx, rand_row);
        }

        (
            indexed_revealed_base_rows,
            indexed_revealed_ext_rows,
            indexed_revealed_rand_rows,
        )
    }
}

#[cfg(test)]
pub(crate) mod triton_stark_tests {
    use itertools::izip;
    use ndarray::Array1;
    use num_traits::Zero;
    use rand::prelude::ThreadRng;
    use rand_core::RngCore;
    use triton_opcodes::instruction::AnInstruction;
    use triton_opcodes::program::Program;

    use crate::shared_tests::*;
    use crate::table::cascade_table::ExtCascadeTable;
    use crate::table::challenges::ChallengeId::StandardInputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardInputTerminal;
    use crate::table::challenges::ChallengeId::StandardOutputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardOutputTerminal;
    use crate::table::cross_table_argument::CrossTableArg;
    use crate::table::cross_table_argument::EvalArg;
    use crate::table::cross_table_argument::GrandCrossTableArg;
    use crate::table::extension_table::Evaluable;
    use crate::table::extension_table::Quotientable;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::lookup_table::ExtLookupTable;
    use crate::table::master_table::all_degrees_with_origin;
    use crate::table::master_table::MasterExtTable;
    use crate::table::master_table::TableId::ProcessorTable;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table::ExtRamTable;
    use crate::table::table_column::MasterBaseTableColumn;
    use crate::table::table_column::MasterExtTableColumn;
    use crate::table::table_column::ProcessorBaseTableColumn;
    use crate::table::table_column::ProcessorExtTableColumn::InputTableEvalArg;
    use crate::table::table_column::ProcessorExtTableColumn::OutputTableEvalArg;
    use crate::table::table_column::RamBaseTableColumn;
    use crate::table::u32_table::ExtU32Table;
    use crate::vm::simulate;
    use crate::vm::triton_vm_tests::property_based_test_programs;
    use crate::vm::triton_vm_tests::small_tasm_test_programs;
    use crate::vm::triton_vm_tests::test_hash_nop_nop_lt;
    use crate::vm::AlgebraicExecutionTrace;

    use super::*;

    pub fn parse_setup_simulate(
        code: &str,
        input_symbols: Vec<BFieldElement>,
        secret_input_symbols: Vec<BFieldElement>,
    ) -> (AlgebraicExecutionTrace, Vec<BFieldElement>) {
        let program = Program::from_code(code);

        assert!(program.is_ok(), "program parses correctly");
        let program = program.unwrap();

        let (aet, stdout, err) = simulate(&program, input_symbols, secret_input_symbols);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {error}");
        }
        (aet, stdout)
    }

    pub fn parse_simulate_pad(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (Stark, MasterBaseTable, MasterBaseTable) {
        let (aet, stdout) = parse_setup_simulate(code, stdin.clone(), secret_in);

        let padded_height = MasterBaseTable::padded_height(&aet);
        let claim = Claim {
            input: stdin,
            program: aet.program.to_bwords(),
            output: stdout,
            padded_height,
        };
        let log_expansion_factor = 2;
        let security_level = 32;
        let parameters = StarkParameters::new(security_level, 1 << log_expansion_factor);
        let stark = Stark::new(claim, parameters);

        let mut master_base_table = MasterBaseTable::new(
            aet,
            stark.parameters.num_trace_randomizers,
            stark.fri.domain,
        );

        let unpadded_master_base_table = master_base_table.clone();
        master_base_table.pad();

        (stark, unpadded_master_base_table, master_base_table)
    }

    pub fn parse_simulate_pad_extend(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (
        Stark,
        MasterBaseTable,
        MasterBaseTable,
        MasterExtTable,
        Challenges,
    ) {
        let (stark, unpadded_master_base_table, master_base_table) =
            parse_simulate_pad(code, stdin, secret_in);

        let dummy_challenges = Challenges::placeholder(&stark.claim.input, &stark.claim.output);
        let master_ext_table = master_base_table.extend(
            &dummy_challenges,
            stark.parameters.num_randomizer_polynomials,
        );

        (
            stark,
            unpadded_master_base_table,
            master_base_table,
            master_ext_table,
            dummy_challenges,
        )
    }

    #[test]
    pub fn print_ram_table_example_for_specification() {
        let program = "
        push  5 push  6 write_mem pop
        push 15 push 16 write_mem pop
        push  5 read_mem pop pop
        push 15 read_mem pop pop
        push  5 push  7 write_mem pop
        push 15 read_mem
        push  5 read_mem
        halt
        ";
        let (_, master_base_table, _) = parse_simulate_pad(program, vec![], vec![]);

        println!();
        println!("Processor Table:");
        println!(
            "| clk        | pi         | ci         | nia        | st0        \
             | st1        | st2        | st3        | ramp       | ramv       |"
        );
        println!(
            "|-----------:|:-----------|:-----------|:-----------|-----------:\
             |-----------:|-----------:|-----------:|-----------:|-----------:|"
        );
        for row in master_base_table.table(ProcessorTable).rows() {
            let clk = row[ProcessorBaseTableColumn::CLK.base_table_index()].to_string();
            let st0 = row[ProcessorBaseTableColumn::ST0.base_table_index()].to_string();
            let st1 = row[ProcessorBaseTableColumn::ST1.base_table_index()].to_string();
            let st2 = row[ProcessorBaseTableColumn::ST2.base_table_index()].to_string();
            let st3 = row[ProcessorBaseTableColumn::ST3.base_table_index()].to_string();
            let ramp = row[ProcessorBaseTableColumn::RAMP.base_table_index()].to_string();
            let ramv = row[ProcessorBaseTableColumn::RAMV.base_table_index()].to_string();

            let prev_instruction =
                row[ProcessorBaseTableColumn::PreviousInstruction.base_table_index()].value();
            let curr_instruction = row[ProcessorBaseTableColumn::CI.base_table_index()].value();
            let next_instruction_or_arg =
                row[ProcessorBaseTableColumn::NIA.base_table_index()].value();

            // sorry about this mess – this is just a test.
            let pi = match AnInstruction::<BFieldElement>::try_from(prev_instruction) {
                Ok(AnInstruction::Halt) | Err(_) => "-".to_string(),
                Ok(instr) => instr.to_string().split('0').collect_vec()[0].to_owned(),
            };
            let ci = AnInstruction::<BFieldElement>::try_from(curr_instruction).unwrap();
            let nia = if ci.size() == 2 {
                next_instruction_or_arg.to_string()
            } else {
                AnInstruction::<BFieldElement>::try_from(next_instruction_or_arg)
                    .unwrap()
                    .to_string()
                    .split('0')
                    .collect_vec()[0]
                    .to_owned()
            };
            let ci_string = if ci.size() == 1 {
                ci.to_string()
            } else {
                ci.to_string().split('0').collect_vec()[0].to_owned()
            };
            let interesting_cols = [clk, pi, ci_string, nia, st0, st1, st2, st3, ramp, ramv];
            println!(
                "{}",
                interesting_cols
                    .iter()
                    .map(|ff| format!("{:>10}", format!("{ff}")))
                    .collect_vec()
                    .join(" | ")
            );
        }
        println!();
        println!("RAM Table:");
        println!("| clk        | pi         | ramp       | ramv       | iord |");
        println!("|-----------:|:-----------|-----------:|-----------:|-----:|");
        for row in master_base_table.table(TableId::RamTable).rows() {
            let clk = row[RamBaseTableColumn::CLK.base_table_index()].to_string();
            let ramp = row[RamBaseTableColumn::RAMP.base_table_index()].to_string();
            let ramv = row[RamBaseTableColumn::RAMV.base_table_index()].to_string();
            let iord =
                row[RamBaseTableColumn::InverseOfRampDifference.base_table_index()].to_string();

            let prev_instruction =
                row[RamBaseTableColumn::PreviousInstruction.base_table_index()].value();
            let pi = match AnInstruction::<BFieldElement>::try_from(prev_instruction) {
                Ok(AnInstruction::Halt) | Err(_) => "-".to_string(),
                Ok(instr) => instr.to_string().split('0').collect_vec()[0].to_owned(),
            };
            let interersting_cols = [clk, pi, ramp, ramv, iord];
            println!(
                "{}",
                interersting_cols
                    .iter()
                    .map(|ff| format!("{:>10}", format!("{ff}")))
                    .collect_vec()
                    .join(" | ")
            );
        }
    }

    /// To be used with `-- --nocapture`. Has mainly informative purpose.
    #[test]
    pub fn print_all_constraint_degrees() {
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        for deg in all_degrees_with_origin(interpolant_degree, padded_height) {
            println!("{deg}");
        }
    }

    #[test]
    pub fn check_io_terminals() {
        let read_nop_code = "read_io read_io read_io nop nop write_io push 17 write_io halt";
        let input_symbols = [3, 5, 7].map(BFieldElement::new).to_vec();
        let (stark, _, _, master_ext_table, all_challenges) =
            parse_simulate_pad_extend(read_nop_code, input_symbols, vec![]);

        let processor_table = master_ext_table.table(ProcessorTable);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        let ptie = processor_table_last_row[InputTableEvalArg.ext_table_index()];
        let ine = EvalArg::compute_terminal(
            &stark.claim.input,
            EvalArg::default_initial(),
            all_challenges.get_challenge(StandardInputIndeterminate),
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = processor_table_last_row[OutputTableEvalArg.ext_table_index()];
        let oute = EvalArg::compute_terminal(
            &stark.claim.output,
            EvalArg::default_initial(),
            all_challenges.get_challenge(StandardOutputIndeterminate),
        );
        assert_eq!(ptoe, oute, "The output evaluation arguments do not match.");
    }

    #[test]
    pub fn check_grand_cross_table_argument() {
        let mut code_collection = small_tasm_test_programs();
        code_collection.append(&mut property_based_test_programs());

        for (code_idx, code_with_input) in code_collection.into_iter().enumerate() {
            println!("Checking Grand Cross-Table Argument for TASM snippet {code_idx}.");
            let code = code_with_input.source_code;
            let input = code_with_input.input;
            let secret_input = code_with_input.secret_input.clone();
            let (_, _, master_base_table, master_ext_table, all_challenges) =
                parse_simulate_pad_extend(&code, input, secret_input);

            let processor_table = master_ext_table.table(ProcessorTable);
            let processor_table_last_row = processor_table.slice(s![-1, ..]);
            assert_eq!(
                all_challenges.get_challenge(StandardInputTerminal),
                processor_table_last_row[InputTableEvalArg.ext_table_index()],
                "The input terminal must match for TASM snippet #{code_idx}."
            );
            assert_eq!(
                all_challenges.get_challenge(StandardOutputTerminal),
                processor_table_last_row[OutputTableEvalArg.ext_table_index()],
                "The output terminal must match for TASM snippet #{code_idx}."
            );

            let master_base_trace_table = master_base_table.trace_table();
            let master_ext_trace_table = master_ext_table.trace_table();
            let last_master_base_row = master_base_trace_table.slice(s![-1, ..]);
            let last_master_ext_row = master_ext_trace_table.slice(s![-1, ..]);
            let evaluated_terminal_constraints = GrandCrossTableArg::evaluate_terminal_constraints(
                last_master_base_row,
                last_master_ext_row,
                &all_challenges,
            );
            assert_eq!(
                1,
                evaluated_terminal_constraints.len(),
                "The number of terminal constraints must be 1 – has the design changed?"
            );
            assert!(
                evaluated_terminal_constraints[0].is_zero(),
                "The terminal constraint must evaluate to 0 for TASM snippet #{code_idx}."
            );
        }
    }

    #[test]
    fn constraint_polynomials_use_right_variable_count_test() {
        let challenges = Challenges::placeholder(&[], &[]);
        let base_row = Array1::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);

        let br = base_row.view();
        let er = ext_row.view();

        ExtProgramTable::evaluate_initial_constraints(br, er, &challenges);
        ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtProgramTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtProgramTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtProcessorTable::evaluate_initial_constraints(br, er, &challenges);
        ExtProcessorTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtProcessorTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtProcessorTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtOpStackTable::evaluate_initial_constraints(br, er, &challenges);
        ExtOpStackTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtOpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtOpStackTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtRamTable::evaluate_initial_constraints(br, er, &challenges);
        ExtRamTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtRamTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtRamTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtJumpStackTable::evaluate_initial_constraints(br, er, &challenges);
        ExtJumpStackTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtJumpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtJumpStackTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtHashTable::evaluate_initial_constraints(br, er, &challenges);
        ExtHashTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtHashTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtHashTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtU32Table::evaluate_initial_constraints(br, er, &challenges);
        ExtU32Table::evaluate_consistency_constraints(br, er, &challenges);
        ExtU32Table::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtU32Table::evaluate_terminal_constraints(br, er, &challenges);
    }

    #[test]
    fn print_number_of_all_constraints_per_table() {
        let table_names = [
            "program table",
            "processor table",
            "op stack table",
            "ram table",
            "jump stack table",
            "hash table",
            "cascade table",
            "lookup table",
            "u32 table",
            "cross-table arg",
        ];
        let all_init = [
            ExtProgramTable::num_initial_quotients(),
            ExtProcessorTable::num_initial_quotients(),
            ExtOpStackTable::num_initial_quotients(),
            ExtRamTable::num_initial_quotients(),
            ExtJumpStackTable::num_initial_quotients(),
            ExtHashTable::num_initial_quotients(),
            ExtCascadeTable::num_initial_quotients(),
            ExtLookupTable::num_initial_quotients(),
            ExtU32Table::num_initial_quotients(),
            GrandCrossTableArg::num_initial_quotients(),
        ];
        let all_cons = [
            ExtProgramTable::num_consistency_quotients(),
            ExtProcessorTable::num_consistency_quotients(),
            ExtOpStackTable::num_consistency_quotients(),
            ExtRamTable::num_consistency_quotients(),
            ExtJumpStackTable::num_consistency_quotients(),
            ExtHashTable::num_consistency_quotients(),
            ExtCascadeTable::num_consistency_quotients(),
            ExtLookupTable::num_consistency_quotients(),
            ExtU32Table::num_consistency_quotients(),
            GrandCrossTableArg::num_consistency_quotients(),
        ];
        let all_trans = [
            ExtProgramTable::num_transition_quotients(),
            ExtProcessorTable::num_transition_quotients(),
            ExtOpStackTable::num_transition_quotients(),
            ExtRamTable::num_transition_quotients(),
            ExtJumpStackTable::num_transition_quotients(),
            ExtHashTable::num_transition_quotients(),
            ExtCascadeTable::num_transition_quotients(),
            ExtLookupTable::num_transition_quotients(),
            ExtU32Table::num_transition_quotients(),
            GrandCrossTableArg::num_transition_quotients(),
        ];
        let all_term = [
            ExtProgramTable::num_terminal_quotients(),
            ExtProcessorTable::num_terminal_quotients(),
            ExtOpStackTable::num_terminal_quotients(),
            ExtRamTable::num_terminal_quotients(),
            ExtJumpStackTable::num_terminal_quotients(),
            ExtHashTable::num_terminal_quotients(),
            ExtCascadeTable::num_terminal_quotients(),
            ExtLookupTable::num_terminal_quotients(),
            ExtU32Table::num_terminal_quotients(),
            GrandCrossTableArg::num_terminal_quotients(),
        ];

        let num_total_init: usize = all_init.iter().sum();
        let num_total_cons: usize = all_cons.iter().sum();
        let num_total_trans: usize = all_trans.iter().sum();
        let num_total_term: usize = all_term.iter().sum();
        let num_total = num_total_init + num_total_cons + num_total_trans + num_total_term;

        println!();
        println!("| Table                |  Init |  Cons | Trans |  Term |   Sum |");
        println!("|:---------------------|------:|------:|------:|------:|------:|");
        for (name, num_init, num_cons, num_trans, num_term) in
            izip!(table_names, all_init, all_cons, all_trans, all_term)
        {
            let num_total = num_init + num_cons + num_trans + num_term;
            println!(
                "| {name:<20} | {num_init:>5} | {num_cons:>5} \
                 | {num_trans:>5} | {num_term:>5} | {num_total:>5} |",
            );
        }
        println!(
            "| {:<20} | {num_total_init:>5} | {num_total_cons:>5} \
             | {num_total_trans:>5} | {num_total_term:>5} | {num_total:>5} |",
            "Sum",
        );
    }

    #[test]
    fn number_of_quotient_degree_bounds_match_number_of_constraints_test() {
        let base_row = Array1::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);
        let challenges = Challenges::placeholder(&[], &[]);
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        // Shorten some names for better formatting. This is just a test.
        let ph = padded_height;
        let id = interpolant_degree;
        let br = base_row.view();
        let er = ext_row.view();

        assert_eq!(
            ExtProgramTable::num_initial_quotients(),
            ExtProgramTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_initial_quotients(),
            ExtProgramTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtProcessorTable::num_initial_quotients(),
            ExtProcessorTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_initial_quotients(),
            ExtProcessorTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtOpStackTable::num_initial_quotients(),
            ExtOpStackTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_initial_quotients(),
            ExtOpStackTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtRamTable::num_initial_quotients(),
            ExtRamTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_initial_quotients(),
            ExtRamTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_initial_quotients(),
            ExtJumpStackTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_initial_quotients(),
            ExtJumpStackTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtHashTable::num_initial_quotients(),
            ExtHashTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_initial_quotients(),
            ExtHashTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtCascadeTable::num_initial_quotients(),
            ExtCascadeTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_initial_quotients(),
            ExtCascadeTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtLookupTable::num_initial_quotients(),
            ExtLookupTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_initial_quotients(),
            ExtLookupTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtU32Table::num_initial_quotients(),
            ExtU32Table::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_initial_quotients(),
            ExtU32Table::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_initial_quotients(),
            GrandCrossTableArg::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_initial_quotients(),
            GrandCrossTableArg::initial_quotient_degree_bounds(id).len()
        );

        assert_eq!(
            ExtProgramTable::num_consistency_quotients(),
            ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_consistency_quotients(),
            ExtProgramTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtProcessorTable::num_consistency_quotients(),
            ExtProcessorTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_consistency_quotients(),
            ExtProcessorTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtOpStackTable::num_consistency_quotients(),
            ExtOpStackTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_consistency_quotients(),
            ExtOpStackTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtRamTable::num_consistency_quotients(),
            ExtRamTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_consistency_quotients(),
            ExtRamTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_consistency_quotients(),
            ExtJumpStackTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_consistency_quotients(),
            ExtJumpStackTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtHashTable::num_consistency_quotients(),
            ExtHashTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_consistency_quotients(),
            ExtHashTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtCascadeTable::num_consistency_quotients(),
            ExtCascadeTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_consistency_quotients(),
            ExtCascadeTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtLookupTable::num_consistency_quotients(),
            ExtLookupTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_consistency_quotients(),
            ExtLookupTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtU32Table::num_consistency_quotients(),
            ExtU32Table::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_consistency_quotients(),
            ExtU32Table::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_consistency_quotients(),
            GrandCrossTableArg::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_consistency_quotients(),
            GrandCrossTableArg::consistency_quotient_degree_bounds(id, ph).len()
        );

        assert_eq!(
            ExtProgramTable::num_transition_quotients(),
            ExtProgramTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_transition_quotients(),
            ExtProgramTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtProcessorTable::num_transition_quotients(),
            ExtProcessorTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_transition_quotients(),
            ExtProcessorTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtOpStackTable::num_transition_quotients(),
            ExtOpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_transition_quotients(),
            ExtOpStackTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtRamTable::num_transition_quotients(),
            ExtRamTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_transition_quotients(),
            ExtRamTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_transition_quotients(),
            ExtJumpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_transition_quotients(),
            ExtJumpStackTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtHashTable::num_transition_quotients(),
            ExtHashTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_transition_quotients(),
            ExtHashTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtCascadeTable::num_transition_quotients(),
            ExtCascadeTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_transition_quotients(),
            ExtCascadeTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtLookupTable::num_transition_quotients(),
            ExtLookupTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_transition_quotients(),
            ExtLookupTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtU32Table::num_transition_quotients(),
            ExtU32Table::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_transition_quotients(),
            ExtU32Table::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_transition_quotients(),
            GrandCrossTableArg::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_transition_quotients(),
            GrandCrossTableArg::transition_quotient_degree_bounds(id, ph).len()
        );

        assert_eq!(
            ExtProgramTable::num_terminal_quotients(),
            ExtProgramTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_terminal_quotients(),
            ExtProgramTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtProcessorTable::num_terminal_quotients(),
            ExtProcessorTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_terminal_quotients(),
            ExtProcessorTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtOpStackTable::num_terminal_quotients(),
            ExtOpStackTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_terminal_quotients(),
            ExtOpStackTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtRamTable::num_terminal_quotients(),
            ExtRamTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_terminal_quotients(),
            ExtRamTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_terminal_quotients(),
            ExtJumpStackTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_terminal_quotients(),
            ExtJumpStackTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtHashTable::num_terminal_quotients(),
            ExtHashTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_terminal_quotients(),
            ExtHashTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtCascadeTable::num_terminal_quotients(),
            ExtCascadeTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_terminal_quotients(),
            ExtCascadeTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtLookupTable::num_terminal_quotients(),
            ExtLookupTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_terminal_quotients(),
            ExtLookupTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtU32Table::num_terminal_quotients(),
            ExtU32Table::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_terminal_quotients(),
            ExtU32Table::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_terminal_quotients(),
            GrandCrossTableArg::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_terminal_quotients(),
            GrandCrossTableArg::terminal_quotient_degree_bounds(id).len()
        );
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_halt_test() {
        triton_table_constraints_evaluate_to_zero(test_halt());
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_fibonacci_test() {
        let source_code_and_input = SourceCodeAndInput {
            source_code: FIBONACCI_SEQUENCE.to_string(),
            input: vec![BFieldElement::new(100)],
            secret_input: vec![],
        };
        triton_table_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_small_programs_test() {
        for (program_idx, program) in small_tasm_test_programs().into_iter().enumerate() {
            println!("Testing program with index {program_idx}.");
            triton_table_constraints_evaluate_to_zero(program);
        }
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_property_based_programs_test() {
        for (program_idx, program) in property_based_test_programs().into_iter().enumerate() {
            println!("Testing program with index {program_idx}.");
            triton_table_constraints_evaluate_to_zero(program);
        }
    }

    pub fn triton_table_constraints_evaluate_to_zero(source_code_and_input: SourceCodeAndInput) {
        let zero = XFieldElement::zero();
        let (_, _, master_base_table, master_ext_table, challenges) = parse_simulate_pad_extend(
            &source_code_and_input.source_code,
            source_code_and_input.input,
            source_code_and_input.secret_input,
        );

        assert_eq!(
            master_base_table.master_base_matrix.nrows(),
            master_ext_table.master_ext_matrix.nrows()
        );
        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let evaluated_initial_constraints = evaluate_all_initial_constraints(
            master_base_trace_table.row(0),
            master_ext_trace_table.row(0),
            &challenges,
        );
        let num_initial_constraints = evaluated_initial_constraints.len();
        assert_eq!(num_all_initial_quotients(), num_initial_constraints);
        for (constraint_idx, ebc) in evaluated_initial_constraints.into_iter().enumerate() {
            let (table_idx, table_name) = initial_constraint_table_idx_and_name(constraint_idx);
            assert_eq!(
                zero, ebc,
                "Failed initial constraint with global index {constraint_idx}. \
                Total number of initial constraints: {num_initial_constraints}. \
                Table: {table_name}. Index within table: {table_idx}",
            );
        }

        let num_rows = master_base_trace_table.nrows();
        for row_idx in 0..num_rows {
            let base_row = master_base_trace_table.row(row_idx);
            let ext_row = master_ext_trace_table.row(row_idx);
            let evaluated_consistency_constraints =
                evaluate_all_consistency_constraints(base_row, ext_row, &challenges);
            let num_consistency_constraints = evaluated_consistency_constraints.len();
            assert_eq!(num_all_consistency_quotients(), num_consistency_constraints);
            for (constraint_idx, ecc) in evaluated_consistency_constraints.into_iter().enumerate() {
                let (table_idx, table_name) =
                    consistency_constraint_table_idx_and_name(constraint_idx);
                assert_eq!(
                    zero, ecc,
                    "Failed consistency constraint with global index {constraint_idx}. \
                    Total number of consistency constraints: {num_consistency_constraints}. \
                    Table: {table_name}. Index within table: {table_idx} \
                    Row index: {row_idx}. \
                    Total rows: {num_rows}",
                );
            }
        }

        for row_idx in 0..num_rows - 1 {
            let base_row = master_base_trace_table.row(row_idx);
            let ext_row = master_ext_trace_table.row(row_idx);
            let next_base_row = master_base_trace_table.row(row_idx + 1);
            let next_ext_row = master_ext_trace_table.row(row_idx + 1);
            let evaluated_transition_constraints = evaluate_all_transition_constraints(
                base_row,
                ext_row,
                next_base_row,
                next_ext_row,
                &challenges,
            );
            let num_transition_constraints = evaluated_transition_constraints.len();
            assert_eq!(num_all_transition_quotients(), num_transition_constraints);
            for (constraint_idx, etc) in evaluated_transition_constraints.into_iter().enumerate() {
                if zero != etc {
                    let pi_idx =
                        ProcessorBaseTableColumn::PreviousInstruction.master_base_table_index();
                    let ci_idx = ProcessorBaseTableColumn::CI.master_base_table_index();
                    let nia_idx = ProcessorBaseTableColumn::NIA.master_base_table_index();
                    let pi = base_row[pi_idx].value();
                    let ci = base_row[ci_idx].value();
                    let nia = base_row[nia_idx].value();
                    let previous_instruction =
                        AnInstruction::<BFieldElement>::try_from(pi).unwrap();
                    let current_instruction = AnInstruction::<BFieldElement>::try_from(ci).unwrap();
                    let next_instruction_str = match AnInstruction::<BFieldElement>::try_from(nia) {
                        Ok(instr) => format!("{instr:?}"),
                        Err(_) => "not an instruction".to_string(),
                    };
                    let (table_idx, table_name) =
                        transition_constraint_table_idx_and_name(constraint_idx);
                    panic!(
                        "Failed transition constraint with global index {constraint_idx}. \
                        Total number of transition constraints: {num_transition_constraints}. \
                        Table: {table_name}. Index within table: {table_idx} \
                        Row index: {row_idx}. \
                        Total rows: {num_rows}\n\
                        Previous Instruction: {previous_instruction:?} – opcode: {pi}\n\
                        Current Instruction:  {current_instruction:?} – opcode: {ci}\n\
                        Next Instruction:     {next_instruction_str} – opcode: {nia}\n"
                    );
                }
            }
        }

        let evaluated_terminal_constraints = evaluate_all_terminal_constraints(
            master_base_trace_table.row(num_rows - 1),
            master_ext_trace_table.row(num_rows - 1),
            &challenges,
        );
        let num_terminal_constraints = evaluated_terminal_constraints.len();
        assert_eq!(num_all_terminal_quotients(), num_terminal_constraints);
        for (constraint_idx, etermc) in evaluated_terminal_constraints.into_iter().enumerate() {
            let (table_idx, table_name) = terminal_constraint_table_idx_and_name(constraint_idx);
            assert_eq!(
                zero, etermc,
                "Failed terminal constraint with global index {constraint_idx}. \
                Total number of terminal constraints: {num_terminal_constraints}. \
                Table: {table_name}. Index within table: {table_idx}",
            );
        }
    }

    #[test]
    fn triton_prove_verify_simple_program_test() {
        let code_with_input = test_hash_nop_nop_lt();
        let (stark, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
            &mut None,
        );

        println!("between prove and verify");

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());
    }

    #[test]
    fn triton_prove_verify_halt_test() {
        let mut profiler = Some(TritonProfiler::new("Prove Halt"));
        let code_with_input = test_halt();
        let (stark, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
            &mut profiler,
        );

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        if let Some(mut p) = profiler {
            p.finish();
            println!(
                "{}",
                p.report(
                    None,
                    Some(stark.claim.padded_height),
                    Some(stark.fri.domain.length),
                )
            );
        }
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_prove_halt_save_error_test() {
        let code_with_input = test_halt();

        for _ in 0..100 {
            let (stark, proof) = parse_simulate_prove(
                &code_with_input.source_code,
                code_with_input.input.clone(),
                code_with_input.secret_input.clone(),
                &mut None,
            );

            let filename = "halt_error.tsp";
            let result = stark.verify(proof.clone(), &mut None);
            if let Err(e) = result {
                if let Err(e) = save_proof(filename, proof) {
                    panic!("Unsyntactical proof and can't save! {e}");
                }
                panic!("Saved proof to {filename} because verifier is unhappy! {e}");
            }
            assert!(result.unwrap());
        }
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_load_verify_halt_test() {
        let code_with_input = test_halt();
        let (stark, _) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
            &mut None,
        );

        let filename = "halt_error.tsp";
        let proof = match load_proof(filename) {
            Ok(p) => p,
            Err(e) => panic!("Could not load proof from disk at {filename}: {e}"),
        };

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());
    }

    #[test]
    fn prove_verify_fibonacci_100_test() {
        let mut profiler = Some(TritonProfiler::new("Prove Fib 100"));
        let source_code = FIBONACCI_SEQUENCE;
        let stdin = vec![100_u64.into()];
        let secret_in = vec![];

        let (stark, proof) = parse_simulate_prove(source_code, stdin, secret_in, &mut profiler);

        println!("between prove and verify");

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        if let Some(mut p) = profiler {
            p.finish();
            println!(
                "{}",
                p.report(
                    None,
                    Some(stark.claim.padded_height),
                    Some(stark.fri.domain.length),
                )
            );
        }
    }

    #[test]
    fn prove_verify_fib_shootout_test() {
        let code = FIBONACCI_SEQUENCE;

        for (fib_seq_idx, fib_seq_val) in [(0, 1), (7, 21), (11, 144)] {
            let stdin = vec![BFieldElement::new(fib_seq_idx)];
            let secret_in = vec![];
            let (stark, proof) = parse_simulate_prove(code, stdin, secret_in, &mut None);
            match stark.verify(proof, &mut None) {
                Ok(result) => assert!(result, "The Verifier disagrees!"),
                Err(err) => panic!("The Verifier is unhappy! {err}"),
            }

            assert_eq!(vec![BFieldElement::new(fib_seq_val)], stark.claim.output);
        }
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations_test() {
        let many_u32_instructions = SourceCodeAndInput::without_input(MANY_U32_INSTRUCTIONS);
        triton_table_constraints_evaluate_to_zero(many_u32_instructions);
    }

    #[test]
    fn triton_prove_verify_many_u32_operations_test() {
        let mut profiler = Some(TritonProfiler::new("Prove Many U32 Ops"));
        let (stark, proof) =
            parse_simulate_prove(MANY_U32_INSTRUCTIONS, vec![], vec![], &mut profiler);
        let mut profiler = profiler.unwrap();
        profiler.finish();

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        println!(
            "{}",
            profiler.report(
                None,
                Some(stark.claim.padded_height),
                Some(stark.fri.domain.length),
            )
        );
    }

    #[test]
    #[ignore = "stress test"]
    fn prove_fib_successively_larger() {
        let source_code = FIBONACCI_SEQUENCE;

        for fibonacci_number in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200] {
            let mut profiler = Some(TritonProfiler::new(&format!(
                "element #{fibonacci_number:>4} from Fibonacci sequence"
            )));
            let stdin = vec![BFieldElement::new(fibonacci_number)];
            let (stark, _) = parse_simulate_prove(source_code, stdin, vec![], &mut profiler);
            if let Some(mut p) = profiler {
                p.finish();
                let report = p.report(
                    None,
                    Some(stark.claim.padded_height),
                    Some(stark.fri.domain.length),
                );
                println!("{report}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "Failed to convert BFieldElement")]
    pub fn negative_log_2_floor_test() {
        let mut rng = ThreadRng::default();
        let st0 = (rng.next_u32() as u64) << 32;

        let source_code = format!("push {st0} log_2_floor halt");
        let (stark, proof) = parse_simulate_prove(&source_code, vec![], vec![], &mut None);
        let result = stark.verify(proof, &mut None);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    #[should_panic(expected = "The logarithm of 0 does not exist")]
    pub fn negative_log_2_floor_of_0_test() {
        let source_code = "push 0 log_2_floor halt";
        let (stark, proof) = parse_simulate_prove(source_code, vec![], vec![], &mut None);
        let result = stark.verify(proof, &mut None);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}
