use std::cmp::max;

use get_size::GetSize;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::tip5::Digest;

use crate::proof_stream::ProofStream;
use crate::stark;
use crate::table::extension_table::ConstraintType;
use crate::table::master_table;
use crate::StarkParameters;

/// Contains the necessary cryptographic information to verify a computation.
/// Should be used together with a [`Claim`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, BFieldCodec)]
pub struct Proof(pub Vec<BFieldElement>);

impl GetSize for Proof {
    fn get_stack_size() -> usize {
        std::mem::size_of::<Self>()
    }

    fn get_heap_size(&self) -> usize {
        self.0.len() * std::mem::size_of::<BFieldElement>()
    }
}

impl Proof {
    /// Given the parameters used to generate this proof,
    /// compute the height of the trace used during proof generation.
    /// This is an upper bound on the length of the computation this proof is for.
    /// It it one of the main contributing factors to the length of the FRI domain.
    pub fn padded_height(&self, parameters: &StarkParameters) -> usize {
        // The forward computation for the FRI domain length is
        //
        // ```rust
        // let interpolant_degree = npo2(padded_height + num_trace_randomizers) - 1;
        // let max_degree_for_fri = npo2(interpolant_degree * constraint_degree - zerofier_degree);
        // let fri_domain_length = fri_expansion_factor * max_degree_for_fri;
        // ```
        //
        // where `npo2` is the “next power of 2” function.
        // In above computation, the pair of `constraint_degree` and `zerofier_degree` are
        // assumed to be from the dominating constraint.
        //
        // Assuming that all arguments to `npo2` are powers of 2, this can be expressed as:
        //
        // ```rust
        // let fri_domain_length = fri_expansion_factor *
        //      ((padded_height + num_trace_randomizers - 1) * constraint_degree - zerofier_degree);
        // ```
        //
        // Note that the `zerofier_degree` potentially depends on the `padded_height`, depending
        // on the _type_ of the dominating constraint:
        // - For initial and terminal constraints, the `zerofier_degree` is 1.
        // - For consistency constraints, the `zerofier_degree` is `padded_height`.
        // - For transition constraints, the `zerofier_degree` is `padded_height - 1`.
        //
        // Using the above equations, it is possible to compute an upper bound for the
        // `padded_height`. Since the `padded_height` must itself be a power of two, the largest
        // power of two smaller than the upper bound for `padded_height` is the result.

        let proof_stream = ProofStream::<stark::StarkHasher>::try_from(self).unwrap();
        let mut auth_path_len = None;
        for item in proof_stream.items {
            if let Ok(auth_structure) = item.as_compressed_authentication_paths() {
                auth_path_len = Some(auth_structure[0].len());

                // It is fine to take the first candidate. If any item in the proof does not
                // conform to this candidate, the proof is invalid. Corresponding inconsistencies
                // will be identified during the verification process.
                break;
            }
        }
        let auth_path_len = auth_path_len.expect("The proof must contain authentication paths.");
        let fri_domain_length = 1 << auth_path_len;
        let max_degree_for_fri = fri_domain_length / parameters.fri_expansion_factor;

        // These dummy values are used to compute the constraint degree.
        // They are factored out later.
        let dummy_interpolant_degree = 2;
        let dummy_padded_height = 2;
        let max_degree_quotient_with_origin =
            master_table::max_degree_with_origin(dummy_interpolant_degree, dummy_padded_height);
        let constraint_degree = (max_degree_quotient_with_origin.degree
            + max_degree_quotient_with_origin.zerofier_degree)
            / dummy_interpolant_degree;
        assert!(constraint_degree > 0, "Constraint degree must be positive.");
        let constraint_degree: usize = constraint_degree.try_into().unwrap();

        let padded_height_times_constraint_degree_minus_zerofier_degree = max_degree_for_fri
            - parameters.num_trace_randomizers * constraint_degree
            + constraint_degree;

        let upper_bound_of_padded_height =
            match max_degree_quotient_with_origin.origin_constraint_type {
                ConstraintType::Initial | ConstraintType::Terminal => {
                    (padded_height_times_constraint_degree_minus_zerofier_degree + 1)
                        / constraint_degree
                }
                ConstraintType::Consistency => {
                    padded_height_times_constraint_degree_minus_zerofier_degree
                        / (constraint_degree - 1)
                }
                ConstraintType::Transition => {
                    (padded_height_times_constraint_degree_minus_zerofier_degree - 1)
                        / (constraint_degree - 1)
                }
            };

        // round down to the previous power of 2
        let padded_height = 1 << (upper_bound_of_padded_height.ilog2() - 1);

        // The `padded_height` must be at least as large as the number of trace randomizers.
        // See [`MasterBaseTable::padded_height()`] for a detailed explanation.
        let min_possible_padded_height_given_num_trace_randomizers =
            1 << parameters.num_trace_randomizers.ilog2();
        let padded_height = max(
            min_possible_padded_height_given_num_trace_randomizers,
            padded_height,
        );

        // The `padded_height` must be at least as large as the smallest possible padded height.
        let smallest_possible_padded_height = 1 << master_table::LOG2_MIN_PADDED_HEIGHT;
        max(smallest_possible_padded_height, padded_height)
    }

    /// The [`Claim`] that this proof is for.
    pub fn claim(&self) -> Claim {
        let proof_stream = ProofStream::<stark::StarkHasher>::try_from(self).unwrap();
        let mut claim = None;
        for item in proof_stream.items {
            if let Ok(found_claim) = item.as_claim() {
                assert!(claim.is_none(), "The proof must contain exactly one claim.");
                claim = Some(found_claim);
            }
        }
        claim.expect("The proof must contain a claim.")
    }
}

/// Contains the public information of a verifiably correct computation.
/// A corresponding [`Proof`] is needed to verify the computation.
/// One additional piece of public information not explicitly listed in the [`Claim`] is the
/// `padded_height`, an upper bound on the length of the computation.
/// It is derivable from a [`Proof`] by calling [`Proof::padded_height()`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, GetSize, BFieldCodec)]
pub struct Claim {
    /// The hash digest of the program that was executed. The hash function in use is Tip5.
    pub program_digest: Digest,

    /// The public input to the computation.
    pub input: Vec<BFieldElement>,

    /// The public output of the computation.
    pub output: Vec<BFieldElement>,
}

impl Claim {
    /// The public input as `u64`s.
    /// If `BFieldElement`s are needed, use field `input`.
    pub fn public_input(&self) -> Vec<u64> {
        self.input.iter().map(|x| x.value()).collect()
    }

    /// The public output as `u64`.
    /// If `BFieldElements`s are needed, use field `output`.
    pub fn public_output(&self) -> Vec<u64> {
        self.output.iter().map(|x| x.value()).collect()
    }
}

#[cfg(test)]
pub mod test_claim_proof {
    use std::collections::HashMap;

    use rand::random;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::bfield_codec::BFieldCodec;
    use twenty_first::shared_math::other::random_elements;

    use crate::stark::Stark;

    use super::*;

    #[test]
    fn test_decode_proof() {
        let data: Vec<BFieldElement> = random_elements(348);
        let proof = Proof(data);

        let encoded = proof.encode();
        let decoded = *Proof::decode(&encoded).unwrap();

        assert_eq!(proof, decoded);
    }

    #[test]
    fn test_decode_claim() {
        let claim = Claim {
            program_digest: random(),
            input: random_elements(346),
            output: random_elements(125),
        };

        let encoded = claim.encode();
        let decoded = *Claim::decode(&encoded).unwrap();

        assert_eq!(claim.program_digest, decoded.program_digest);
        assert_eq!(claim.input, decoded.input);
        assert_eq!(claim.output, decoded.output);
    }

    #[test]
    fn possible_padded_heights_to_fri_domain_lengths_is_bijective_test() {
        let parameters = StarkParameters::default();
        let num_trace_randomizers = parameters.num_trace_randomizers;

        let smallest_padded_height_exp = master_table::LOG2_MIN_PADDED_HEIGHT;
        let largest_padded_height_exp = 25;

        let mut fri_dom_lens_to_phs = HashMap::new();

        println!();
        println!("num_trace_randomizers = {num_trace_randomizers}");
        println!();
        println!("| exp | p_height | fri_dom_len |");
        println!("|----:|---------:|------------:|");
        for padded_height_exponent in smallest_padded_height_exp..=largest_padded_height_exp {
            let ph = 1 << padded_height_exponent;
            let max_degree = Stark::derive_max_degree(ph, num_trace_randomizers);
            let fri = Stark::derive_fri(&parameters, max_degree);
            let fri_domain_length = fri.domain.length;
            println!("| {padded_height_exponent:>3} | {ph:>8} | {fri_domain_length:>11} |");

            if let Some(other_ph) = fri_dom_lens_to_phs.insert(fri_domain_length, ph) {
                panic!(
                    "The FRI domain length {fri_domain_length} results from two padded heights: \
                    {other_ph} and {ph}.",
                );
            }
        }
    }
}
