use itertools::Itertools;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::fmt::Display;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::traits::{Inverse, ModPowU32, PrimeField};
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::timing_reporter::TimingReporter;

use crate::fri_domain::FriDomain;
use crate::table::challenges_endpoints::AllEndpoints;

use super::base_table::BaseTableTrait;
use super::challenges_endpoints::AllChallenges;

// Generic methods specifically for tables that have been extended

type XWord = XFieldElement;

pub trait ExtensionTable: BaseTableTrait<XWord> + Sync {
    /// get boundary constraints if they are set; panic otherwise
    fn get_boundary_constraints(&self) -> Vec<MPolynomial<XWord>> {
        if let Some(bc) = &self.to_base().boundary_constraints {
            bc.to_owned()
        } else {
            panic!("{} does not have boundary constraints!", &self.name());
        }
    }

    /// get transition constraints if they are set; panic otherwise
    fn get_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        if let Some(tc) = &self.to_base().transition_constraints {
            tc.to_owned()
        } else {
            panic!("{} does not have transition constraints!", &self.name());
        }
    }

    /// get consistency constraints if they are set; panic otherwise
    fn get_consistency_constraints(&self) -> Vec<MPolynomial<XWord>> {
        if let Some(cc) = &self.to_base().consistency_constraints {
            cc.to_owned()
        } else {
            panic!("{} does not have consistency constraints! ", &self.name());
        }
    }

    /// get terminal constraints if they are set; panic otherwise
    fn get_terminal_constraints(&self) -> Vec<MPolynomial<XWord>> {
        if let Some(tc) = &self.to_base().terminal_constraints {
            tc.to_owned()
        } else {
            panic!("{} does not have terminal constraints!", &self.name());
        }
    }

    /// Compute the degrees of the quotients from all AIR constraints that apply to the table.
    /// TODO: cover other constraints beyond just transitions
    /// TODO: work with unset/general terminals
    fn all_degrees_with_origin(&self) -> Vec<DegreeWithOrigin> {
        let interpolant_degree = self.interpolant_degree();
        let interpolants_degrees = vec![interpolant_degree; self.full_width()];
        let duplicated_interpolants_degrees = vec![interpolant_degree; self.full_width() * 2];

        let boundary_zerofier_degree = 1;
        let transition_zerofier_degree = interpolant_degree - 1;
        let consistency_zerofier_degree = interpolant_degree;
        let terminal_zerofier_degree = 1;

        let boundary_degrees_with_origin = self
            .dynamic_boundary_constraints(&AllChallenges::dummy())
            .iter()
            .enumerate()
            .map(|(i, air)| {
                let boundary_polynomial_degree_bound =
                    air.symbolic_degree_bound(&interpolants_degrees);
                DegreeWithOrigin {
                    degree: boundary_polynomial_degree_bound - boundary_zerofier_degree,
                    origin_table_name: self.name(),
                    origin_index: i,
                    origin_air_degree: air.degree(),
                    origin_table_height: self.padded_height(),
                    origin_constraint_type: "boundary constraint".to_string(),
                }
            })
            .collect_vec();

        let transition_degrees_with_origin = self
            .dynamic_transition_constraints(&AllChallenges::dummy())
            .iter()
            .enumerate()
            .map(|(i, air)| {
                let transition_polynomial_degree_bound =
                    air.symbolic_degree_bound(&duplicated_interpolants_degrees);
                DegreeWithOrigin {
                    degree: transition_polynomial_degree_bound - transition_zerofier_degree,
                    origin_table_name: self.name(),
                    origin_index: i,
                    origin_air_degree: air.degree(),
                    origin_table_height: self.padded_height(),
                    origin_constraint_type: "transition constraint".to_string(),
                }
            })
            .collect();

        let consistency_degrees_with_origin = self
            .dynamic_consistency_constraints()
            .iter()
            .enumerate()
            .map(|(i, air)| {
                let consistency_polynomial_degree_bound =
                    air.symbolic_degree_bound(&interpolants_degrees);
                DegreeWithOrigin {
                    degree: consistency_polynomial_degree_bound - consistency_zerofier_degree,
                    origin_table_name: self.name(),
                    origin_index: i,
                    origin_air_degree: air.degree(),
                    origin_table_height: self.padded_height(),
                    origin_constraint_type: "consistency constraint".to_string(),
                }
            })
            .collect();

        let terminal_degrees_with_origin = self
            .dynamic_terminal_constraints(&AllChallenges::dummy(), &AllEndpoints::dummy())
            .iter()
            .enumerate()
            .map(|(i, air)| {
                let terminal_polynomial_degree_bound =
                    air.symbolic_degree_bound(&interpolants_degrees);
                DegreeWithOrigin {
                    degree: terminal_polynomial_degree_bound - terminal_zerofier_degree,
                    origin_table_name: self.name(),
                    origin_index: i,
                    origin_air_degree: air.degree(),
                    origin_table_height: self.padded_height(),
                    origin_constraint_type: "terminal constraint".to_string(),
                }
            })
            .collect();

        [
            boundary_degrees_with_origin,
            transition_degrees_with_origin,
            consistency_degrees_with_origin,
            terminal_degrees_with_origin,
        ]
        .concat()
    }

    fn dynamic_boundary_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>>;

    fn dynamic_transition_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>>;

    fn dynamic_consistency_constraints(&self) -> Vec<MPolynomial<XFieldElement>>;

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
        terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>>;

    fn get_all_quotient_degree_bounds(&self) -> Vec<Degree> {
        vec![
            self.get_boundary_quotient_degree_bounds(),
            self.get_transition_quotient_degree_bounds(),
            self.get_consistency_quotient_degree_bounds(),
            self.get_terminal_quotient_degree_bounds(),
        ]
        .concat()
    }

    fn get_boundary_quotient_degree_bounds(&self) -> Vec<Degree> {
        if let Some(db) = &self.to_base().boundary_quotient_degree_bounds {
            db.to_owned()
        } else {
            panic!(
                "{} does not have boundary quotient degree bounds!",
                &self.name()
            );
        }
    }

    fn get_transition_quotient_degree_bounds(&self) -> Vec<Degree> {
        if let Some(db) = &self.to_base().transition_quotient_degree_bounds {
            db.to_owned()
        } else {
            panic!(
                "{} does not have transition quotient degree bounds!",
                &self.name()
            );
        }
    }

    fn get_consistency_quotient_degree_bounds(&self) -> Vec<Degree> {
        if let Some(db) = &self.to_base().consistency_quotient_degree_bounds {
            db.to_owned()
        } else {
            panic!(
                "{} does not have consistency quotient degree bounds!",
                &self.name()
            );
        }
    }

    fn get_terminal_quotient_degree_bounds(&self) -> Vec<Degree> {
        if let Some(db) = &self.to_base().terminal_quotient_degree_bounds {
            db.to_owned()
        } else {
            panic!(
                "{} does not have terminal quotient degree bounds!",
                &self.name()
            );
        }
    }
}

pub trait Quotientable: ExtensionTable {
    fn boundary_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
    ) -> Vec<Vec<XWord>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        let zerofier_codeword = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - XFieldElement::ring_one())
            .collect();

        let boundary_constraints = self.get_boundary_constraints();
        let quotient_codewords =
            self.quotients(codewords, zerofier_codeword, &boundary_constraints);
        self.debug_degree_bound_check(
            fri_domain,
            &boundary_constraints,
            &quotient_codewords,
            "boundary",
        );

        quotient_codewords
    }
    fn transition_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
    ) -> Vec<Vec<XFieldElement>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        let one = XWord::ring_one();
        let height = self.padded_height() as u32;
        let omicron_inverse = self.omicron().inverse();
        let fri_domain_values = fri_domain.domain_values();

        let subgroup_zerofier: Vec<_> = fri_domain_values
            .par_iter()
            .map(|fri_dom_v| fri_dom_v.mod_pow_u32(height) - one)
            .collect();
        let subgroup_zerofier_inverse = if height == 0 {
            subgroup_zerofier
        } else {
            XWord::batch_inversion(subgroup_zerofier)
        };
        let zerofier_inverse: Vec<_> = fri_domain_values
            .into_par_iter()
            .zip_eq(subgroup_zerofier_inverse.into_par_iter())
            .map(|(fri_dom_v, sub_z_inv)| (fri_dom_v - omicron_inverse) * sub_z_inv)
            .collect();

        let mut quotients: Vec<Vec<XWord>> = vec![];
        let unit_distance = self.unit_distance(fri_domain.length);
        let transition_constraints = self.get_transition_constraints();

        for tc in transition_constraints.iter() {
            let quotient_codeword: Vec<_> = zerofier_inverse
                .par_iter()
                .enumerate()
                .map(|(current_row_idx, &z_inverse)| {
                    let current_row = codewords
                        .iter()
                        .map(|codeword| codeword[current_row_idx])
                        .collect_vec();
                    let next_row_idx = (current_row_idx + unit_distance) % fri_domain.length;
                    let next_row = codewords
                        .iter()
                        .map(|codeword| codeword[next_row_idx])
                        .collect_vec();
                    let evaluation_point = vec![current_row, next_row].concat();
                    let evaluated_constraint = tc.evaluate(&evaluation_point);
                    evaluated_constraint * z_inverse
                })
                .collect();
            quotients.push(quotient_codeword);
        }
        self.debug_degree_bound_check(
            fri_domain,
            &transition_constraints,
            &quotients,
            "transition",
        );

        quotients
    }

    fn consistency_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
    ) -> Vec<Vec<XWord>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        let zerofier_codeword = fri_domain
            .domain_values()
            .iter()
            .map(|x| x.mod_pow_u32(self.padded_height() as u32) - XWord::ring_one())
            .collect();

        let consistency_constraints = self.get_consistency_constraints();
        let quotient_codewords =
            self.quotients(codewords, zerofier_codeword, &consistency_constraints);
        self.debug_degree_bound_check(
            fri_domain,
            &consistency_constraints,
            &quotient_codewords,
            "consistency",
        );

        quotient_codewords
    }

    fn terminal_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
    ) -> Vec<Vec<XWord>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        // The zerofier for the terminal quotient has a root in the last
        // value in the cyclical group generated from omicron.
        let zerofier_codeword = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - self.omicron().inverse())
            .collect_vec();

        let terminal_constraints = self.get_terminal_constraints();
        let quotient_codewords =
            self.quotients(codewords, zerofier_codeword, &terminal_constraints);
        self.debug_degree_bound_check(
            fri_domain,
            &terminal_constraints,
            &quotient_codewords,
            "terminal",
        );

        quotient_codewords
    }

    /// Given some `constraints`, `codewords`, and a `zerofier`, computes `constraints.len()`-many
    /// `quotient_codewords` by
    /// 1. evaluating the `constraints` on the `codewords`, then
    /// 1. dividing the result by the `zerofier`.
    ///
    /// All `constraints` must be multivariate polynomials with `codewords.len()`-many variables.
    fn quotients(
        &self,
        codewords: &[Vec<XWord>],
        zerofier: Vec<XFieldElement>,
        constraints: &[MPolynomial<XWord>],
    ) -> Vec<Vec<XWord>> {
        let zerofier_inverse = if self.padded_height() == 0 {
            zerofier
        } else {
            XWord::batch_inversion(zerofier)
        };

        let mut quotient_codewords = vec![];
        for constraint in constraints.iter() {
            let quotient_codeword: Vec<_> = zerofier_inverse
                .par_iter()
                .enumerate()
                .map(|(fri_dom_i, z_inv)| {
                    let row = codewords
                        .iter()
                        .map(|codeword| codeword[fri_dom_i])
                        .collect_vec();
                    constraint.evaluate(&row) * *z_inv
                })
                .collect();
            quotient_codewords.push(quotient_codeword);
        }
        quotient_codewords
    }

    fn all_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
    ) -> Vec<Vec<XWord>> {
        let mut timer = TimingReporter::start();
        timer.elapsed(&format!("Table name: {}", self.name()));

        let boundary_quotients = self.boundary_quotients(fri_domain, codewords);
        timer.elapsed("boundary quotients");

        let transition_quotients = self.transition_quotients(fri_domain, codewords);
        timer.elapsed("transition quotients");

        let consistency_quotients = self.consistency_quotients(fri_domain, codewords);
        timer.elapsed("Done calculating consistency quotients");

        let terminal_quotients = self.terminal_quotients(fri_domain, codewords);
        timer.elapsed("terminal quotients");

        println!("{}", timer.finish());
        vec![
            boundary_quotients,
            transition_quotients,
            consistency_quotients,
            terminal_quotients,
        ]
        .concat()
    }

    /// Intended for debugging. Will not do anything unless environment variable `DEBUG` is set.
    /// The performed check
    /// 1. takes `quotients` in value form (i.e., as codewords),
    /// 1. interpolates them over the given `fri_domain`, and
    /// 1. checks their degree.
    ///
    /// Panics if an interpolant has maximal degree, indicating that the quotient codeword is most
    /// probably the result of un-clean division.
    fn debug_degree_bound_check(
        &self,
        fri_domain: &FriDomain<XWord>,
        constraints: &[MPolynomial<XWord>],
        quotient_codewords: &[Vec<XFieldElement>],
        quotient_type: &str,
    ) {
        if std::env::var("DEBUG").is_err() {
            return;
        }
        for (idx, qc) in quotient_codewords.iter().enumerate() {
            let interpolated = fri_domain.interpolate(qc);
            assert!(
                interpolated.degree() < fri_domain.length as isize - 1,
                "Degree of {} quotient index {idx} (total {} quotients) in {} must not be maximal. \
                    Got degree {}, and FRI domain length was {}.\
                    Unsatisfied constraint: {}",
                quotient_type,
                quotient_codewords.len(),
                self.name(),
                interpolated.degree(),
                fri_domain.length,
                constraints[idx]
            );
        }
    }
}
pub trait QuotientableExtensionTable: ExtensionTable + Quotientable {}

/// Helps debugging and benchmarking. The maximal degree achieved in any table dictates the length
/// of the FRI domain, which in turn is responsible for the main performance bottleneck.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct DegreeWithOrigin {
    pub degree: Degree,
    pub origin_table_name: String,
    pub origin_index: usize,
    pub origin_air_degree: Degree,
    pub origin_table_height: usize,
    pub origin_constraint_type: String,
}

impl Default for DegreeWithOrigin {
    fn default() -> Self {
        DegreeWithOrigin {
            degree: -1,
            origin_table_name: "NoTable".to_string(),
            origin_index: usize::MAX,
            origin_air_degree: -1,
            origin_table_height: 0,
            origin_constraint_type: "NoType".to_string(),
        }
    }
}

impl Display for DegreeWithOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Degree of poly for table {} (index {:02}) of type {} is {:02}. \
            AIR has degree {:02}. Table height was {}.",
            self.origin_table_name,
            self.origin_index,
            self.origin_constraint_type,
            self.degree,
            self.origin_air_degree,
            self.origin_table_height,
        )
    }
}
