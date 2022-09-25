use std::fmt::Display;

use itertools::Itertools;
use num_traits::One;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::traits::{FiniteField, Inverse, ModPowU32};
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::timing_reporter::TimingReporter;

use crate::fri_domain::FriDomain;
use crate::stark::Stark;
use crate::table::table_collection::interpolant_degree;

use super::base_table::TableLike;
use super::challenges::AllChallenges;

// Generic methods specifically for tables that have been extended

pub trait ExtensionTable: TableLike<XFieldElement> + Sync {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>>;

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>>;

    fn dynamic_transition_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>>;

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>>;
}

pub trait Evaluable: ExtensionTable {
    /// evaluate initial constraints on given point if they are set; panic otherwise
    fn evaluate_initial_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        if let Some(initial_constraints) = &self.inherited_table().initial_constraints {
            initial_constraints
                .iter()
                .map(|bc| bc.evaluate(evaluation_point))
                .collect()
        } else {
            panic!("{} does not have initial constraints!", &self.name());
        }
    }

    /// evaluate consistency constraints on given point if they are set; panic otherwise
    fn evaluate_consistency_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        if let Some(consistency_constraints) = &self.inherited_table().consistency_constraints {
            consistency_constraints
                .iter()
                .map(|cc| cc.evaluate(evaluation_point))
                .collect()
        } else {
            panic!("{} does not have consistency constraints!", &self.name());
        }
    }

    /// evaluate transition constraints if they are set; panic otherwise
    fn evaluate_transition_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        if let Some(transition_constraints) = &self.inherited_table().transition_constraints {
            transition_constraints
                .iter()
                .map(|tc| tc.evaluate(evaluation_point))
                .collect()
        } else {
            panic!("{} does not have transition constraints!", &self.name());
        }
    }

    /// evaluate terminal constraints on given point if they are set; panic otherwise
    fn evaluate_terminal_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        if let Some(terminal_constraints) = &self.inherited_table().terminal_constraints {
            terminal_constraints
                .iter()
                .map(|termc| termc.evaluate(evaluation_point))
                .collect()
        } else {
            panic!("{} does not have terminal constraints!", &self.name());
        }
    }
}

pub trait Quotientable: ExtensionTable + Evaluable {
    /// Compute the degrees of the quotients from all AIR constraints that apply to the table.
    fn all_degrees_with_origin(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<DegreeWithOrigin> {
        let initial_degrees_with_origin = self
            .get_initial_quotient_degree_bounds(padded_height, num_trace_randomizers)
            .into_iter()
            .enumerate()
            .map(|(i, d)| DegreeWithOrigin {
                degree: d,
                origin_table_name: self.name(),
                origin_index: i,
                origin_table_height: padded_height,
                origin_constraint_type: "initial constraint".to_string(),
            })
            .collect_vec();

        let consistency_degrees_with_origin = self
            .get_consistency_quotient_degree_bounds(padded_height, num_trace_randomizers)
            .into_iter()
            .enumerate()
            .map(|(i, d)| DegreeWithOrigin {
                degree: d,
                origin_table_name: self.name(),
                origin_index: i,
                origin_table_height: padded_height,
                origin_constraint_type: "consistency constraint".to_string(),
            })
            .collect();

        let transition_degrees_with_origin = self
            .get_transition_quotient_degree_bounds(padded_height, num_trace_randomizers)
            .into_iter()
            .enumerate()
            .map(|(i, d)| DegreeWithOrigin {
                degree: d,
                origin_table_name: self.name(),
                origin_index: i,
                origin_table_height: padded_height,
                origin_constraint_type: "transition constraint".to_string(),
            })
            .collect();

        let terminal_degrees_with_origin = self
            .get_terminal_quotient_degree_bounds(padded_height, num_trace_randomizers)
            .into_iter()
            .enumerate()
            .map(|(i, d)| DegreeWithOrigin {
                degree: d,
                origin_table_name: self.name(),
                origin_index: i,
                origin_table_height: padded_height,
                origin_constraint_type: "terminal constraint".to_string(),
            })
            .collect();

        [
            initial_degrees_with_origin,
            consistency_degrees_with_origin,
            transition_degrees_with_origin,
            terminal_degrees_with_origin,
        ]
        .concat()
    }

    fn initial_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        codewords: &[Vec<XFieldElement>],
    ) -> Vec<Vec<XFieldElement>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        let zerofier_codeword = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - XFieldElement::one())
            .collect();
        let zerofier_inverse = XFieldElement::batch_inversion(zerofier_codeword);

        let transposed_quotient_codewords: Vec<_> = zerofier_inverse
            .par_iter()
            .enumerate()
            .map(|(fri_dom_i, &z_inv)| {
                let row = codewords
                    .iter()
                    .map(|codeword| codeword[fri_dom_i])
                    .collect_vec();
                let evaluated_bcs = self.evaluate_initial_constraints(&row);
                evaluated_bcs.iter().map(|&ebc| ebc * z_inv).collect()
            })
            .collect();
        let quotient_codewords = Stark::transpose_codewords(&transposed_quotient_codewords);
        self.debug_fri_domain_bound_check(fri_domain, &quotient_codewords, "initial");

        quotient_codewords
    }

    fn consistency_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        codewords: &[Vec<XFieldElement>],
        padded_height: usize,
    ) -> Vec<Vec<XFieldElement>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        let zerofier_codeword = fri_domain
            .domain_values()
            .iter()
            .map(|x| x.mod_pow_u32(padded_height as u32) - XFieldElement::one())
            .collect();
        let zerofier_inverse = XFieldElement::batch_inversion(zerofier_codeword);

        let transposed_quotient_codewords: Vec<_> = zerofier_inverse
            .par_iter()
            .enumerate()
            .map(|(fri_dom_i, &z_inv)| {
                let row = codewords
                    .iter()
                    .map(|codeword| codeword[fri_dom_i])
                    .collect_vec();
                let evaluated_ccs = self.evaluate_consistency_constraints(&row);
                evaluated_ccs.iter().map(|&ecc| ecc * z_inv).collect()
            })
            .collect();
        let quotient_codewords = Stark::transpose_codewords(&transposed_quotient_codewords);
        self.debug_fri_domain_bound_check(fri_domain, &quotient_codewords, "consistency");

        quotient_codewords
    }

    fn transition_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        codewords: &[Vec<XFieldElement>],
        omicron: XFieldElement,
        padded_height: usize,
    ) -> Vec<Vec<XFieldElement>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        let one = XFieldElement::one();
        let omicron_inverse = omicron.inverse();
        let fri_domain_values = fri_domain.domain_values();

        let subgroup_zerofier: Vec<_> = fri_domain_values
            .par_iter()
            .map(|fri_dom_v| fri_dom_v.mod_pow_u32(padded_height as u32) - one)
            .collect();
        let subgroup_zerofier_inverse = XFieldElement::batch_inversion(subgroup_zerofier);
        let zerofier_inverse: Vec<_> = fri_domain_values
            .into_par_iter()
            .zip_eq(subgroup_zerofier_inverse.into_par_iter())
            .map(|(fri_dom_v, sub_z_inv)| (fri_dom_v - omicron_inverse) * sub_z_inv)
            .collect();
        // the relation between the FRI domain and the omicron domain
        let unit_distance = fri_domain.length / padded_height;

        let transposed_quotient_codewords: Vec<_> = zerofier_inverse
            .par_iter()
            .enumerate()
            .map(|(current_row_idx, &z_inv)| {
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
                let evaluated_tcs = self.evaluate_transition_constraints(&evaluation_point);
                evaluated_tcs.iter().map(|&etc| etc * z_inv).collect()
            })
            .collect();
        let quotient_codewords = Stark::transpose_codewords(&transposed_quotient_codewords);
        self.debug_fri_domain_bound_check(fri_domain, &quotient_codewords, "transition");

        quotient_codewords
    }

    fn terminal_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        codewords: &[Vec<XFieldElement>],
        omicron: XFieldElement,
    ) -> Vec<Vec<XFieldElement>> {
        for codeword in codewords.iter() {
            debug_assert_eq!(fri_domain.length, codeword.len());
        }

        // The zerofier for the terminal quotient has a root in the last
        // value in the cyclical group generated from omicron.
        let zerofier_codeword = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - omicron.inverse())
            .collect_vec();
        let zerofier_inverse = XFieldElement::batch_inversion(zerofier_codeword);

        let transposed_quotient_codewords: Vec<_> = zerofier_inverse
            .par_iter()
            .enumerate()
            .map(|(fri_dom_i, &z_inv)| {
                let row = codewords
                    .iter()
                    .map(|codeword| codeword[fri_dom_i])
                    .collect_vec();
                let evaluated_termcs = self.evaluate_terminal_constraints(&row);
                evaluated_termcs.iter().map(|&etc| etc * z_inv).collect()
            })
            .collect();
        let quotient_codewords = Stark::transpose_codewords(&transposed_quotient_codewords);
        self.debug_fri_domain_bound_check(fri_domain, &quotient_codewords, "terminal");

        quotient_codewords
    }

    fn all_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        codewords: &[Vec<XFieldElement>],
        omicron: XFieldElement,
        padded_height: usize,
    ) -> Vec<Vec<XFieldElement>> {
        let mut timer = TimingReporter::start();
        timer.elapsed(&format!("Table name: {}", self.name()));

        let initial_quotients = self.initial_quotients(fri_domain, codewords);
        timer.elapsed("initial quotients");

        let consistency_quotients =
            self.consistency_quotients(fri_domain, codewords, padded_height);
        timer.elapsed("Done calculating consistency quotients");

        let transition_quotients =
            self.transition_quotients(fri_domain, codewords, omicron, padded_height);
        timer.elapsed("transition quotients");

        let terminal_quotients = self.terminal_quotients(fri_domain, codewords, omicron);
        timer.elapsed("terminal quotients");

        println!("{}", timer.finish());
        vec![
            initial_quotients,
            consistency_quotients,
            transition_quotients,
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
    fn debug_fri_domain_bound_check(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
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
                    Got degree {}, and FRI domain length was {}.",
                quotient_type,
                quotient_codewords.len(),
                self.name(),
                interpolated.degree(),
                fri_domain.length,
            );
        }
    }

    fn get_all_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        vec![
            self.get_initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
            self.get_consistency_quotient_degree_bounds(padded_height, num_trace_randomizers),
            self.get_transition_quotient_degree_bounds(padded_height, num_trace_randomizers),
            self.get_terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ]
        .concat()
    }

    fn get_initial_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        if let Some(db) = &self.inherited_table().initial_quotient_degree_bounds {
            db.to_owned()
        } else {
            let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
            let max_degrees = vec![interpolant_degree; self.full_width()];
            let zerofier_degree = 1;
            self.dynamic_initial_constraints(&AllChallenges::dummy())
                .iter()
                .map(|air| air.symbolic_degree_bound(&max_degrees) - zerofier_degree)
                .collect()
        }
    }

    fn get_consistency_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        if let Some(db) = &self.inherited_table().consistency_quotient_degree_bounds {
            db.to_owned()
        } else {
            let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
            let max_degrees = vec![interpolant_degree; self.full_width()];
            let zerofier_degree = padded_height as Degree;
            self.dynamic_consistency_constraints(&AllChallenges::dummy())
                .iter()
                .map(|air| air.symbolic_degree_bound(&max_degrees) - zerofier_degree)
                .collect()
        }
    }

    fn get_transition_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        if let Some(db) = &self.inherited_table().transition_quotient_degree_bounds {
            db.to_owned()
        } else {
            let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
            let max_degrees = vec![interpolant_degree; 2 * self.full_width()];
            let zerofier_degree = padded_height as Degree - 1;
            self.dynamic_transition_constraints(&AllChallenges::dummy())
                .iter()
                .map(|air| air.symbolic_degree_bound(&max_degrees) - zerofier_degree)
                .collect()
        }
    }

    fn get_terminal_quotient_degree_bounds(
        &self,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<Degree> {
        if let Some(db) = &self.inherited_table().terminal_quotient_degree_bounds {
            db.to_owned()
        } else {
            let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
            let max_degrees = vec![interpolant_degree; self.full_width()];
            let zerofier_degree = 1 as Degree;
            self.dynamic_terminal_constraints(&AllChallenges::dummy())
                .iter()
                .map(|air| air.symbolic_degree_bound(&max_degrees) - zerofier_degree)
                .collect()
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
    pub origin_table_height: usize,
    pub origin_constraint_type: String,
}

impl Default for DegreeWithOrigin {
    fn default() -> Self {
        DegreeWithOrigin {
            degree: -1,
            origin_table_name: "NoTable".to_string(),
            origin_index: usize::MAX,
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
            Table height was {}.",
            self.origin_table_name,
            self.origin_index,
            self.origin_constraint_type,
            self.degree,
            self.origin_table_height,
        )
    }
}
