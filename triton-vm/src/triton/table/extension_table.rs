use super::base_table::Table;
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use crate::shared_math::mpolynomial::{Degree, MPolynomial};
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::traits::{Inverse, ModPowU32, PrimeField};
use crate::shared_math::x_field_element::XFieldElement;
use crate::timing_reporter::TimingReporter;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

// Generic methods specifically for tables that have been extended

type XWord = XFieldElement;

pub trait ExtensionTable: Table<XWord> + Sync {
    fn ext_boundary_constraints(&self, challenges: &AllChallenges) -> Vec<MPolynomial<XWord>>;

    fn ext_consistency_constraints(&self, challenges: &AllChallenges) -> Vec<MPolynomial<XWord>>;

    fn ext_transition_constraints(&self, challenges: &AllChallenges) -> Vec<MPolynomial<XWord>>;

    fn ext_terminal_constraints(
        &self,
        challenges: &AllChallenges,
        terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>>;

    fn max_degree(&self) -> Degree {
        let degree_bounds: Vec<Degree> = vec![self.interpolant_degree(); self.full_width() * 2];

        // 1. Insert dummy challenges
        // 2. Refactor so we can calculate max_degree without specifying challenges
        //    (and possibly without even calling ext_transition_constraints).
        self.ext_transition_constraints(&AllChallenges::dummy())
            .iter()
            .map(|air| {
                let symbolic_degree_bound: Degree = air.symbolic_degree_bound(&degree_bounds);
                let padded_height: Degree = self.padded_height() as Degree;

                symbolic_degree_bound - padded_height + 1
            })
            .max()
            .unwrap_or(-1)
    }

    fn all_quotient_degree_bounds(
        &self,
        challenges: &AllChallenges,
        terminals: &AllEndpoints,
    ) -> Vec<Degree> {
        vec![
            self.boundary_quotient_degree_bounds(challenges),
            self.transition_quotient_degree_bounds(challenges),
            self.terminal_quotient_degree_bounds(challenges, terminals),
        ]
        .concat()
    }

    fn boundary_quotient_degree_bounds(&self, challenges: &AllChallenges) -> Vec<Degree> {
        let max_degrees: Vec<Degree> = vec![self.interpolant_degree(); self.full_width()];

        let degree_bounds: Vec<Degree> = self
            .ext_boundary_constraints(challenges)
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&max_degrees) - 1)
            .collect();

        degree_bounds
    }

    fn transition_quotient_degree_bounds(&self, challenges: &AllChallenges) -> Vec<Degree> {
        let max_degrees: Vec<Degree> = vec![self.interpolant_degree(); 2 * self.full_width()];

        let transition_constraints = self.ext_transition_constraints(challenges);

        // Safe because padded height is at most 2^30.
        let padded_height: Degree = self.padded_height().try_into().unwrap();

        transition_constraints
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&max_degrees) - padded_height + 1)
            .collect::<Vec<Degree>>()
    }

    fn terminal_quotient_degree_bounds(
        &self,
        challenges: &AllChallenges,
        terminals: &AllEndpoints,
    ) -> Vec<Degree> {
        let max_degrees: Vec<Degree> = vec![self.interpolant_degree(); self.full_width()];
        self.ext_terminal_constraints(challenges, terminals)
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&max_degrees) - 1)
            .collect::<Vec<Degree>>()
    }

    fn all_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
        challenges: &AllChallenges,
        terminals: &AllEndpoints,
    ) -> Vec<Vec<XWord>> {
        let mut timer = TimingReporter::start();
        timer.elapsed(&format!("Table name: {}", self.name()));

        let boundary_quotients = self.boundary_quotients(fri_domain, codewords, challenges);
        timer.elapsed("boundary quotients");

        // TODO take consistency quotients into account
        // let consistency_quotients = self.consistency_quotients(fri_domain, codewords, challenges);
        // timer.elapsed("Done calculating consistency quotients");

        let transition_quotients = self.transition_quotients(fri_domain, codewords, challenges);
        timer.elapsed("transition quotients");

        let terminal_quotients =
            self.terminal_quotients(fri_domain, codewords, challenges, terminals);
        timer.elapsed("terminal quotients");

        println!("{}", timer.finish());
        vec![
            boundary_quotients,
            // consistency_quotients,
            transition_quotients,
            terminal_quotients,
        ]
        .concat()
    }

    fn transition_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords_transposed: &[Vec<XWord>],
        challenges: &AllChallenges,
    ) -> Vec<Vec<XWord>> {
        let mut timer = TimingReporter::start();
        timer.elapsed("Start transition quotients");

        let one = XWord::ring_one();
        let x_values = fri_domain.domain_values();
        timer.elapsed("Domain values");

        let subgroup_zerofier: Vec<XWord> = x_values
            .iter()
            .map(|x| x.mod_pow_u32(self.padded_height() as u32) - one)
            .collect();
        let subgroup_zerofier_inverse = if self.padded_height() == 0 {
            subgroup_zerofier
        } else {
            XWord::batch_inversion(subgroup_zerofier)
        };
        timer.elapsed("Batch Inversion");
        let omicron_inverse = self.omicron().inverse();
        let zerofier_inverse: Vec<XWord> = x_values
            .into_iter()
            .enumerate()
            .map(|(i, x)| subgroup_zerofier_inverse[i] * (x - omicron_inverse))
            .collect();
        timer.elapsed("Zerofier Inverse");
        let transition_constraints = self.ext_transition_constraints(challenges);
        timer.elapsed("Transition Constraints");

        let mut quotients: Vec<Vec<XWord>> = vec![];
        let unit_distance = self.unit_distance(fri_domain.length);

        for tc in transition_constraints.iter() {
            //timer.elapsed(&format!("START for-loop for tc of {}", tc.degree()));
            let quotient_codeword: Vec<XWord> = zerofier_inverse
                .par_iter()
                .enumerate()
                .map(|(i, z_inverse)| {
                    let current_row: Vec<XWord> = (0..self.full_width())
                        .map(|j| codewords_transposed[j][i])
                        .collect();

                    let next_row: Vec<XWord> = (0..self.full_width())
                        .map(|j| codewords_transposed[j][(i + unit_distance) % fri_domain.length])
                        .collect();

                    let point = vec![current_row, next_row].concat();
                    let composition_evaluation = tc.evaluate(&point);
                    composition_evaluation * *z_inverse
                })
                .collect();

            quotients.push(quotient_codeword);
            timer.elapsed(&format!("END for-loop for tc of {}", tc.degree()));
        }
        timer.elapsed("DONE Transition Constraints");

        if std::env::var("DEBUG").is_ok() {
            // interpolate the quotient and check the degree
            for (idx, qc) in quotients.iter().enumerate() {
                let interpolated: Polynomial<XWord> = fri_domain.interpolate(qc);
                assert!(
                    interpolated.degree() < fri_domain.length as isize - 1,
                    "Degree of transition quotient number {idx} (of {}) in {} must not be maximal. \
                    Got degree {}, and FRI domain length was {}. \
                    Unsatisfied constraint: {}",
                    quotients.len(),
                    self.name(),
                    interpolated.degree(),
                    fri_domain.length,
                    transition_constraints[idx]
                );
            }
        }

        quotients
    }

    fn terminal_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
        challenges: &AllChallenges,
        terminals: &AllEndpoints,
    ) -> Vec<Vec<XWord>> {
        let omicron_inverse = self.omicron().inverse();

        // The zerofier for the terminal quotient has a root in the last
        // value in the cyclical group generated from omicron.
        let zerofier_codeword: Vec<XWord> = fri_domain
            .domain_values()
            .into_iter()
            .map(|x| x - omicron_inverse)
            .collect();

        let zerofier_inverse = XWord::batch_inversion(zerofier_codeword);
        let terminal_constraints = self.ext_terminal_constraints(challenges, terminals);
        let mut quotient_codewords: Vec<Vec<XWord>> = vec![];
        for termc in terminal_constraints.iter() {
            let quotient_codeword: Vec<XWord> = (0..fri_domain.length)
                .into_par_iter()
                .map(|i| {
                    let point: Vec<XWord> =
                        (0..self.full_width()).map(|j| codewords[j][i]).collect();

                    termc.evaluate(&point) * zerofier_inverse[i]
                })
                .collect();
            quotient_codewords.push(quotient_codeword);
        }

        if std::env::var("DEBUG").is_ok() {
            for (idx, qc) in quotient_codewords.iter().enumerate() {
                let interpolated = fri_domain.interpolate(qc);
                assert!(
                    interpolated.degree() < fri_domain.length as isize - 1,
                    "Degree of terminal quotient number {idx} (of {}) in {} must not be maximal. \
                    Got degree {}, and FRI domain length was {}. \
                    Unsatisfied constraint: {}",
                    quotient_codewords.len(),
                    self.name(),
                    interpolated.degree(),
                    fri_domain.length,
                    terminal_constraints[idx]
                );
            }
        }

        quotient_codewords
    }

    fn boundary_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        codewords: &[Vec<XWord>],
        challenges: &AllChallenges,
    ) -> Vec<Vec<XWord>> {
        assert!(!codewords.is_empty(), "Codewords must be non-empty");
        for row in codewords.iter() {
            debug_assert_eq!(
                fri_domain.length,
                row.len(),
                "Codewords have fri_domain.length columns ({}), not {}.",
                fri_domain.length,
                row.len()
            );
        }

        let mut quotient_codewords: Vec<Vec<XWord>> = vec![];

        let boundary_constraints: Vec<MPolynomial<XWord>> =
            self.ext_boundary_constraints(challenges);
        let one = XWord::ring_one();
        let zerofier: Vec<XWord> = (0..fri_domain.length)
            .map(|i| fri_domain.domain_value(i as u32) - one)
            .collect();
        let zerofier_inverse = XWord::batch_inversion(zerofier);

        for bc in boundary_constraints.iter() {
            let quotient_codeword: Vec<XWord> = (0..fri_domain.length)
                .into_iter()
                .map(|fri_dom_i| {
                    let point: Vec<XWord> = (0..self.full_width())
                        .map(|j| codewords[j][fri_dom_i])
                        .collect();
                    bc.evaluate(&point) * zerofier_inverse[fri_dom_i]
                })
                .collect();
            quotient_codewords.push(quotient_codeword);
        }

        // If the `DEBUG` environment variable is set, run this extra validity check
        if std::env::var("DEBUG").is_ok() {
            for (idx, qc) in quotient_codewords.iter().enumerate() {
                let interpolated = fri_domain.interpolate(qc);
                assert!(
                    interpolated.degree() < fri_domain.length as isize - 1,
                    "Degree of boundary quotient number {idx} (of {}) in {} must not be maximal. \
                    Got degree {}, and FRI domain length was {}.\
                    Unsatisfied constraint: {}",
                    quotient_codewords.len(),
                    self.name(),
                    interpolated.degree(),
                    fri_domain.length,
                    boundary_constraints[idx]
                );
            }
        }

        quotient_codewords
    }
}
