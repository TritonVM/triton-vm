//! Checks that the constants and computed bounds in the specification's
//! chapter on soundness agree with the code.
//!
//! The chapter contains two tables delimited by HTML comments. The first lists
//! the protocol constants that the soundness analysis depends on; this test
//! compares each against its source in the code. The second lists the
//! resulting soundness errors; this test recomputes them from the formulas of
//! the chapter and compares the results up to the precision printed there.

use std::collections::HashMap;
use std::path::Path;

use air::table_column::HashAuxColumn;
use strum::IntoEnumIterator;
use twenty_first::tip5;

use super::*;
use crate::table::master_table::MasterAuxTable;
use crate::tests::test;

const SPEC_PATH: &str = "../specification/src/soundness.md";
const CONSTANTS_MARKERS: [&str; 2] = [
    "<!-- soundness-constants start -->",
    "<!-- soundness-constants stop -->",
];
const SENSITIVITY_MARKERS: [&str; 2] = [
    "<!-- soundness-sensitivity start -->",
    "<!-- soundness-sensitivity stop -->",
];
const RESULTS_MARKERS: [&str; 2] = [
    "<!-- soundness-results start -->",
    "<!-- soundness-results stop -->",
];

/// The trace lengths (as log₂) for which the chapter lists results, in the
/// order of the columns of the results table: worst case, then average case.
const LOG2_TRACE_LENGTHS: [u32; 2] = [MAX_LOG2_PADDED_HEIGHT, 21];

/// A row of a markdown table: the key in the first cell (in backticks), and
/// the numeric cells, each with the number of decimal places it is written
/// with.
type Row = Vec<(f64, usize)>;

fn read_table(spec: &str, [start, stop]: [&str; 2]) -> HashMap<String, Row> {
    let start = spec.find(start).expect("start marker must exist") + start.len();
    let stop = spec.find(stop).expect("stop marker must exist");

    let mut rows = HashMap::new();
    for line in spec[start..stop].lines() {
        let cells = line.split('|').map(str::trim).collect_vec();
        let Some(key) = cells.get(1) else { continue };
        let Some(key) = key.strip_prefix('`').and_then(|k| k.strip_suffix('`')) else {
            continue;
        };
        let values = cells[2..]
            .iter()
            .filter_map(|cell| {
                let value = cell.parse::<f64>().ok()?;
                let decimals = cell.split_once('.').map_or(0, |(_, frac)| frac.len());
                Some((value, decimals))
            })
            .collect();
        rows.insert(key.to_string(), values);
    }
    rows
}

fn binary_entropy(x: f64) -> f64 {
    -x * x.log2() - (1.0 - x) * (1.0 - x).log2()
}

/// The `Q`-ary entropy function `H_Q(x)`, with `log2_q` = log₂(Q).
fn q_ary_entropy(x: f64, log2_q: f64) -> f64 {
    // log₂(Q − 1) = log₂(Q) for the fields in question, up to f64 precision.
    (x * log2_q + binary_entropy(x)) / log2_q
}

/// The Elias radius `r` defined by `rate = 1 − H_Q(r)`, by bisection.
fn elias_radius(rate: f64, log2_q: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1.0 - rate);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if 1.0 - q_ary_entropy(mid, log2_q) > rate {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// All quantities of the chapter's results table, for one regime and one
/// trace length. Errors are given as log₂.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    log2_eta: f64,
    log2_list_size: f64,
    eps_challenges: f64,
    eps_quotient_weights: f64,
    eps_ood_point: f64,
    eps_deep_weights: f64,
    eps_folding: f64,
    eps_queries: f64,
    eps_last_round: f64,
    /// The sum of the errors of all folding rounds, as log₂.
    eps_folding_sum: f64,
    /// The failure probability of the extractor, as log₂.
    eps_extractor: f64,
    num_folding_rounds: f64,
}

impl Bounds {
    fn errors(&self) -> [f64; 7] {
        [
            self.eps_challenges,
            self.eps_quotient_weights,
            self.eps_ood_point,
            self.eps_deep_weights,
            self.eps_folding,
            self.eps_queries,
            self.eps_last_round,
        ]
    }

    /// The round-by-round soundness error, as log₂.
    fn eps_rbr(&self) -> f64 {
        self.errors().into_iter().fold(f64::MIN, f64::max)
    }

    /// The (knowledge) soundness error of the IOP, the sum of the per-round
    /// errors with one folding error per folding round, as log₂.
    fn eps_iop(&self) -> f64 {
        let others = [
            self.eps_folding_sum,
            self.eps_extractor,
            self.eps_challenges,
            self.eps_quotient_weights,
            self.eps_ood_point,
            self.eps_deep_weights,
            self.eps_queries,
            self.eps_last_round,
        ];
        others.into_iter().map(f64::exp2).sum::<f64>().log2()
    }
}

/// The form of the conjectured bound on the number of exceptional challenges
/// per line, for code length m, rate ρ, and slack η.
#[derive(Debug, Clone, Copy)]
enum Form {
    /// S-two whitepaper, Appendix A.5: m · c₁ · 2^(c₂ H₂(ρ)/η).
    Exponential { c1: f64, c2: f64 },
    /// BCIKS20, Conjecture 8.4: m^e_n / (ηρ)^e_η.
    Polynomial { e_eta: f64, e_n: f64 },
}

impl Form {
    fn exceptional(self, m: f64, eta: f64, rate: f64) -> f64 {
        match self {
            Form::Exponential { c1, c2 } => m * c1 * (c2 * binary_entropy(rate) / eta).exp2(),
            Form::Polynomial { e_eta, e_n } => m.powf(e_n) / (eta * rate).powf(e_eta),
        }
    }
}

/// Round up to two decimals.
fn ceil2(x: f64) -> f64 {
    (x * 100.0).ceil() / 100.0
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Regime {
    Proven,
    Conjectured,
}

/// The protocol parameters the bounds depend on.
#[derive(Debug, Clone, Copy)]
struct Params {
    log2_field: f64,
    num_queries: f64,
    num_segments: f64,
    trace_len: f64,
    /// ρ|D|
    randomized_trace_len: f64,
    /// |D|
    fri_domain_len: f64,
    last_fri_domain_len: f64,
    num_folding_rounds: f64,
    /// The maximum over all table-linking arguments of the number of terms
    /// plus the degree of the defect polynomial, divided by the trace length.
    max_link_terms_per_row: f64,
    /// The form of the conjecture, with its constants.
    form: Form,
}

impl Params {
    fn log2_ext_field(&self) -> f64 {
        3.0 * self.log2_field
    }

    /// The chapter's error terms for a given slack η and list size ℓ,
    /// exceptional count B(m) for code length m, and agreement A.
    fn bounds(
        &self,
        eta: f64,
        list_size: f64,
        exceptional: impl Fn(f64) -> f64,
        agreement: f64,
    ) -> Bounds {
        let log2_e = self.log2_ext_field();
        let n = self.fri_domain_len;
        let k = self.num_segments;
        let l = list_size;
        let qe_degree = self.trace_len + k * self.randomized_trace_len + k - 1.0;
        let ood_exclusions = self.trace_len + (2.0 + 2.0 * k) * n;

        // Folding round r tests a line on the domain D_r of length |D|/2^r.
        let fold_lens = (1..=self.num_folding_rounds as i32).map(|r| n / 2_f64.powi(r));
        let fold_sum = fold_lens.map(&exceptional).sum::<f64>();

        Bounds {
            log2_eta: eta.log2(),
            log2_list_size: l.log2(),
            eps_challenges: (l * self.max_link_terms_per_row * self.trace_len).log2() - log2_e,
            eps_quotient_weights: l.log2() - log2_e,
            eps_ood_point: (l * qe_degree + ood_exclusions).log2() - log2_e,
            // The false witness probability is B/(|E| − 1) for exceptional
            // count B; the − 1 is invisible at f64 precision.
            eps_deep_weights: (2.0 * exceptional(n)).log2() - log2_e,
            eps_folding: exceptional(n / 2.0).log2() - log2_e,
            eps_queries: self.num_queries * (agreement / n).log2(),
            eps_last_round: (self.last_fri_domain_len - 1.0).log2() - log2_e,
            eps_folding_sum: fold_sum.log2() - log2_e,
            eps_extractor: (l * n).log2() - log2_e,
            num_folding_rounds: self.num_folding_rounds,
        }
    }

    /// Dao–Kominers–Thaler, Theorem 5.12, plus the Johnson bound.
    fn proven(&self, eta: f64) -> Option<Bounds> {
        let n = self.fri_domain_len;
        let rho_minus = (self.randomized_trace_len - 1.0) / n;
        let agreement = ((rho_minus.sqrt() + eta) * n).ceil();
        let rate = self.randomized_trace_len / n;
        let exceptional = |m: f64| {
            let rho_minus = (rate * m - 1.0) / m;
            let theta = (rho_minus.sqrt() / (2.0 * eta)).ceil().max(3.0) + 0.5;
            8.0 / 3.0 * m * theta.powi(3) / rho_minus
        };

        // list size for polynomials of degree at most ρ|D|
        let list_denominator = agreement * agreement - n * self.randomized_trace_len;
        if list_denominator <= 0.0 {
            return None;
        }
        let list_size = n * (agreement - self.randomized_trace_len) / list_denominator;
        Some(self.bounds(eta, list_size, exceptional, agreement))
    }

    /// The conjecture of section “Conjectured Regime” of `soundness.md`.
    fn conjectured(&self, eta: f64) -> Option<Bounds> {
        let rate = self.randomized_trace_len / self.fri_domain_len;
        let eta_elias = 1.0 - rate - elias_radius(rate, self.log2_field);
        if eta < eta_elias {
            return None;
        }
        let n = self.fri_domain_len;
        let exceptional = |m: f64| self.form.exceptional(m, eta, rate);
        let list_size = exceptional(n) / n;
        let agreement = (rate + eta) * self.fri_domain_len;
        Some(self.bounds(eta, list_size, exceptional, agreement))
    }

    /// Minimize the soundness error of the IOP over η ∈ {2^(−j/50)}.
    fn optimal(&self, regime: Regime) -> Bounds {
        (50..1500)
            .map(|j| (-f64::from(j) / 50.0).exp2())
            .filter_map(|eta| match regime {
                Regime::Proven => self.proven(eta),
                Regime::Conjectured => self.conjectured(eta),
            })
            .reduce(|best, b| {
                if b.eps_iop() < best.eps_iop() {
                    b
                } else {
                    best
                }
            })
            .unwrap()
    }
}

fn constant(constants: &HashMap<String, Row>, key: &str) -> f64 {
    let row = constants
        .get(key)
        .unwrap_or_else(|| panic!("constant `{key}` must be listed"));
    assert_eq!(1, row.len(), "constant `{key}` must have exactly one value");
    row[0].0
}

#[macro_rules_attr::apply(test)]
fn soundness_chapter_agrees_with_code() {
    let spec = std::fs::read_to_string(Path::new(SPEC_PATH)).unwrap();
    let constants = read_table(&spec, CONSTANTS_MARKERS);
    let results = read_table(&spec, RESULTS_MARKERS);
    let sensitivity = read_table(&spec, SENSITIVITY_MARKERS);
    let c = |key| constant(&constants, key);

    // constants that are taken from the code
    let stark = Stark::default();
    let num_cascade_clients = HashAuxColumn::iter()
        .filter(|col| {
            let name = col.to_string();
            name.starts_with("CascadeState") && name.ends_with("ClientLogDerivative")
        })
        .count();
    let tip5_capacity = tip5::STATE_SIZE - tip5::RATE;
    let max_fri = stark.fri(1 << MAX_LOG2_PADDED_HEIGHT).unwrap();
    let last_fri_domain_len = max_fri.domain.len() >> max_fri.num_rounds();

    let from_code = [
        ("p", BFieldElement::P as f64),
        ("extension_degree", x_field_element::EXTENSION_DEGREE as f64),
        ("num_queries", stark.num_collinearity_checks as f64),
        ("fri_expansion_factor", stark.fri_expansion_factor as f64),
        ("num_trace_randomizers", stark.num_trace_randomizers as f64),
        ("num_quotient_segments", NUM_QUOTIENT_SEGMENTS as f64),
        ("air_fan_in", AIR_FAN_IN as f64),
        ("constraint_degree", air::TARGET_DEGREE as f64),
        ("num_constraints", MasterAuxTable::NUM_CONSTRAINTS as f64),
        ("num_cascade_clients", num_cascade_clients as f64),
        ("max_log2_padded_height", f64::from(MAX_LOG2_PADDED_HEIGHT)),
        ("last_fri_domain_len", last_fri_domain_len as f64),
        ("tip5_digest_len", Digest::LEN as f64),
        ("tip5_capacity", tip5_capacity as f64),
    ];
    for (key, value) in from_code {
        assert_eq!(value, c(key), "constant `{key}` disagrees with the code");
    }

    // The quotient-equation degree bound uses that the number of quotient
    // segments equals the constraint degree. The chapter states this.
    assert_eq!(NUM_QUOTIENT_SEGMENTS as isize, air::TARGET_DEGREE);

    // The conjecture's constants, fitted to the published attack: the smallest
    // values, rounded up to two decimals, whose threshold the attack's number of
    // bad challenges does not exceed.
    let attack_radius = c("attack_unsafe_index") / c("attack_domain_len");
    let attack_eta = 1.0 - c("attack_rate") - attack_radius;
    let attack_log2_n = c("attack_domain_len").log2();
    let attack_log2_bad = c("attack_log2_bad_challenges");
    let attack_log2_inv_eta_rho = -(attack_eta * c("attack_rate")).log2();
    let exponential_c2 =
        ceil2((attack_log2_bad - attack_log2_n) * attack_eta / binary_entropy(c("attack_rate")));
    let polynomial_e_eta =
        |e_n: f64| ceil2((attack_log2_bad - attack_log2_n * e_n) / attack_log2_inv_eta_rho);
    let equal_e = ceil2(attack_log2_bad / (attack_log2_n + attack_log2_inv_eta_rho));
    let forms = [
        (
            "exponential",
            Form::Exponential {
                c1: 1.0,
                c2: exponential_c2,
            },
        ),
        (
            "polynomial_linear",
            Form::Polynomial {
                e_eta: polynomial_e_eta(1.0),
                e_n: 1.0,
            },
        ),
        (
            "polynomial_en_1.5",
            Form::Polynomial {
                e_eta: polynomial_e_eta(1.5),
                e_n: 1.5,
            },
        ),
        (
            "polynomial_quadratic",
            Form::Polynomial {
                e_eta: polynomial_e_eta(2.0),
                e_n: 2.0,
            },
        ),
        (
            "polynomial_equal",
            Form::Polynomial {
                e_eta: equal_e,
                e_n: equal_e,
            },
        ),
    ];
    assert_eq!(exponential_c2, c("c2"), "c₂ must be fitted to the attack");
    let form = forms[0].1;

    let log2_field = (BFieldElement::P as f64).log2();
    let max_link_terms_per_row = 2.0 * (c("num_cascade_clients") + 1.0);

    let mut computed: HashMap<&str, Vec<f64>> = HashMap::new();
    for log2_trace_len in LOG2_TRACE_LENGTHS {
        let fri = stark.fri(1 << log2_trace_len).unwrap();
        let randomized_trace_len =
            Stark::randomized_trace_len(1 << log2_trace_len, stark.num_trace_randomizers);
        let params = Params {
            log2_field,
            num_queries: c("num_queries"),
            num_segments: c("num_quotient_segments"),
            trace_len: (randomized_trace_len / 2) as f64,
            randomized_trace_len: randomized_trace_len as f64,
            fri_domain_len: fri.domain.len() as f64,
            last_fri_domain_len: (fri.domain.len() >> fri.num_rounds()) as f64,
            num_folding_rounds: fri.num_rounds() as f64,
            max_link_terms_per_row,
            form,
        };
        for (key, form) in forms {
            let b = Params { form, ..params }.optimal(Regime::Conjectured);
            let (c1, c2) = match form {
                Form::Exponential { c1, c2 } => (c1, c2),
                Form::Polynomial { e_eta, e_n } => (e_eta, e_n),
            };
            let row = sensitivity
                .get(key)
                .unwrap_or_else(|| panic!("form `{key}` must be listed"));
            assert_eq!(4, row.len(), "form `{key}`: wrong number of columns");
            assert_eq!((c1, c2), (row[0].0, row[1].0), "form `{key}`: constants");
            let column = if log2_trace_len == MAX_LOG2_PADDED_HEIGHT {
                2
            } else {
                3
            };
            let (doc_value, decimals) = row[column];
            let tolerance = 0.5 * 10_f64.powi(-(decimals as i32)) + 1e-9;
            assert!(
                (b.eps_iop() - doc_value).abs() <= tolerance,
                "form `{key}`, log₂ N = {log2_trace_len}: chapter says {doc_value}, computed {}",
                b.eps_iop()
            );
        }
        for regime in [Regime::Proven, Regime::Conjectured] {
            let b = params.optimal(regime);
            let entries = [
                ("log2_eta", b.log2_eta),
                ("log2_list_size", b.log2_list_size),
                ("eps_challenges", b.eps_challenges),
                ("eps_quotient_weights", b.eps_quotient_weights),
                ("eps_ood_point", b.eps_ood_point),
                ("eps_deep_weights", b.eps_deep_weights),
                ("eps_folding", b.eps_folding),
                ("eps_folding_sum", b.eps_folding_sum),
                ("eps_extractor", b.eps_extractor),
                ("eps_queries", b.eps_queries),
                ("eps_last_round", b.eps_last_round),
                ("eps_rbr", b.eps_rbr()),
                ("eps_iop", b.eps_iop()),
                ("num_folding_rounds", b.num_folding_rounds),
            ];
            for (key, value) in entries {
                computed.entry(key).or_default().push(value);
            }
        }
    }

    let report = computed
        .iter()
        .sorted_by_key(|(k, _)| *k)
        .map(|(k, v)| format!("{k}: {}", v.iter().map(|x| format!("{x:.3}")).join(" | ")))
        .join("\n");
    for (key, values) in &computed {
        let row = results
            .get(*key)
            .unwrap_or_else(|| panic!("result `{key}` must be listed\n{report}"));
        assert_eq!(
            values.len(),
            row.len(),
            "result `{key}`: wrong number of columns"
        );
        for (&value, &(doc_value, decimals)) in values.iter().zip(row) {
            let tolerance = 0.5 * 10_f64.powi(-(decimals as i32)) + 1e-9;
            assert!(
                (value - doc_value).abs() <= tolerance,
                "result `{key}`: chapter says {doc_value}, computed {value}\n{report}"
            );
        }
    }
}
