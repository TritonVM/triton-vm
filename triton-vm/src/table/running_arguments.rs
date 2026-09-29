//! Parallel computation of the “running” columns of the auxiliary tables:
//! running evaluations, running products, running sums, and log derivatives.
//!
//! All of these are prefix computations over the rows of a table. Computing
//! them row by row is inherently sequential, and, for log derivatives,
//! requires one field inversion per row. Instead, the per-row contributions
//! are computed in parallel, inversions are batched, and the prefix
//! computation is a parallel scan.

use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::Zero;
use rayon::prelude::*;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

/// The running evaluation with the given `indeterminate`, starting from
/// `initial`, where `addends[i]` being `Some(x)` means that row `i` updates
/// the running evaluation to `running_evaluation · indeterminate + x`, and
/// `None` means it leaves it unchanged. The returned vector holds the value
/// of the running evaluation _after_ processing each row.
pub(crate) fn par_running_evaluation(
    addends: &[Option<XFieldElement>],
    indeterminate: XFieldElement,
    initial: XFieldElement,
) -> Vec<XFieldElement> {
    let affine_maps = addends
        .par_iter()
        .map(|addend| match addend {
            Some(x) => (indeterminate, *x),
            None => (XFieldElement::ONE, XFieldElement::ZERO),
        })
        .collect();
    par_running_affine_maps(affine_maps, initial)
}

/// The running value starting from `initial`, where row `i` updates the
/// running value `acc` to `a · acc + b` for `affine_maps[i] == (a, b)`. The
/// returned vector holds the running value _after_ processing each row.
///
/// This generalizes [`par_running_evaluation`]: a row that contributes
/// several symbols to a running evaluation is one affine map, namely the
/// composition of the symbols' individual updates.
pub(crate) fn par_running_affine_maps(
    mut affine_maps: Vec<(XFieldElement, XFieldElement)>,
    initial: XFieldElement,
) -> Vec<XFieldElement> {
    // Composing the maps of all rows up to and including row `i` gives the
    // affine map that takes the initial value to the running value at row `i`.
    par_scan(&mut affine_maps, |(a_prev, b_prev), (a_next, b_next)| {
        (a_next * a_prev, a_next * b_prev + b_next)
    });
    affine_maps
        .into_par_iter()
        .map(|(a, b)| a * initial + b)
        .collect()
}

/// The running product starting from `initial`, where `factors[i]` being
/// `Some(f)` means that row `i` multiplies the running product by `f`, and
/// `None` means it leaves it unchanged. The returned vector holds the value
/// of the running product _after_ processing each row.
pub(crate) fn par_running_product(
    factors: &[Option<XFieldElement>],
    initial: XFieldElement,
) -> Vec<XFieldElement> {
    let mut products = factors
        .par_iter()
        .map(|factor| factor.unwrap_or(XFieldElement::ONE))
        .collect::<Vec<_>>();
    par_scan(&mut products, |prev, next| prev * next);
    if initial != XFieldElement::ONE {
        products.par_iter_mut().for_each(|p| *p *= initial);
    }
    products
}

/// The running sum starting from `initial`. The returned vector holds the
/// value of the running sum _after_ processing each row.
pub(crate) fn par_running_sum(
    mut summands: Vec<XFieldElement>,
    initial: XFieldElement,
) -> Vec<XFieldElement> {
    par_scan(&mut summands, |prev, next| prev + next);
    if !initial.is_zero() {
        summands.par_iter_mut().for_each(|s| *s += initial);
    }
    summands
}

/// The log derivative, i.e., the running sum of fractions, starting from
/// `initial`. An entry `Some((denominator, numerator))` in `fractions` means
/// that the corresponding row adds `numerator / denominator` to the running
/// sum, `None` means it leaves it unchanged. The returned vector holds the
/// value of the running sum _after_ processing each row.
///
/// The inversions are batched.
///
/// # Panics
///
/// Panics if any of the denominators is zero.
pub(crate) fn par_log_derivative(
    fractions: &[Option<(XFieldElement, XFieldElement)>],
    initial: XFieldElement,
) -> Vec<XFieldElement> {
    par_running_sum(par_fractions(fractions), initial)
}

/// The value of each fraction, i.e., `numerator / denominator`, or zero for
/// `None`. The inversions are batched.
///
/// # Panics
///
/// Panics if any of the denominators is zero.
pub(crate) fn par_fractions(
    fractions: &[Option<(XFieldElement, XFieldElement)>],
) -> Vec<XFieldElement> {
    let denominators = fractions
        .par_iter()
        .map(|fraction| fraction.map_or(XFieldElement::ONE, |(denominator, _)| denominator))
        .collect();
    let denominator_inverses = XFieldElement::par_batch_inversion(denominators);
    fractions
        .par_iter()
        .zip(denominator_inverses)
        .map(|(fraction, denominator_inverse)| {
            fraction.map_or(XFieldElement::ZERO, |(_, numerator)| {
                numerator * denominator_inverse
            })
        })
        .collect()
}

/// In-place, parallel, inclusive prefix scan with the associative operation
/// `combine`. Afterwards, `xs[i] == combine(…combine(combine(xs[0], xs[1]),
/// xs[2])…, xs[i])`.
///
/// The operation does not need to be commutative: its first argument is
/// always the combination of the earlier elements, its second argument the
/// combination of the later elements.
pub(crate) fn par_scan<T, F>(xs: &mut [T], combine: F)
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Sync,
{
    // Few, large chunks keep the sequential second pass short; the lower
    // bound keeps the per-chunk overhead negligible for short inputs.
    const MIN_CHUNK_LEN: usize = 1 << 10;
    let num_chunks = 4 * rayon::current_num_threads().max(1);
    let chunk_len = xs.len().div_ceil(num_chunks).max(MIN_CHUNK_LEN);

    // 1. scan every chunk independently; remember each chunk's total
    let chunk_totals = xs
        .par_chunks_mut(chunk_len)
        .map(|chunk| {
            for i in 1..chunk.len() {
                chunk[i] = combine(chunk[i - 1], chunk[i]);
            }
            chunk[chunk.len() - 1]
        })
        .collect::<Vec<_>>();

    // 2. exclusive prefix over the chunk totals
    let mut offsets = Vec::with_capacity(chunk_totals.len());
    let mut offset = None;
    for total in chunk_totals {
        offsets.push(offset);
        offset = Some(match offset {
            Some(offset) => combine(offset, total),
            None => total,
        });
    }

    // 3. fold each chunk's offset into its elements
    xs.par_chunks_mut(chunk_len)
        .zip(offsets)
        .for_each(|(chunk, offset)| {
            if let Some(offset) = offset {
                for x in chunk {
                    *x = combine(offset, *x);
                }
            }
        });
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_adapter::arb;

    use super::*;
    use crate::tests::proptest;

    /// Lengths that exercise the empty case, a single chunk, and chunk
    /// boundaries of [`par_scan`].
    fn interesting_lengths() -> impl Strategy<Value = usize> {
        prop_oneof![
            Just(0),
            Just(1),
            Just(2),
            1_usize..100,
            Just((1 << 10) - 1),
            Just(1 << 10),
            Just((1 << 10) + 1),
            Just((1 << 12) + 7),
        ]
    }

    #[macro_rules_attr::apply(proptest)]
    fn scan_agrees_with_sequential_prefix_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] xs: Vec<(XFieldElement, XFieldElement)>,
    ) {
        // composition of affine maps: associative, but not commutative
        let combine = |(a_prev, b_prev): (XFieldElement, XFieldElement),
                       (a_next, b_next): (XFieldElement, XFieldElement)| {
            (a_next * a_prev, a_next * b_prev + b_next)
        };
        let mut expected = xs.clone();
        for i in 1..expected.len() {
            expected[i] = combine(expected[i - 1], expected[i]);
        }

        let mut actual = xs;
        par_scan(&mut actual, combine);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn running_evaluation_agrees_with_sequential_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] addends: Vec<Option<XFieldElement>>,
        #[strategy(arb())] indeterminate: XFieldElement,
        #[strategy(arb())] initial: XFieldElement,
    ) {
        let mut acc = initial;
        let mut expected = Vec::with_capacity(addends.len());
        for addend in &addends {
            if let Some(x) = addend {
                acc = acc * indeterminate + *x;
            }
            expected.push(acc);
        }

        let actual = par_running_evaluation(&addends, indeterminate, initial);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn running_product_agrees_with_sequential_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] factors: Vec<Option<XFieldElement>>,
        #[strategy(arb())] initial: XFieldElement,
    ) {
        let mut acc = initial;
        let mut expected = Vec::with_capacity(factors.len());
        for factor in &factors {
            if let Some(f) = factor {
                acc *= *f;
            }
            expected.push(acc);
        }

        let actual = par_running_product(&factors, initial);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn running_affine_maps_agree_with_sequential_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] maps: Vec<(XFieldElement, XFieldElement)>,
        #[strategy(arb())] initial: XFieldElement,
    ) {
        let mut acc = initial;
        let mut expected = Vec::with_capacity(maps.len());
        for &(a, b) in &maps {
            acc = a * acc + b;
            expected.push(acc);
        }

        let actual = par_running_affine_maps(maps, initial);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn log_derivative_agrees_with_sequential_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] fractions: Vec<Option<(XFieldElement, XFieldElement)>>,
        #[strategy(arb())] initial: XFieldElement,
    ) {
        let fractions = fractions
            .into_iter()
            .map(|f| f.filter(|(denominator, _)| !denominator.is_zero()))
            .collect_vec();

        let mut acc = initial;
        let mut expected = Vec::with_capacity(fractions.len());
        for fraction in &fractions {
            if let Some((denominator, numerator)) = fraction {
                acc += *numerator * denominator.inverse();
            }
            expected.push(acc);
        }

        let actual = par_log_derivative(&fractions, initial);
        prop_assert_eq!(expected, actual);
    }
}
