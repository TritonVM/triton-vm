//! Parallel computation of the “running” columns of the auxiliary tables:
//! running evaluations, running products, running sums, and log derivatives.
//!
//! All of these are prefix computations over the rows of a table. Computing
//! them row by row is inherently sequential, and, for log derivatives,
//! requires one field inversion per row. Instead, the rows are processed in
//! chunks, in parallel, and the inversions are batched. The per-row
//! contributions are never materialized: every column is written directly,
//! and the only other memory traffic is reading the main table.

use std::ops::Range;

use ndarray::prelude::*;
use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::Zero;
use rayon::prelude::*;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

/// Run the given function with the column as a contiguous slice. The columns
/// of a column-major table are contiguous; any other column is written to
/// through an intermediate buffer.
pub(crate) fn with_contiguous_column(
    mut column: ArrayViewMut1<XFieldElement>,
    f: impl FnOnce(&mut [XFieldElement]),
) {
    if let Some(column) = column.as_slice_mut() {
        return f(column);
    }
    let mut buffer = vec![XFieldElement::ZERO; column.len()];
    f(&mut buffer);
    column.assign(&ArrayView1::from(&buffer));
}

/// The number of rows per chunk for prefix computations over `num_rows` rows.
///
/// Few, large chunks keep the sequential pass over the chunks short; the lower
/// bound keeps the per-chunk overhead negligible for short columns.
fn prefix_chunk_len(num_rows: usize) -> usize {
    const MIN_CHUNK_LEN: usize = 1 << 10;

    let num_chunks = 4 * rayon::current_num_threads().max(1);
    num_rows.div_ceil(num_chunks).max(MIN_CHUNK_LEN)
}

/// Fill the `column` with a running value that starts from `initial` and is
/// updated row by row, in two parallel passes over chunks of rows:
///
/// 1. `summarize` condenses the updates of a range of rows, such that
///    `advance` can apply them all at once to the running value from before
///    the range. This gives the running value from before every chunk.
/// 2. `fill` writes the running value _after_ every row of a range of rows,
///    given the running value from before the range.
///
/// Neither pass materializes the updates: the rows' updates are computed
/// twice, which is a lot cheaper than passing them through memory.
fn par_running_value_into<S: Send>(
    column: &mut [XFieldElement],
    initial: XFieldElement,
    summarize: impl Fn(Range<usize>) -> S + Sync,
    advance: impl Fn(XFieldElement, S) -> XFieldElement,
    fill: impl Fn(Range<usize>, XFieldElement, &mut [XFieldElement]) + Sync,
) {
    let num_rows = column.len();
    let chunk_len = prefix_chunk_len(num_rows);
    let rows_of_chunk = |chunk_idx: usize| {
        let first_row = chunk_idx * chunk_len;
        first_row..(first_row + chunk_len).min(num_rows)
    };

    // the last chunk's summary is not needed
    let num_chunks = num_rows.div_ceil(chunk_len);
    let summaries = (0..num_chunks.saturating_sub(1))
        .into_par_iter()
        .map(|chunk_idx| summarize(rows_of_chunk(chunk_idx)))
        .collect::<Vec<_>>();

    let mut running_value = initial;
    let mut initial_values = vec![initial];
    for summary in summaries {
        running_value = advance(running_value, summary);
        initial_values.push(running_value);
    }

    column
        .par_chunks_mut(chunk_len)
        .zip(initial_values)
        .enumerate()
        .for_each(|(chunk_idx, (chunk, initial))| fill(rows_of_chunk(chunk_idx), initial, chunk));
}

/// Fill the `column` with the running product starting from `initial`, where
/// `factor(i)` being `Some(f)` means that row `i` multiplies the running
/// product by `f`, and `None` means it leaves it unchanged. The column holds
/// the value of the running product _after_ processing each row.
pub(crate) fn par_running_product_into(
    column: &mut [XFieldElement],
    initial: XFieldElement,
    factor: impl Fn(usize) -> Option<XFieldElement> + Sync,
) {
    par_running_value_into(
        column,
        initial,
        |rows| {
            rows.filter_map(&factor)
                .fold(XFieldElement::ONE, |product, factor| product * factor)
        },
        |running_product, product| running_product * product,
        |rows, mut running_product, chunk| {
            for (row, cell) in rows.zip(chunk) {
                if let Some(factor) = factor(row) {
                    running_product *= factor;
                }
                *cell = running_product;
            }
        },
    );
}

/// Fill the `column` with the running value starting from `initial`, where
/// row `i` updates the running value `acc` to `a · acc + b` for
/// `affine_map(i) == Some((a, b))`, and leaves it unchanged for `None`. The
/// column holds the running value _after_ processing each row.
///
/// A running evaluation with indeterminate `x` is the special case where
/// every row that absorbs a symbol `s` has the map `(x, s)`. A row that
/// absorbs several symbols has the composition of the symbols' maps.
pub(crate) fn par_running_affine_maps_into(
    column: &mut [XFieldElement],
    initial: XFieldElement,
    affine_map: impl Fn(usize) -> Option<(XFieldElement, XFieldElement)> + Sync,
) {
    par_running_value_into(
        column,
        initial,
        |rows| {
            // the composition of the rows' maps
            let identity = (XFieldElement::ONE, XFieldElement::ZERO);
            rows.filter_map(&affine_map)
                .fold(identity, |(a_acc, b_acc), (a, b)| {
                    (a * a_acc, a * b_acc + b)
                })
        },
        |running_value, (a, b)| a * running_value + b,
        |rows, mut running_value, chunk| {
            for (row, cell) in rows.zip(chunk) {
                if let Some((a, b)) = affine_map(row) {
                    running_value = a * running_value + b;
                }
                *cell = running_value;
            }
        },
    );
}

/// Fill the `column` with the running evaluation with the given
/// `indeterminate`, starting from `initial`, where `addend(i)` being `Some(x)`
/// means that row `i` updates the running evaluation to `running_evaluation ·
/// indeterminate + x`, and `None` means it leaves it unchanged. The column
/// holds the value of the running evaluation _after_ processing each row.
pub(crate) fn par_running_evaluation_into(
    column: &mut [XFieldElement],
    indeterminate: XFieldElement,
    initial: XFieldElement,
    addend: impl Fn(usize) -> Option<XFieldElement> + Sync,
) {
    par_running_affine_maps_into(column, initial, |row_idx| {
        addend(row_idx).map(|addend| (indeterminate, addend))
    });
}

/// Fill the `column` with the log derivative, i.e., the running sum of
/// fractions, starting from `initial`. An entry `Some((denominator,
/// numerator))` in `fractions(i)` means that row `i` adds `numerator /
/// denominator` to the running sum, `None` means it adds nothing. The column
/// holds the value of the running sum _after_ processing each row.
///
/// The inversions are batched.
///
/// # Panics
///
/// Panics if any of the denominators is zero.
pub(crate) fn par_log_derivative_into<const N: usize>(
    column: &mut [XFieldElement],
    initial: XFieldElement,
    fractions: impl Fn(usize) -> [Option<(XFieldElement, XFieldElement)>; N] + Sync,
) {
    /// The number of rows whose inversions are batched. The denominators,
    /// their inverses, and the numerators fit into the L2 cache.
    const ROWS_PER_BATCH: usize = 1 << 9;

    // 1. every chunk's running sum, starting from zero
    let chunk_len = prefix_chunk_len(column.len());
    let chunk_totals = column
        .par_chunks_mut(chunk_len)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut running_sum = XFieldElement::ZERO;
            let mut denominators = Vec::with_capacity(N * ROWS_PER_BATCH);
            let mut numerators = Vec::with_capacity(N * ROWS_PER_BATCH);
            let mut num_fractions_of_rows = Vec::with_capacity(ROWS_PER_BATCH);
            let first_rows = (chunk_idx * chunk_len..).step_by(ROWS_PER_BATCH);
            for (first_row, batch) in first_rows.zip(chunk.chunks_mut(ROWS_PER_BATCH)) {
                denominators.clear();
                numerators.clear();
                num_fractions_of_rows.clear();
                for row in first_row..first_row + batch.len() {
                    let num_fractions_before = denominators.len();
                    for (denominator, numerator) in fractions(row).into_iter().flatten() {
                        denominators.push(denominator);
                        numerators.push(numerator);
                    }
                    num_fractions_of_rows.push(denominators.len() - num_fractions_before);
                }

                let inverses = XFieldElement::batch_inversion(std::mem::take(&mut denominators));
                let mut summands = numerators.iter().zip(&inverses).map(|(&n, &i)| n * i);
                for (cell, &num_fractions) in batch.iter_mut().zip(&num_fractions_of_rows) {
                    for summand in summands.by_ref().take(num_fractions) {
                        running_sum += summand;
                    }
                    *cell = running_sum;
                }
                denominators = inverses;
            }
            running_sum
        })
        .collect::<Vec<_>>();

    // 2. the running sum from before every chunk
    let mut running_sum = initial;
    let mut initial_values = Vec::with_capacity(chunk_totals.len());
    for chunk_total in chunk_totals {
        initial_values.push(running_sum);
        running_sum += chunk_total;
    }

    // 3. every chunk's running sum, starting from the correct value
    column
        .par_chunks_mut(chunk_len)
        .zip(initial_values)
        .filter(|(_, initial)| !initial.is_zero())
        .for_each(|(chunk, initial)| chunk.iter_mut().for_each(|cell| *cell += initial));
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
    /// boundaries.
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

        let mut actual = vec![XFieldElement::ZERO; addends.len()];
        par_running_evaluation_into(&mut actual, indeterminate, initial, |i| addends[i]);
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

        let mut actual = vec![XFieldElement::ZERO; factors.len()];
        par_running_product_into(&mut actual, initial, |i| factors[i]);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn running_affine_maps_agree_with_sequential_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] maps: Vec<Option<(XFieldElement, XFieldElement)>>,
        #[strategy(arb())] initial: XFieldElement,
    ) {
        let mut acc = initial;
        let mut expected = Vec::with_capacity(maps.len());
        for map in &maps {
            if let &Some((a, b)) = map {
                acc = a * acc + b;
            }
            expected.push(acc);
        }

        let mut actual = vec![XFieldElement::ZERO; maps.len()];
        par_running_affine_maps_into(&mut actual, initial, |i| maps[i]);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn log_derivative_agrees_with_sequential_computation(
        #[strategy(interesting_lengths())] _len: usize,
        #[strategy(vec(arb(), #_len))] fractions: Vec<[Option<(XFieldElement, XFieldElement)>; 2]>,
        #[strategy(arb())] initial: XFieldElement,
    ) {
        let fractions = fractions
            .into_iter()
            .map(|row| row.map(|f| f.filter(|(denominator, _)| !denominator.is_zero())))
            .collect_vec();

        let mut acc = initial;
        let mut expected = Vec::with_capacity(fractions.len());
        for row in &fractions {
            for (denominator, numerator) in row.iter().flatten() {
                acc += *numerator * denominator.inverse();
            }
            expected.push(acc);
        }

        let mut actual = vec![XFieldElement::ZERO; fractions.len()];
        par_log_derivative_into(&mut actual, initial, |i| fractions[i]);
        prop_assert_eq!(expected, actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn non_contiguous_column_is_written_through_buffer(
        #[strategy(vec(arb(), 1..50))] values: Vec<XFieldElement>,
    ) {
        // the columns of a row-major table are not contiguous
        let mut table = Array2::<XFieldElement>::zeros((values.len(), 2));
        with_contiguous_column(table.column_mut(1), |column| {
            column.copy_from_slice(&values)
        });
        prop_assert_eq!(Array1::from(values), table.column(1));
        prop_assert!(table.column(0).iter().all(|x| x.is_zero()));
    }
}
