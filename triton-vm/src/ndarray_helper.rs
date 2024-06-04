use std::fmt::Debug;

use ndarray::s;
use ndarray::ArrayViewMut2;
use ndarray::Ix;

/// Slice a two-dimensional array into many non-overlapping mutable subviews
/// of the same height as the array, based on the contiguous partition induced
/// by a set of column indices. Except for the last, the supplied indices
/// indicate the first column of each new slice. The last index indicates the
/// end point of the last slice. So the number of slices returned is one less
/// than the number of indices supplied.
///
/// # Panics
/// Panics if column indices are not sorted or out of bounds.
pub fn horizontal_multi_slice_mut<'a, T: 'a + Debug>(
    array: ArrayViewMut2<'a, T>,
    column_indices: &[Ix],
) -> Vec<ArrayViewMut2<'a, T>> {
    let mut returnable_slices = vec![];

    let mut stop_index = array.ncols();
    let mut remainder = array;
    for &start_index in column_indices.iter().rev() {
        let (new_remainder, slice) =
            remainder.multi_slice_move((s![.., ..start_index], s![.., start_index..stop_index]));
        returnable_slices.push(slice);
        remainder = new_remainder;
        stop_index = start_index;
    }

    returnable_slices.reverse();
    returnable_slices.pop();
    returnable_slices
}

/// Computes the list of partial sums, beginning with zero and including the
/// total.
pub fn partial_sums(summands: &[usize]) -> Vec<usize> {
    let mut sums = vec![0];
    for &summand in summands {
        sums.push(sums.last().copied().unwrap() + summand);
    }
    sums
}

/// Given a list of neighboring columns, represented as their sorted indices,
/// return a list of indices whose overlapping windows of width 2 denote the
/// start and end point of every column. In practice, this means "append last+1".
pub fn contiguous_column_slices(column_indices: &[usize]) -> Vec<usize> {
    [
        column_indices.to_vec(),
        vec![*column_indices.last().unwrap() + 1],
    ]
    .concat()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use ndarray::concatenate;
    use ndarray::Array2;
    use ndarray::Axis;
    use proptest::collection::vec;
    use proptest::prelude::BoxedStrategy;
    use proptest::prop_assert_eq;
    use proptest::strategy::Strategy;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn contiguous_column_slices_appends_last_plus_one(
        #[strategy(0usize..100)] start: usize,
        #[strategy(#start+1..=#start+100)] stop: usize,
    ) {
        let columns = (start..=stop).collect_vec();
        let columns_slice = [columns.clone(), vec![*columns.last().unwrap() + 1]].concat();
        prop_assert_eq!(columns_slice, contiguous_column_slices(&columns));
    }

    #[test]
    fn can_start_at_non_zero_index_and_stop_before_end() {
        let dimensions = (2, 6);
        let mut array = Array2::<usize>::zeros(dimensions);

        let [mut a] = horizontal_multi_slice_mut(array.view_mut(), &[2, 3])
            .try_into()
            .unwrap();

        a.mapv_inplace(|_| 2);

        assert_eq!(array![[0, 0, 2, 0, 0, 0], [0, 0, 2, 0, 0, 0]], array);
    }

    #[test]
    fn horizontal_multi_slice_works_as_expected() {
        let m = 2;
        let n = 6;
        let mut array = Array2::<usize>::zeros((m, n));

        let [mut a, mut b] = horizontal_multi_slice_mut(array.view_mut(), &[0, 1, 3])
            .try_into()
            .unwrap();

        a.mapv_inplace(|_| 1);
        b.mapv_inplace(|_| 2);

        assert_eq!(array![[1, 2, 2, 0, 0, 0], [1, 2, 2, 0, 0, 0]], array);
    }

    #[test]
    fn repeated_index_gives_empty_slice() {
        let m = 2;
        let n = 6;
        let mut array = Array2::<usize>::zeros((m, n));

        let [mut a, mut b] = horizontal_multi_slice_mut(array.view_mut(), &[0, 1, 1])
            .try_into()
            .unwrap();

        a.mapv_inplace(|_| 1);
        b.mapv_inplace(|_| 2);

        assert_eq!(0, b.ncols());
        assert_eq!(array![[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], array);
    }

    fn strategy_of_widths() -> BoxedStrategy<[usize; 10]> {
        vec(0usize..10, 10)
            .prop_map(|v| v.try_into().unwrap())
            .boxed()
    }

    #[proptest]
    fn horizontal_slice_with_partial_sums(
        #[strategy(strategy_of_widths())] widths: [usize; 10],
        #[strategy(0usize..10)] height: usize,
    ) {
        let width = widths.iter().copied().sum::<usize>();
        let mut array = Array2::zeros((height, width));
        let mutable_slices: [_; 10] =
            horizontal_multi_slice_mut(array.view_mut(), &partial_sums(&widths))
                .try_into()
                .unwrap();
        for (i, mut slice) in mutable_slices.into_iter().enumerate() {
            slice.mapv_inplace(|_| i as u32);
        }

        let expected_slices = widths
            .iter()
            .enumerate()
            .map(|(i, &w)| Array2::from_shape_vec((height, w), vec![i as u32; w * height]).unwrap())
            .collect_vec();
        let expected_views = expected_slices
            .iter()
            .map(|slice| slice.view())
            .collect_vec();
        let expected_array = concatenate(Axis(1), &expected_views).unwrap();
        prop_assert_eq!(expected_array, array);
    }
}
