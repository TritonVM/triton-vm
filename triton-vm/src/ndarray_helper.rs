use itertools::Itertools;
use ndarray::s;
use ndarray::ArrayViewMut2;
use ndarray::Ix;
use std::fmt::Debug;

/// Slice a two-dimensional array into many non-overlapping mutable subviews
/// of the same height as the array, based on the contiguous partition induced
/// by a set of column indices. The supplied indices indicate the first column
/// of each new slice; and the last slice ends with the array's last column.
///
/// # Panics
/// Panics if column indices are not sorted or out of bounds.
pub fn horizontal_multi_slice_mut<'a, T: 'a + Debug, const N: usize>(
    array: ArrayViewMut2<'a, T>,
    column_indices: [Ix; N],
) -> [ArrayViewMut2<'a, T>; N] {
    let mut returnable_slices = vec![];

    let mut stop_index = array.ncols();
    let mut remainder = array;
    for start_index in column_indices.into_iter().rev() {
        let (new_remainder, slice) =
            remainder.multi_slice_move((s![.., ..start_index], s![.., start_index..stop_index]));
        returnable_slices.push(slice);
        remainder = new_remainder;
        stop_index = start_index;
    }

    returnable_slices.reverse();
    returnable_slices.try_into().unwrap()
}

/// Computes the list of partial sums, beginning with zero and excluding the
/// total.
pub fn partial_sums<const N: usize>(indices: [usize; N]) -> [usize; N] {
    indices
        .into_iter()
        .scan(0, |acc, index| {
            let yld = *acc;
            *acc += index;
            Some(yld)
        })
        .collect_vec()
        .try_into()
        .unwrap()
}

#[cfg(test)]
mod test {
    use ndarray::concatenate;
    use ndarray::Array2;
    use ndarray::Axis;

    use proptest::collection::vec;
    use proptest::prelude::BoxedStrategy;
    use proptest::prop_assert_eq;
    use proptest::strategy::Strategy;
    use test_strategy::proptest;

    use super::*;

    #[test]
    fn horizontal_multi_slice_works_as_expected() {
        let m = 2;
        let n = 6;
        let mut array = Array2::<usize>::zeros((m, n));

        let [mut a, mut b, mut c] = horizontal_multi_slice_mut(array.view_mut(), [0, 1, 3]);

        a.mapv_inplace(|_| 1);
        b.mapv_inplace(|_| 2);
        c.mapv_inplace(|_| 3);

        assert_eq!(
            Array2::from_shape_vec(
                (m, n),
                [vec![1, 2, 2, 3, 3, 3], vec![1, 2, 2, 3, 3, 3]].concat()
            )
            .unwrap(),
            array
        );
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
        let width = widths.iter().cloned().sum::<usize>();
        let mut array = Array2::zeros((height, width));
        let mutable_slices = horizontal_multi_slice_mut(array.view_mut(), partial_sums(widths));
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
