use itertools::Itertools;
use ndarray::s;
use ndarray::ArrayBase;
use ndarray::ArrayViewMut2;
use ndarray::DataMut;
use ndarray::Dim;
use ndarray::Ix;
use ndarray::RawData;
use std::fmt::Debug;

/// Slice a two-dimensional array into many non-overlapping mutable subviews
/// of the same height as the array, based on the contiguous partition induced
/// by a set of column indices. The supplied indices indicate the first column
/// of each new slice; and the last slice ends with the array's last column.
///
/// # Panics
/// Panics if column indices are not sorted or out of bounds.
pub fn horizontal_multi_slice_mut<'a, T: 'a + RawData + DataMut, const N: usize>(
    array: &'a mut ArrayBase<T, Dim<[Ix; 2]>>,
    column_indices: [Ix; N],
) -> [ArrayViewMut2<'a, <T as RawData>::Elem>; N]
where
    <T as ndarray::RawData>::Elem: Debug,
{
    let mut returnable_slices = vec![];

    let mut stop_index = array.ncols();
    let mut remainder = array.view_mut();
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
    use ndarray::Array2;

    use super::horizontal_multi_slice_mut;

    #[test]
    fn horizontal_multi_slice_works_as_expected() {
        let m = 2;
        let n = 6;
        let mut array = Array2::<usize>::zeros((m, n));

        let [mut a, mut b, mut c] = horizontal_multi_slice_mut(&mut array, [0, 1, 3]);

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
}
