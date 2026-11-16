//! Zero-copy views of B- and X-field slices as raw `u64`s.

use twenty_first::prelude::*;

mod sealed {
    pub trait Sealed {}
    impl Sealed for twenty_first::prelude::BFieldElement {}
    impl Sealed for twenty_first::prelude::XFieldElement {}
}

/// A field whose elements are `NCOMP` raw Montgomery `u64`s:
/// [`BFieldElement`] (1) and [`XFieldElement`] (3). The GPU works on these
/// raw words directly, so tables are uploaded without any conversion.
pub trait GpuField: sealed::Sealed + Copy {
    const NCOMP: u32;
    fn as_u64s(s: &[Self]) -> &[u64];
    fn as_u64s_mut(s: &mut [Self]) -> &mut [u64];
    fn from_u64s(v: Vec<u64>) -> Vec<Self>;
}

macro_rules! impl_gpu_field {
    ($t:ty, $n:expr) => {
        impl GpuField for $t {
            const NCOMP: u32 = $n;
            fn as_u64s(s: &[Self]) -> &[u64] {
                // SAFETY: the type is #[repr(transparent)] over $n u64s (raw Montgomery form).
                unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u64, s.len() * $n) }
            }
            fn as_u64s_mut(s: &mut [Self]) -> &mut [u64] {
                // SAFETY: as above; the kernels only produce canonical elements.
                unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u64, s.len() * $n) }
            }
            fn from_u64s(v: Vec<u64>) -> Vec<Self> {
                v.chunks_exact($n)
                    .map(|c| {
                        // SAFETY: same layout, see above.
                        unsafe { std::ptr::read_unaligned(c.as_ptr() as *const $t) }
                    })
                    .collect()
            }
        }
    };
}

impl_gpu_field!(BFieldElement, 1);
impl_gpu_field!(XFieldElement, 3);

const _: () = assert!(std::mem::size_of::<BFieldElement>() == 8);
const _: () = assert!(std::mem::size_of::<XFieldElement>() == 24);
