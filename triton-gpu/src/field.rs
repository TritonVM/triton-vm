//! Zero-copy views of B- and X-field slices as raw `u64`s.

use num_traits::ConstOne;
use num_traits::ConstZero;
use twenty_first::prelude::*;

mod sealed {
    pub trait Sealed {}
    impl Sealed for twenty_first::prelude::BFieldElement {}
    impl Sealed for twenty_first::prelude::XFieldElement {}
}

/// A field whose elements are `NCOMP` raw Montgomery `u64`s:
/// [`BFieldElement`] (1) and [`XFieldElement`] (3). The GPU works on these
/// raw words directly, so tables are uploaded without any conversion.
pub trait GpuField: sealed::Sealed + Copy + Send + Sync {
    const NCOMP: u32;
    fn as_u64s(s: &[Self]) -> &[u64];
    fn as_u64s_mut(s: &mut [Self]) -> &mut [u64];
    fn from_u64s(v: Vec<u64>) -> Vec<Self>;

    /// The coset low-degree extension of one column on the CPU, exactly as
    /// triton-vm's `evaluate_randomized_interpolant_into`: `out[i] =
    /// (interpolant[i] + zerofier · randomizer[i]) · offset^i`, zero padded,
    /// then an NTT. Used by the hybrid CPU + GPU pipeline.
    fn cpu_coset_lde(
        interpolant: &[Self],
        randomizer: &[Self],
        zerofier: BFieldElement,
        offset: BFieldElement,
        out: &mut [Self],
    );
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
            fn cpu_coset_lde(
                interpolant: &[Self],
                randomizer: &[Self],
                zerofier: BFieldElement,
                offset: BFieldElement,
                out: &mut [Self],
            ) {
                let (scaled, pad) = out.split_at_mut(interpolant.len());
                let mut power = BFieldElement::ONE;
                for (cell, &c) in scaled.iter_mut().zip(interpolant) {
                    *cell = c * power;
                    power *= offset;
                }
                pad.fill(<$t>::from(BFieldElement::ZERO));
                let mut power = zerofier;
                for (cell, &c) in out.iter_mut().zip(randomizer) {
                    *cell += c * power;
                    power *= offset;
                }
                twenty_first::math::ntt::ntt(out);
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
