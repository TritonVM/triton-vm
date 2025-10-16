use std::ops::Mul;
use std::ops::MulAssign;

use num_traits::ConstOne;
use num_traits::One;
use num_traits::Zero;
use rayon::prelude::*;
use twenty_first::math::traits::FiniteField;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::prelude::*;

use crate::error::ArithmeticDomainError;

type Result<T> = std::result::Result<T, ArithmeticDomainError>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ArithmeticDomain {
    pub offset: BFieldElement,
    pub generator: BFieldElement,
    pub length: usize,
}

impl ArithmeticDomain {
    /// Create a new domain with the given length.
    /// No offset is applied, but can be added through
    /// [`with_offset()`](Self::with_offset).
    ///
    /// # Errors
    ///
    /// Errors if the domain length is not a power of 2.
    pub fn of_length(length: usize) -> Result<Self> {
        let domain = Self {
            offset: BFieldElement::ONE,
            generator: Self::generator_for_length(length as u64)?,
            length,
        };
        Ok(domain)
    }

    /// Set the offset of the domain.
    #[must_use]
    pub fn with_offset(mut self, offset: BFieldElement) -> Self {
        self.offset = offset;
        self
    }

    /// Derive a generator for a domain of the given length.
    ///
    /// # Errors
    ///
    /// Errors if the domain length is not a power of 2.
    pub fn generator_for_length(domain_length: u64) -> Result<BFieldElement> {
        let error = ArithmeticDomainError::PrimitiveRootNotSupported(domain_length);
        BFieldElement::primitive_root_of_unity(domain_length).ok_or(error)
    }

    pub fn evaluate<FF>(&self, polynomial: &Polynomial<FF>) -> Vec<FF>
    where
        FF: FiniteField
            + MulAssign<BFieldElement>
            + Mul<BFieldElement, Output = FF>
            + From<BFieldElement>
            + 'static,
    {
        let (offset, length) = (self.offset, self.length);
        let evaluate_from = |chunk| Polynomial::from(chunk).fast_coset_evaluate(offset, length);

        // avoid `enumerate` to directly get index of the right type
        let mut indexed_chunks = (0..).zip(polynomial.coefficients().chunks(length));

        // only allocate a bunch of zeros if there are no chunks
        let mut values = indexed_chunks.next().map_or_else(
            || vec![FF::ZERO; length],
            |(_, first_chunk)| evaluate_from(first_chunk),
        );
        for (chunk_index, chunk) in indexed_chunks {
            let coefficient_index = chunk_index * u64::try_from(length).unwrap();
            let scaled_offset = offset.mod_pow(coefficient_index);
            values
                .par_iter_mut()
                .zip(evaluate_from(chunk))
                .for_each(|(value, evaluation)| *value += evaluation * scaled_offset);
        }

        values
    }

    /// # Panics
    ///
    /// Panics if the length of the argument does not match the length of
    /// `self`.
    pub fn interpolate<FF>(&self, values: &[FF]) -> Polynomial<'static, FF>
    where
        FF: FiniteField + MulAssign<BFieldElement> + Mul<BFieldElement, Output = FF>,
    {
        debug_assert_eq!(self.length, values.len()); // required by `fast_coset_interpolate`

        Polynomial::fast_coset_interpolate(self.offset, values)
    }

    pub fn low_degree_extension<FF>(&self, codeword: &[FF], target_domain: Self) -> Vec<FF>
    where
        FF: FiniteField
            + MulAssign<BFieldElement>
            + Mul<BFieldElement, Output = FF>
            + From<BFieldElement>
            + 'static,
    {
        target_domain.evaluate(&self.interpolate(codeword))
    }

    /// Compute the `n`th element of the domain.
    pub fn domain_value(&self, n: u32) -> BFieldElement {
        self.generator.mod_pow_u32(n) * self.offset
    }

    pub fn domain_values(&self) -> Vec<BFieldElement> {
        let mut accumulator = bfe!(1);
        let mut domain_values = Vec::with_capacity(self.length);

        for _ in 0..self.length {
            domain_values.push(accumulator * self.offset);
            accumulator *= self.generator;
        }
        assert!(
            accumulator.is_one(),
            "length must be the order of the generator"
        );
        domain_values
    }

    /// A polynomial that evaluates to 0 on (and only on)
    /// a [domain value][Self::domain_values].
    pub fn zerofier(&self) -> Polynomial<'_, BFieldElement> {
        if self.offset.is_zero() {
            return Polynomial::x_to_the(1);
        }

        Polynomial::x_to_the(self.length)
            - Polynomial::from_constant(self.offset.mod_pow(self.length as u64))
    }

    /// [`Self::zerofier`] times the argument.
    /// More performant than polynomial multiplication.
    /// See [`Self::zerofier`] for details.
    pub fn mul_zerofier_with<FF>(&self, polynomial: Polynomial<FF>) -> Polynomial<'static, FF>
    where
        FF: FiniteField + Mul<BFieldElement, Output = FF>,
    {
        // use knowledge of zerofier's shape for faster multiplication
        polynomial.clone().shift_coefficients(self.length)
            - polynomial.scalar_mul(self.offset.mod_pow(self.length as u64))
    }

    pub(crate) fn halve(&self) -> Result<Self> {
        if self.length < 2 {
            return Err(ArithmeticDomainError::TooSmallForHalving(self.length));
        }
        let domain = Self {
            offset: self.offset.square(),
            generator: self.generator.square(),
            length: self.length / 2,
        };
        Ok(domain)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::shared_tests::arbitrary_polynomial;
    use crate::shared_tests::arbitrary_polynomial_of_degree;

    use super::*;

    prop_compose! {
        fn arbitrary_domain()(
            length in (0_usize..17).prop_map(|x| 1 << x),
        )(
            domain in arbitrary_domain_of_length(length),
        ) -> ArithmeticDomain {
            domain
        }
    }

    prop_compose! {
        fn arbitrary_halveable_domain()(
            length in (2_usize..17).prop_map(|x| 1 << x),
        )(
            domain in arbitrary_domain_of_length(length),
        ) -> ArithmeticDomain {
            domain
        }
    }

    prop_compose! {
        fn arbitrary_domain_of_length(length: usize)(
            offset in arb(),
        ) -> ArithmeticDomain {
            ArithmeticDomain::of_length(length).unwrap().with_offset(offset)
        }
    }

    #[proptest]
    fn evaluate_empty_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial_of_degree(-1))] poly: Polynomial<'static, XFieldElement>,
    ) {
        domain.evaluate(&poly);
    }

    #[proptest]
    fn evaluate_constant_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial_of_degree(0))] poly: Polynomial<'static, XFieldElement>,
    ) {
        domain.evaluate(&poly);
    }

    #[proptest]
    fn evaluate_linear_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial_of_degree(1))] poly: Polynomial<'static, XFieldElement>,
    ) {
        domain.evaluate(&poly);
    }

    #[proptest]
    fn evaluate_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial())] polynomial: Polynomial<'static, XFieldElement>,
    ) {
        domain.evaluate(&polynomial);
    }

    #[test]
    fn domain_values() {
        let poly = Polynomial::<BFieldElement>::x_to_the(3);
        let x_cubed_coefficients = poly.clone().into_coefficients();

        for order in [4, 8, 32] {
            let generator = BFieldElement::primitive_root_of_unity(order).unwrap();
            let offset = BFieldElement::generator();
            let b_domain = ArithmeticDomain::of_length(order as usize)
                .unwrap()
                .with_offset(offset);

            let expected_b_values = (0..order)
                .map(|i| offset * generator.mod_pow(i))
                .collect_vec();
            let actual_b_values_1 = b_domain.domain_values();
            let actual_b_values_2 = (0..order as u32)
                .map(|i| b_domain.domain_value(i))
                .collect_vec();
            assert_eq!(expected_b_values, actual_b_values_1);
            assert_eq!(expected_b_values, actual_b_values_2);

            let values = b_domain.evaluate(&poly);
            assert_ne!(values, x_cubed_coefficients);

            let interpolant = b_domain.interpolate(&values);
            assert_eq!(poly, interpolant);

            // Verify that batch-evaluated values match a manual evaluation
            for i in 0..order {
                let indeterminate = b_domain.domain_value(i as u32);
                let evaluation: BFieldElement = poly.evaluate(indeterminate);
                assert_eq!(evaluation, values[i as usize]);
            }
        }
    }

    #[test]
    fn low_degree_extension() {
        let short_domain_len = 32;
        let long_domain_len = 128;
        let unit_distance = long_domain_len / short_domain_len;

        let short_domain = ArithmeticDomain::of_length(short_domain_len).unwrap();
        let long_domain = ArithmeticDomain::of_length(long_domain_len).unwrap();

        let polynomial = Polynomial::new(bfe_vec![1, 2, 3, 4]);
        let short_codeword = short_domain.evaluate(&polynomial);
        let long_codeword = short_domain.low_degree_extension(&short_codeword, long_domain);

        assert_eq!(long_codeword.len(), long_domain_len);

        let long_codeword_sub_view = long_codeword
            .into_iter()
            .step_by(unit_distance)
            .collect_vec();
        assert_eq!(short_codeword, long_codeword_sub_view);
    }

    #[proptest]
    fn halving_domain_squares_all_points(
        #[strategy(arbitrary_halveable_domain())] domain: ArithmeticDomain,
    ) {
        let half_domain = domain.halve()?;
        prop_assert_eq!(domain.length / 2, half_domain.length);

        let domain_points = domain.domain_values();
        let half_domain_points = half_domain.domain_values();

        for (domain_point, halved_domain_point) in domain_points
            .into_iter()
            .zip(half_domain_points.into_iter())
        {
            prop_assert_eq!(domain_point.square(), halved_domain_point);
        }
    }

    #[test]
    fn too_small_domains_cannot_be_halved() {
        for i in [0, 1] {
            let domain = ArithmeticDomain::of_length(i).unwrap();
            let_assert!(Err(err) = domain.halve());
            assert!(ArithmeticDomainError::TooSmallForHalving(i) == err);
        }
    }

    #[proptest]
    fn can_evaluate_polynomial_larger_than_domain(
        #[strategy(1_usize..10)] _log_domain_length: usize,
        #[strategy(1_usize..5)] _expansion_factor: usize,
        #[strategy(Just(1 << #_log_domain_length))] domain_length: usize,
        #[strategy(vec(arb(),#domain_length*#_expansion_factor))] coefficients: Vec<BFieldElement>,
        #[strategy(arb())] offset: BFieldElement,
    ) {
        let domain = ArithmeticDomain::of_length(domain_length)
            .unwrap()
            .with_offset(offset);
        let polynomial = Polynomial::new(coefficients);

        let values0 = domain.evaluate(&polynomial);
        let values1 = polynomial.batch_evaluate(&domain.domain_values());
        assert_eq!(values0, values1);
    }

    #[proptest]
    fn zerofier_is_actually_zerofier(#[strategy(arbitrary_domain())] domain: ArithmeticDomain) {
        let actual_zerofier = Polynomial::zerofier(&domain.domain_values());
        prop_assert_eq!(actual_zerofier, domain.zerofier());
    }

    #[proptest]
    fn multiplication_with_zerofier_is_identical_to_method_mul_with_zerofier(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial())] polynomial: Polynomial<'static, XFieldElement>,
    ) {
        let mul = domain.zerofier() * polynomial.clone();
        let mul_with = domain.mul_zerofier_with(polynomial);
        prop_assert_eq!(mul, mul_with);
    }
}
