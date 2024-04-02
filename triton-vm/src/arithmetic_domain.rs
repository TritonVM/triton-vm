use itertools::Itertools;
use std::ops::Mul;
use std::ops::MulAssign;

use num_traits::One;
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
    /// No offset is applied, but can be added through [`with_offset()`](Self::with_offset).
    ///
    /// # Errors
    ///
    /// Errors if the domain length is not a power of 2.
    pub fn of_length(length: usize) -> Result<Self> {
        let domain = Self {
            offset: bfe!(1),
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
            + From<BFieldElement>,
    {
        // The limitation arises in `Polynomial::fast_coset_evaluate` in dependency `twenty-first`.
        let batch_evaluation_is_possible = self.length >= polynomial.coefficients.len();
        if batch_evaluation_is_possible {
            let (offset, generator, length) = (self.offset, self.generator, self.length);
            polynomial.fast_coset_evaluate::<BFieldElement>(offset, generator, length)
        } else {
            let domain_values = self.domain_values().into_iter();
            let domain_values = domain_values.map(FF::from).collect_vec();
            polynomial.batch_evaluate(&domain_values)
        }
    }

    pub fn interpolate<FF>(&self, values: &[FF]) -> Polynomial<FF>
    where
        FF: FiniteField + MulAssign<BFieldElement> + Mul<BFieldElement, Output = FF>,
    {
        // generic type made explicit to avoid performance regressions due to auto-conversion
        Polynomial::fast_coset_interpolate::<BFieldElement>(self.offset, self.generator, values)
    }

    pub fn low_degree_extension<FF>(&self, codeword: &[FF], target_domain: Self) -> Vec<FF>
    where
        FF: FiniteField
            + MulAssign<BFieldElement>
            + Mul<BFieldElement, Output = FF>
            + From<BFieldElement>,
    {
        target_domain.evaluate(&self.interpolate(codeword))
    }

    /// Compute the nth element of the domain.
    pub fn domain_value(&self, index: u32) -> BFieldElement {
        self.generator.mod_pow_u32(index) * self.offset
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
mod tests {
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::shared_tests::*;

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
        #[strategy(arbitrary_polynomial_of_degree(-1))] polynomial: Polynomial<XFieldElement>,
    ) {
        domain.evaluate(&polynomial);
    }

    #[proptest]
    fn evaluate_constant_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial_of_degree(0))] polynomial: Polynomial<XFieldElement>,
    ) {
        domain.evaluate(&polynomial);
    }

    #[proptest]
    fn evaluate_linear_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial_of_degree(1))] polynomial: Polynomial<XFieldElement>,
    ) {
        domain.evaluate(&polynomial);
    }

    #[proptest]
    fn evaluate_polynomial(
        #[strategy(arbitrary_domain())] domain: ArithmeticDomain,
        #[strategy(arbitrary_polynomial())] polynomial: Polynomial<XFieldElement>,
    ) {
        domain.evaluate(&polynomial);
    }

    #[test]
    fn domain_values() {
        let x_cubed_coefficients = [0, 0, 0, 1].map(BFieldElement::new).to_vec();
        let poly = Polynomial::new(x_cubed_coefficients.clone());

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
            assert_eq!(
                expected_b_values, actual_b_values_1,
                "domain_values() generates the arithmetic domain's BFieldElement values"
            );
            assert_eq!(
                expected_b_values, actual_b_values_2,
                "domain_value() generates the given domain BFieldElement value"
            );

            let values = b_domain.evaluate(&poly);
            assert_ne!(values, x_cubed_coefficients);

            let interpolant = b_domain.interpolate(&values);
            assert_eq!(poly, interpolant);

            // Verify that batch-evaluated values match a manual evaluation
            for i in 0..order {
                assert_eq!(
                    poly.evaluate(b_domain.domain_value(i as u32)),
                    values[i as usize]
                );
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

        let polynomial = Polynomial::new([1, 2, 3, 4].map(BFieldElement::new).to_vec());
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
}
