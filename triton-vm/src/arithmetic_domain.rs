use std::ops::MulAssign;

use num_traits::One;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::is_power_of_two;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArithmeticDomain {
    pub offset: BFieldElement,
    pub generator: BFieldElement,
    pub length: usize,
}

impl ArithmeticDomain {
    /// Create a new domain with the given length.
    /// No offset is applied, but can added through [`with_offset()`](Self::with_offset).
    pub fn of_length(length: usize) -> Self {
        Self {
            offset: BFieldElement::one(),
            generator: Self::generator_for_length(length as u64),
            length,
        }
    }

    /// Set the offset of the domain.
    pub fn with_offset(mut self, offset: BFieldElement) -> Self {
        self.offset = offset;
        self
    }

    /// Derive a generator for a domain of the given length.
    /// The domain length must be a power of 2.
    pub fn generator_for_length(domain_length: u64) -> BFieldElement {
        assert!(
            0 == domain_length || is_power_of_two(domain_length),
            "The domain length must be a power of 2 but was {domain_length}.",
        );
        BFieldElement::primitive_root_of_unity(domain_length).unwrap()
    }

    pub fn evaluate<FF>(&self, polynomial: &Polynomial<FF>) -> Vec<FF>
    where
        FF: FiniteField + MulAssign<BFieldElement>,
    {
        polynomial.fast_coset_evaluate(self.offset, self.generator, self.length)
    }

    pub fn interpolate<FF>(&self, values: &[FF]) -> Polynomial<FF>
    where
        FF: FiniteField + MulAssign<BFieldElement>,
    {
        Polynomial::fast_coset_interpolate(self.offset, self.generator, values)
    }

    pub fn low_degree_extension<FF>(&self, codeword: &[FF], target_domain: Self) -> Vec<FF>
    where
        FF: FiniteField + MulAssign<BFieldElement>,
    {
        target_domain.evaluate(&self.interpolate(codeword))
    }

    pub fn domain_value(&self, index: u32) -> BFieldElement {
        self.generator.mod_pow_u32(index) * self.offset
    }

    pub fn domain_values(&self) -> Vec<BFieldElement> {
        let mut accumulator = BFieldElement::one();
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
}

#[cfg(test)]
mod domain_tests {
    use itertools::Itertools;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::traits::PrimitiveRootOfUnity;

    use super::*;

    use crate::shared_tests::*;
    use proptest::prelude::*;

    prop_compose! {
        fn arbitrary_domain()(
            length in (0_u64..17).prop_map(|x| 1 << x),
            offset in arbitrary_bfield_element(),
        ) -> ArithmeticDomain {
            let generator = BFieldElement::primitive_root_of_unity(length).unwrap();
            ArithmeticDomain {
                offset,
                generator,
                length: length as usize,
            }
        }
    }

    proptest! {
        #[test]
        fn evaluate_empty_polynomial(
            domain in arbitrary_domain(),
            polynomial in arbitrary_polynomial_of_degree(-1),
        ) {
            domain.evaluate(&polynomial);
        }
    }

    proptest! {
        #[test]
        fn evaluate_constant_polynomial(
            domain in arbitrary_domain(),
            polynomial in arbitrary_polynomial_of_degree(0),
        ) {
            domain.evaluate(&polynomial);
        }
    }

    proptest! {
        #[test]
        fn evaluate_linear_polynomial(
            domain in arbitrary_domain(),
            polynomial in arbitrary_polynomial_of_degree(1),
        ) {
            domain.evaluate(&polynomial);
        }
    }

    proptest! {
        #[test]
        fn evaluate_polynomial(
            domain in arbitrary_domain(),
            polynomial in arbitrary_polynomial(),
        ) {
            domain.evaluate(&polynomial);
        }
    }

    #[test]
    fn domain_values_test() {
        let x_cubed_coefficients = [0, 0, 0, 1].map(BFieldElement::new).to_vec();
        let poly = Polynomial::new(x_cubed_coefficients.clone());

        for order in [4, 8, 32] {
            let generator = BFieldElement::primitive_root_of_unity(order).unwrap();
            let offset = BFieldElement::generator();
            let b_domain = ArithmeticDomain::of_length(order as usize).with_offset(offset);

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
                    poly.evaluate(&b_domain.domain_value(i as u32)),
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

        let short_domain = ArithmeticDomain::of_length(short_domain_len);
        let long_domain = ArithmeticDomain::of_length(long_domain_len);

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
}
