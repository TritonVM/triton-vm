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
    /// Create a new domain with the given length and offset.
    pub fn of_length_with_offset(length: usize, offset: BFieldElement) -> Self {
        Self {
            offset,
            generator: Self::generator_for_length(length as u64),
            length,
        }
    }

    /// Create a new domain with the given length. No offset is applied.
    pub fn of_length(length: usize) -> Self {
        Self {
            offset: BFieldElement::one(),
            generator: Self::generator_for_length(length as u64),
            length,
        }
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
        polynomial.fast_coset_evaluate(&self.offset, self.generator, self.length)
    }

    pub fn interpolate<FF>(&self, values: &[FF]) -> Polynomial<FF>
    where
        FF: FiniteField + MulAssign<BFieldElement>,
    {
        Polynomial::fast_coset_interpolate(&self.offset, self.generator, values)
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

    #[test]
    fn domain_values_test() {
        // f(x) = x^3
        let x_squared_coefficients = vec![0u64.into(), 0u64.into(), 0u64.into(), 1u64.into()];
        let poly = Polynomial::<BFieldElement>::new(x_squared_coefficients.clone());

        for order in [4, 8, 32] {
            let generator = BFieldElement::primitive_root_of_unity(order).unwrap();
            let offset = BFieldElement::generator();
            let b_domain = ArithmeticDomain::of_length_with_offset(order as usize, offset);

            let expected_b_values: Vec<BFieldElement> =
                (0..order).map(|i| offset * generator.mod_pow(i)).collect();
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
            assert_ne!(values, x_squared_coefficients);

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
