use std::marker::PhantomData;
use std::ops::MulAssign;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::{FiniteField, ModPowU32};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FriDomain<FF> {
    pub offset: BFieldElement,
    pub omega: BFieldElement,
    pub length: usize,
    _field: PhantomData<FF>,
}

impl<FF> FriDomain<FF>
where
    FF: FiniteField + From<BFieldElement> + MulAssign<BFieldElement>,
{
    pub fn new(offset: BFieldElement, omega: BFieldElement, length: usize) -> Self {
        let _field = PhantomData;
        Self {
            offset,
            omega,
            length,
            _field,
        }
    }

    pub fn evaluate(&self, polynomial: &Polynomial<FF>) -> Vec<FF> {
        polynomial.fast_coset_evaluate(&self.offset, self.omega, self.length)
    }

    pub fn interpolate(&self, values: &[FF]) -> Polynomial<FF> {
        Polynomial::<FF>::fast_coset_interpolate(&self.offset, self.omega, values)
    }

    pub fn domain_value(&self, index: u32) -> FF {
        let domain_value: BFieldElement = self.omega.mod_pow_u32(index) * self.offset;
        domain_value.into()
    }

    pub fn domain_values(&self) -> Vec<FF> {
        let mut res = Vec::with_capacity(self.length);
        let mut acc = FF::one();

        for _ in 0..self.length {
            let domain_value = {
                let mut tmp = acc;
                tmp *= self.offset;
                tmp
            };
            res.push(domain_value);
            acc *= self.omega;
        }

        res
    }
}

#[cfg(test)]
mod fri_domain_tests {
    use super::*;
    use itertools::Itertools;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::traits::PrimitiveRootOfUnity;
    use twenty_first::shared_math::x_field_element::XFieldElement;

    #[test]
    fn domain_values_test() {
        // f(x) = x^3
        let x_squared_coefficients = vec![0u64.into(), 0u64.into(), 0u64.into(), 1u64.into()];
        let poly = Polynomial::<BFieldElement>::new(x_squared_coefficients.clone());

        for order in [4, 8, 32] {
            let omega = BFieldElement::primitive_root_of_unity(order).unwrap();
            let offset = BFieldElement::generator();
            let b_domain = FriDomain::<BFieldElement>::new(offset, omega, order as usize);
            let x_domain = FriDomain::<XFieldElement>::new(offset, omega, order as usize);

            let expected_b_values: Vec<BFieldElement> =
                (0..order).map(|i| offset * omega.mod_pow(i)).collect();
            let actual_b_values_1 = b_domain.domain_values();
            let actual_b_values_2 = (0..order as u32)
                .map(|i| b_domain.domain_value(i))
                .collect_vec();
            assert_eq!(
                expected_b_values, actual_b_values_1,
                "domain_values() generates the FRI domain BFieldElement values"
            );
            assert_eq!(
                expected_b_values, actual_b_values_2,
                "domain_value() generates the given FRI domain BFieldElement value"
            );

            let expected_x_values: Vec<XFieldElement> =
                expected_b_values.iter().map(|bfe| bfe.lift()).collect();
            let actual_x_values_1 = x_domain.domain_values();
            let actual_x_values_2 = (0..order as u32)
                .map(|i| x_domain.domain_value(i))
                .collect_vec();
            assert_eq!(
                expected_x_values, actual_x_values_1,
                "domain_values() generates the FRI domain XFieldElement values"
            );
            assert_eq!(
                expected_x_values, actual_x_values_2,
                "domain_value() generates the given FRI domain XFieldElement values"
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

            let x_squared_coefficients_lifted: Vec<XFieldElement> = x_squared_coefficients
                .clone()
                .into_iter()
                .map(|x| x.lift())
                .collect();
            let xpol = Polynomial::new(x_squared_coefficients_lifted.clone());

            let x_field_x_values = x_domain.evaluate(&xpol);
            assert_ne!(x_field_x_values, x_squared_coefficients_lifted);

            let x_interpolant = x_domain.interpolate(&x_field_x_values);
            assert_eq!(xpol, x_interpolant);
        }
    }
}
