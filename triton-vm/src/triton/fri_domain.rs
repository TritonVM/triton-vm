use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::PrimeField;
use crate::shared_math::x_field_element::XFieldElement;

#[derive(Debug, Clone)]
pub struct FriDomain<PF>
where
    PF: PrimeField,
{
    pub offset: PF,
    pub omega: PF,
    pub length: usize,
}

impl<PF> FriDomain<PF>
where
    PF: PrimeField,
{
    pub fn evaluate(&self, polynomial: &Polynomial<PF>) -> Vec<PF> {
        polynomial.fast_coset_evaluate(&self.offset, self.omega, self.length)
    }

    pub fn interpolate(&self, values: &[PF]) -> Polynomial<PF> {
        Polynomial::<PF>::fast_coset_interpolate(&self.offset, self.omega, values)
    }

    pub fn domain_value(&self, index: u32) -> PF {
        self.omega.mod_pow_u32(index) * self.offset
    }

    pub fn domain_values(&self) -> Vec<PF> {
        let mut res = Vec::with_capacity(self.length);
        let mut acc = self.omega.ring_one();

        for _ in 0..self.length {
            res.push(acc * self.offset);
            acc *= self.omega;
        }

        res
    }
}

pub fn lift_domain(domain: &FriDomain<BFieldElement>) -> FriDomain<XFieldElement> {
    FriDomain {
        offset: domain.offset.lift(),
        omega: domain.omega.lift(),
        length: domain.length,
    }
}

#[cfg(test)]
mod fri_domain_tests {
    use super::*;
    use crate::shared_math::{
        b_field_element::BFieldElement, stark::triton::fri_domain::lift_domain,
        traits::GetPrimitiveRootOfUnity, x_field_element::XFieldElement,
    };

    #[test]
    fn x_values_test() {
        // f(x) = x^3
        let x_squared_coefficients = vec![0.into(), 0.into(), 0.into(), 1.into()];
        let poly = Polynomial::<BFieldElement>::new(x_squared_coefficients.clone());

        for order in [4, 8, 32] {
            let omega = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            let offset = BFieldElement::generator();
            let b_domain = FriDomain {
                offset,
                omega,
                length: order as usize,
            };

            let expected_x_values: Vec<BFieldElement> =
                (0..order).map(|i| offset * omega.mod_pow(i)).collect();

            let actual_x_values = b_domain.domain_values();
            assert_eq!(expected_x_values.len(), actual_x_values.len());
            assert_eq!(expected_x_values, actual_x_values);

            // Verify that `x_value` also returns expected values
            for i in 0..order {
                assert_eq!(
                    expected_x_values[i as usize],
                    b_domain.domain_value(i as u32)
                );
            }

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
            let x_domain = lift_domain(&b_domain);
            let xpol = Polynomial::new(x_squared_coefficients_lifted.clone());

            let x_field_x_values = x_domain.evaluate(&xpol);
            assert_ne!(x_field_x_values, x_squared_coefficients_lifted);

            let x_interpolant = x_domain.interpolate(&x_field_x_values);
            assert_eq!(xpol, x_interpolant);
        }
    }
}
