pub mod mint_mont {
    use crate::algebra::*;
    use std::ops::*;

    pub trait ModSpec: Copy {
        type U: Unsigned;
        type D: Unsigned;
        const MODULUS: Self::U;
        const M_INV: Self::U;
        const R2: Self::U;
        fn to_double(u: Self::U) -> Self::D;
        fn reduce_double(value: Self::D) -> Self::U;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct MInt<M: ModSpec>(M::U);

    impl<M: ModSpec> MInt<M> {
        fn new(value: M::U) -> Self {
            MInt(value) * MInt(M::R2)
        }
    }

    macro_rules! impl_modspec {
        ($wrapper:ident $spec:ident $spec_impl:ident, U = $single:ty, D = $double:ty, EXP = $exp:expr, LOG2_EXP = $log2_exp: expr) => {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $spec<const M: $single>;

            impl<const MOD: $single> ModSpec for $spec<MOD> {
                type U = $single;
                type D = $double;
                const MODULUS: $single = MOD;
                const M_INV: $single = $spec_impl::eval_m_inv(MOD);
                const R2: $single = $spec_impl::eval_r2(MOD);

                fn to_double(u: Self::U) -> Self::D {
                    u as Self::D
                }

                fn reduce_double(x: Self::D) -> Self::U {
                    debug_assert!(x < (MOD as $double) * (MOD as $double));
                    let q = (x as $single).wrapping_mul(Self::M_INV);
                    let a = ((q as $double * Self::MODULUS as $double) >> $exp) as $single;
                    let mut res = (x >> $exp) as $single + Self::MODULUS - a;
                    if res >= Self::MODULUS {
                        res -= Self::MODULUS;
                    }
                    res
                }
            }

            mod $spec_impl {
                pub const fn eval_m_inv(m: $single) -> $single {
                    debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
                    let mut m_inv: $single = 1;
                    let two: $single = 2;

                    let mut iter = 0;
                    while iter < $log2_exp {
                        m_inv = m_inv.wrapping_mul(two.wrapping_sub(m.wrapping_mul(m_inv)));
                        iter += 1;
                    }
                    m_inv
                }

                pub const fn eval_r2(m: $single) -> $single {
                    let r = m.wrapping_neg() % m;
                    (r as $double * r as $double % m as $double) as $single
                }
            }

            pub type $wrapper<const M: $single> = MInt<$spec<M>>;
        };
    }
    impl_modspec!(M32 __ByU32 u32_impl, U = u32, D = u64, EXP = 32, LOG2_EXP = 5);
    impl_modspec!(M64 __ByU64 u64_impl, U = u64, D = u128, EXP = 64, LOG2_EXP = 6);

    impl<M: ModSpec> AddAssign<&'_ Self> for MInt<M> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<M: ModSpec> SubAssign<&'_ Self> for MInt<M> {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<M: ModSpec> MulAssign<&'_ Self> for MInt<M> {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 = M::reduce_double(M::to_double(self.0) * M::to_double(rhs.0));
        }
    }

    impl<M: ModSpec> DivAssign<&'_ Self> for MInt<M> {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    macro_rules! forward_binop {
        ($OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
            impl<M: ModSpec> $OpAssign for MInt<M> {
                fn $op_assign(&mut self, rhs: Self) {
                    self.$op_assign(&rhs);
                }
            }

            impl<M: ModSpec> $Op<&'_ Self> for MInt<M> {
                type Output = Self;
                fn $op(mut self, rhs: &Self) -> Self {
                    self.$op_assign(rhs);
                    self
                }
            }

            impl<M: ModSpec> $Op for MInt<M> {
                type Output = MInt<M>;
                fn $op(self, rhs: Self) -> Self::Output {
                    self.clone().$op(&rhs)
                }
            }
        };
    }
    forward_binop!(AddAssign add_assign, Add add);
    forward_binop!(SubAssign sub_assign, Sub sub);
    forward_binop!(MulAssign mul_assign, Mul mul);
    forward_binop!(DivAssign div_assign, Div div);

    impl<M: ModSpec> Neg for &'_ MInt<M> {
        type Output = MInt<M>;
        fn neg(self) -> MInt<M> {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            MInt(res)
        }
    }

    impl<M: ModSpec> Neg for MInt<M> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl<M: ModSpec> Default for MInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> SemiRing for MInt<M> {
        fn one() -> Self {
            Self(1.into()) * Self(M::R2)
        }
    }

    impl<M: ModSpec> CommRing for MInt<M> {}

    impl<M: ModSpec> Field for MInt<M> {
        fn inv(&self) -> Self {
            self.pow(M::MODULUS - M::U::from(2))
        }
    }

    pub trait CastImpl<U: Unsigned>: Unsigned {
        fn cast_into<M: ModSpec<U = U>>(self) -> MInt<M>;
    }

    // impl<M: ModSpec, S: CastImpl<M::U>> From<S> for MInt<M> {
    //     fn from(value: S) -> Self {
    //         value.cast_into()
    //     }
    // }

    impl<U: Unsigned> CastImpl<U> for U {
        fn cast_into<M: ModSpec<U = U>>(self) -> MInt<M> {
            MInt::new(self)
        }
    }

    macro_rules! impl_cast {
        (@common $src:ident) => {
            impl<M: ModSpec> From<$src> for MInt<M>
                where $src: CastImpl<M::U>
            {
                fn from(value: $src) -> Self {
                    value.cast_into()
                }
            }
        };

        (@eq $src:ident $u:ident) => {
            impl<M: ModSpec<U = $u>> From<MInt<M>> for $u {
                fn from(x: MInt<M>) -> Self {
                    M::reduce_double(M::to_double(x.0))
                }
            }

        };
        (@lt $src:ident $u:ident) => {
            impl CastImpl<$u> for $src {
                fn cast_into<M: ModSpec<U = $u>>(self) -> MInt<M> {
                    (self as $u).cast_into()
                }
            }
        };
        (@rt $src:ident $u:ident) => {
            impl CastImpl<$u> for $src {
                fn cast_into<M: ModSpec<U = $u>>(self) -> MInt<M> {
                    ((self % M::MODULUS as $src) as $u).cast_into()
                }
            }
        };

        (@cascade $lower:ident $($upper:ident)*) => {
            $(
                impl_cast!(@lt $lower $upper);
                impl_cast!(@rt $upper $lower);
            )*
            impl_cast!(@eq $lower $lower);
        };
        ($lower:ident $($rest:ident)*) => {
            impl_cast!(@common $lower);
            impl_cast!(@cascade $lower $($rest)*);
            impl_cast!($($rest)*);
        };
        () => {};
    }
    impl_cast!(u8 u16 u32 u64 u128);

    impl<M: ModSpec> std::fmt::Debug for MInt<M>
    where
        M::U: std::fmt::Debug + From<MInt<M>>,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            M::U::from(*self).fmt(f)
        }
    }

    impl<M: ModSpec> std::fmt::Display for MInt<M>
    where
        M::U: std::fmt::Display + From<MInt<M>>,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            M::U::from(*self).fmt(f)
        }
    }

    impl<M: ModSpec> std::str::FromStr for MInt<M>
    where
        M::U: std::str::FromStr,
    {
        type Err = <M::U as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse::<M::U>().map(|x| MInt::new(x))
        }
    }
}
