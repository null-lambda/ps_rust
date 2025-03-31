pub mod mint_dyn {
    use super::algebra::*;
    use std::{cell::Cell, ops::*};

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct M64(u64);

    thread_local! {
        static MODULUS: Cell<u64> = Cell::new(1);
    }

    impl M64 {
        pub fn modulus() -> u64 {
            MODULUS.with(|m| m.get())
        }

        pub fn set_modulus(x: u64) {
            MODULUS.with(|m| m.set(x));
        }

        pub fn new(s: u64) -> Self {
            Self(s % Self::modulus())
        }
    }

    impl AddAssign<&'_ Self> for M64 {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= Self::modulus() {
                self.0 -= Self::modulus();
            }
        }
    }

    impl SubAssign<&'_ Self> for M64 {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += Self::modulus();
            }
            self.0 -= rhs.0;
        }
    }

    impl MulAssign<&'_ Self> for M64 {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 *= rhs.0;
            self.0 %= Self::modulus();
        }
    }

    impl DivAssign<&'_ Self> for M64 {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    macro_rules! forward_binop {
        ($OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
            impl $OpAssign for M64 {
                fn $op_assign(&mut self, rhs: Self) {
                    self.$op_assign(&rhs);
                }
            }

            impl $Op<&'_ Self> for M64 {
                type Output = Self;
                fn $op(mut self, rhs: &Self) -> Self {
                    self.$op_assign(rhs);
                    self
                }
            }

            impl $Op for M64 {
                type Output = M64;
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

    impl Neg for &'_ M64 {
        type Output = M64;
        fn neg(self) -> M64 {
            let mut res = M64::modulus() - self.0;
            if res == M64::modulus() {
                res = 0u8.into();
            }
            M64(res)
        }
    }

    impl Neg for M64 {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl Default for M64 {
        fn default() -> Self {
            Self(0)
        }
    }

    impl SemiRing for M64 {
        fn one() -> Self {
            Self(1u8.into())
        }
    }
    impl CommRing for M64 {}

    impl Field for M64 {
        fn inv(&self) -> Self {
            self.pow(Self::modulus() - u64::from(2u8))
        }
    }

    pub trait CmpUType<Rhs: Unsigned>: Unsigned {
        type MaxT: Unsigned;
        fn upcast(lhs: Self) -> Self::MaxT;
        fn upcast_rhs(rhs: Rhs) -> Self::MaxT;
        fn downcast(max: Self::MaxT) -> Self;
    }

    macro_rules! impl_cmp_utype {
        (@pairwise $lhs:ident $rhs:ident => $wider:ident) => {
            impl CmpUType<$rhs> for $lhs {
                type MaxT = $wider;
                fn upcast(lhs: Self) -> Self::MaxT {
                    lhs as Self::MaxT
                }
                fn upcast_rhs(rhs: $rhs) -> Self::MaxT {
                    rhs as Self::MaxT
                }
                fn downcast(wider: Self::MaxT) -> Self {
                    wider as Self
                }
            }
        };

        (@cascade $target:ident $($upper:ident)*) => {
            $(
                impl_cmp_utype!(@pairwise $target $upper => $upper);
                impl_cmp_utype!(@pairwise $upper $target => $upper);
            )*
            impl_cmp_utype!(@pairwise $target $target => $target);
        };

        ($target:ident $($rest:ident)*) => {
            impl_cmp_utype!(@cascade $target $($rest)*);
            impl_cmp_utype!($($rest)*);
        };

        () => {};
    }
    impl_cmp_utype!(u8 u16 u32 u64 u128);

    impl<S> From<S> for M64
    where
        u64: CmpUType<S>,
        S: Unsigned,
    {
        fn from(s: S) -> Self {
            Self(u64::downcast(
                u64::upcast_rhs(s) % u64::upcast(M64::modulus()),
            ))
        }
    }

    macro_rules! impl_cast_to_unsigned {
        ($($u:ty)+) => {
            $(
                impl From<M64> for $u {
                    fn from(n: M64) -> Self {
                        n.0 as $u
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl std::fmt::Debug for M64 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl std::fmt::Display for M64 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl std::str::FromStr for M64 {
        type Err = <u64 as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(|x| M64::new(x))
        }
    }

    impl crate::ntt::NTTSpec for M64 {
        fn try_nth_proot(_: u32) -> Option<Self> {
            None
        }
    }
}
