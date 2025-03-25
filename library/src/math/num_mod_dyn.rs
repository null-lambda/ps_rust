pub mod num_mod_dyn {
    use super::algebra::*;
    use std::{cell::Cell, ops::*};

    pub trait Unsigned:
        Copy
        + Default
        + SemiRing
        + Div<Output = Self>
        + Rem<Output = Self>
        + RemAssign
        + PartialEq
        + PartialOrd
        + From<u8>
    {
        fn zero() -> Self {
            Self::default()
        }
        fn one() -> Self;
    }

    macro_rules! impl_unsigned {
        ($($t:ty)+) => {
            $(
                impl Unsigned for $t {
                    fn one() -> Self {
                        1
                    }
                }
            )+
        };
    }
    impl_unsigned!(u8 u16 u32 u64 u128 usize);

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct ModInt64(u64);

    thread_local! {
        static MODULUS: Cell<u64> = Cell::new(1);
    }

    impl ModInt64 {
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

    impl AddAssign<&'_ Self> for ModInt64 {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= Self::modulus() {
                self.0 -= Self::modulus();
            }
        }
    }

    impl SubAssign<&'_ Self> for ModInt64 {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += Self::modulus();
            }
            self.0 -= rhs.0;
        }
    }

    impl MulAssign<&'_ Self> for ModInt64 {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 *= rhs.0;
            self.0 %= Self::modulus();
        }
    }

    macro_rules! forward_ref_binop {
        ($($OpAssign:ident $op_assign:ident),+) => {
            $(
                impl $OpAssign for ModInt64 {
                    fn $op_assign(&mut self, rhs: Self) {
                        self.$op_assign(&rhs);
                    }
                }
            )+
        };
    }
    forward_ref_binop!(AddAssign add_assign, MulAssign mul_assign, SubAssign sub_assign);

    macro_rules! impl_op_by_op_assign {
        ($($Op:ident $op:ident $op_assign:ident),+) => {
            $(
                impl $Op<&'_ Self> for ModInt64 {
                    type Output = Self;
                    fn $op(mut self, rhs: &Self) -> Self {
                        self.$op_assign(rhs);
                        self
                    }
                }

                impl $Op for ModInt64 {
                    type Output = ModInt64;
                    fn $op(self, rhs: Self) -> Self::Output {
                        self.clone().$op(&rhs)
                    }
                }
            )+
        };
    }
    impl_op_by_op_assign!(Add add add_assign, Mul mul mul_assign, Sub sub sub_assign);

    impl Neg for &'_ ModInt64 {
        type Output = ModInt64;
        fn neg(self) -> ModInt64 {
            let mut res = ModInt64::modulus() - self.0;
            if res == ModInt64::modulus() {
                res = 0u8.into();
            }
            ModInt64(res)
        }
    }

    impl Neg for ModInt64 {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl Default for ModInt64 {
        fn default() -> Self {
            Self(0)
        }
    }

    impl SemiRing for ModInt64 {
        fn one() -> Self {
            Self(1u8.into())
        }
    }
    impl CommRing for ModInt64 {}

    impl DivAssign<&'_ Self> for ModInt64
    where
        ModInt64: PowBy<u64>,
    {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    impl DivAssign for ModInt64
    where
        ModInt64: PowBy<u64>,
    {
        fn div_assign(&mut self, rhs: Self) {
            self.div_assign(&rhs);
        }
    }

    impl Div<&'_ Self> for ModInt64
    where
        ModInt64: PowBy<u64>,
    {
        type Output = Self;
        fn div(mut self, rhs: &Self) -> Self {
            self.div_assign(rhs);
            self
        }
    }

    impl Div for ModInt64
    where
        ModInt64: PowBy<u64>,
    {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            self / &rhs
        }
    }

    impl Field for ModInt64
    where
        ModInt64: PowBy<u64>,
    {
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
        (
            $( $($lower:ident)* < $target:ident < $($upper:ident)* ),* $(,)?
        ) => {
            $(
                $(
                    impl CmpUType<$lower> for $target {
                        type MaxT = $target;
                        fn upcast(lhs: Self) -> Self::MaxT {
                            lhs as Self::MaxT
                        }
                        fn upcast_rhs(rhs: $lower) -> Self::MaxT {
                            rhs as Self::MaxT
                        }
                        fn downcast(max: Self::MaxT) -> Self {
                            max as Self
                        }
                    }
                )*
                impl CmpUType<$target> for $target {
                    type MaxT = $target;
                    fn upcast(lhs: Self) -> Self::MaxT {
                        lhs as Self::MaxT
                    }
                    fn upcast_rhs(rhs: $target) -> Self::MaxT {
                        rhs as Self::MaxT
                    }
                    fn downcast(max: Self::MaxT) -> Self {
                        max as Self
                    }
                }
                $(
                    impl CmpUType<$upper> for $target {
                        type MaxT = $upper;
                        fn upcast(lhs: Self) -> Self::MaxT {
                            lhs as Self::MaxT
                        }
                        fn upcast_rhs(rhs: $upper) -> Self::MaxT {
                            rhs as Self::MaxT
                        }
                        fn downcast(max: Self::MaxT) -> Self {
                            max as Self
                        }
                    }
                )*
            )*
        };
    }
    impl_cmp_utype!(
        < u8 < u16 u32 u64 u128,
        u8 < u16 < u32 u64 u128,
        u8 u16 < u32 < u64 u128,
        u8 u16 u32 < u64 < u128,
        u8 u16 u32 u64 < u128 <,
    );

    impl<S> From<S> for ModInt64
    where
        u64: CmpUType<S>,
        S: Unsigned,
    {
        fn from(s: S) -> Self {
            Self(u64::downcast(
                u64::upcast_rhs(s) % u64::upcast(ModInt64::modulus()),
            ))
        }
    }

    macro_rules! impl_cast_to_unsigned {
        ($($u:ty)+) => {
            $(
                impl From<ModInt64> for $u {
                    fn from(n: ModInt64) -> Self {
                        n.0 as $u
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl std::fmt::Debug for ModInt64 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl std::fmt::Display for ModInt64 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl std::str::FromStr for ModInt64 {
        type Err = <u64 as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(|x| ModInt64::new(x))
        }
    }

    impl crate::ntt::NTTSpec for ModInt64 {
        fn try_nth_proot(_: u32) -> Option<Self> {
            None
        }
    }
}
