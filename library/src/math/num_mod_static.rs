pub mod algebra {
    use std::ops::*;
    pub trait SemiRing:
        Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + for<'a> Add<&'a Self, Output = Self>
        + for<'a> Sub<&'a Self, Output = Self>
        + for<'a> Mul<&'a Self, Output = Self>
        + for<'a> AddAssign<&'a Self>
        + for<'a> SubAssign<&'a Self>
        + for<'a> MulAssign<&'a Self>
        + Default
        + Clone
    {
        fn zero() -> Self {
            Self::default()
        }
        fn one() -> Self;
    }

    pub trait CommRing: SemiRing + Neg<Output = Self> {}

    pub trait PowBy<E> {
        fn pow(&self, exp: E) -> Self;
    }

    pub trait Field:
        CommRing
        + Div<Output = Self>
        + DivAssign
        + for<'a> Div<&'a Self, Output = Self>
        + for<'a> DivAssign<&'a Self>
    {
        fn inv(&self) -> Self;
    }

    macro_rules! impl_semiring {
        ($($t:ty)+) => {
            $(
                impl SemiRing for $t {
                    fn one() -> Self {
                        1
                    }
                }
            )+
        };
    }

    macro_rules! impl_commring {
        ($($t:ty)+) => {
            $(
                impl CommRing for $t {}
            )+
        };
    }

    impl_semiring!(u8 u16 u32 u64 u128 usize);
    impl_semiring!(i8 i16 i32 i64 i128 isize);
    impl_commring!(i8 i16 i32 i64 i128 isize);

    macro_rules! impl_powby {
        ($(($uexp:ty, $iexp:ty),)+) => {
            $(
                impl<R: CommRing> PowBy<$uexp> for R {
                    fn pow(&self, exp: $uexp) -> R {
                        let mut res = R::one();
                        let mut base = self.clone();
                        let mut exp = exp;
                        while exp > 0 {
                            if exp & 1 == 1 {
                                res *= base.clone();
                            }
                            base *= base.clone();
                            exp >>= 1;
                        }
                        res
                    }
                }

                impl<R: Field> PowBy<$iexp> for R {
                    fn pow(&self, exp: $iexp) -> R {
                        if exp < 0 {
                            self.inv().pow((-exp) as $uexp)
                        } else {
                            self.pow(exp as $uexp)
                        }
                    }
                }
            )+
        };
    }
    impl_powby!(
        (u8, i8),
        (u16, i16),
        (u32, i32),
        (u64, i64),
        (u128, i128),
        (usize, isize),
    );
}

pub mod num_mod {
    use super::algebra::*;
    use std::ops::*;

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

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
    }

    pub trait ByPrime: ModSpec {}

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct ModInt<M: ModSpec>(M::U);

    impl<M: ModSpec> ModInt<M> {
        pub fn new(s: M::U) -> Self {
            Self(s % M::MODULUS)
        }
    }

    macro_rules! impl_modspec {
        ($($t:ident $u:ty),+) => {
            $(
                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                pub struct $t<const M: $u>;

                impl<const MOD: $u> ModSpec for $t<MOD> {
                    type U = $u;
                    const MODULUS: $u = MOD;
                }

            )+
        };
    }
    impl_modspec!(
        ByU32 u32, ByU32Prime u32,
        ByU64 u64, ByU64Prime u64,
        ByU128 u128, ByU128Prime u128
    );

    macro_rules! impl_by_prime {
        ($($t:ident $u:ty),+) => {
            $(
                impl<const MOD: $u> ByPrime for $t<MOD> {}
            )+
        };
    }
    impl_by_prime!(ByU32Prime u32, ByU64Prime u64, ByU128Prime u128);

    impl<M: ModSpec> AddAssign<&'_ Self> for ModInt<M> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<M: ModSpec> SubAssign<&'_ Self> for ModInt<M> {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<M: ModSpec> MulAssign<&'_ Self> for ModInt<M> {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 *= rhs.0;
            self.0 %= M::MODULUS;
        }
    }

    macro_rules! forward_ref_binop {
        ($($OpAssign:ident $op_assign:ident),+) => {
            $(
                impl<M: ModSpec> $OpAssign for ModInt<M> {
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
                impl<M: ModSpec> $Op<&'_ Self> for ModInt<M> {
                    type Output = Self;
                    fn $op(mut self, rhs: &Self) -> Self {
                        self.$op_assign(rhs);
                        self
                    }
                }

                impl< M: ModSpec> $Op for ModInt<M> {
                    type Output = ModInt<M>;
                    fn $op(self, rhs: Self) -> Self::Output {
                        self.clone().$op(&rhs)
                    }
                }
            )+
        };
    }
    impl_op_by_op_assign!(Add add add_assign, Mul mul mul_assign, Sub sub sub_assign);

    impl<M: ModSpec> Neg for &'_ ModInt<M> {
        type Output = ModInt<M>;
        fn neg(self) -> ModInt<M> {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            ModInt(res)
        }
    }

    impl<M: ModSpec> Neg for ModInt<M> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl<M: ModSpec> Default for ModInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> SemiRing for ModInt<M> {
        fn one() -> Self {
            Self(1.into())
        }
    }
    impl<M: ModSpec> CommRing for ModInt<M> {}

    impl<M: ByPrime> DivAssign<&'_ Self> for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    impl<M: ByPrime> DivAssign for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn div_assign(&mut self, rhs: Self) {
            self.div_assign(&rhs);
        }
    }

    impl<M: ByPrime> Div<&'_ Self> for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        type Output = Self;
        fn div(mut self, rhs: &Self) -> Self {
            self.div_assign(rhs);
            self
        }
    }

    impl<M: ByPrime> Div for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            self / &rhs
        }
    }

    impl<M: ByPrime> Field for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn inv(&self) -> Self {
            self.pow(M::MODULUS - M::U::from(2))
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

    impl<U, S, M> From<S> for ModInt<M>
    where
        U: CmpUType<S>,
        S: Unsigned,
        M: ModSpec<U = U>,
    {
        fn from(s: S) -> Self {
            Self(U::downcast(U::upcast_rhs(s) % U::upcast(M::MODULUS)))
        }
    }

    macro_rules! impl_cast_to_unsigned {
        ($($u:ty)+) => {
            $(
                impl<M: ModSpec<U = $u>> From<ModInt<M>> for $u {
                    fn from(n: ModInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl<U: std::fmt::Debug, M: ModSpec<U = U>> std::fmt::Debug for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::fmt::Display, M: ModSpec<U = U>> std::fmt::Display for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::str::FromStr, M: ModSpec<U = U>> std::str::FromStr for ModInt<M> {
        type Err = <U as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(|x| ModInt::new(x))
        }
    }
}
