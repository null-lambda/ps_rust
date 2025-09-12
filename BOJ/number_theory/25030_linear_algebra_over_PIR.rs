use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn linear_sieve(n_max: u32) -> (Vec<u32>, Vec<u32>) {
    let mut min_prime_factor = vec![0; n_max as usize + 1];
    let mut primes = Vec::new();

    for i in 2..=n_max {
        if min_prime_factor[i as usize] == 0 {
            primes.push(i);
        }
        for &p in primes.iter() {
            if i * p > n_max {
                break;
            }
            min_prime_factor[(i * p) as usize] = p;
            if i % p == 0 {
                break;
            }
        }
    }

    (min_prime_factor, primes)
}

fn factorize(min_prime_factor: &[u32], n: u32) -> Vec<(u32, u8)> {
    let mut factors = Vec::new();
    let mut x = n;
    while x > 1 {
        let p = min_prime_factor[x as usize];
        if p == 0 {
            factors.push((x as u32, 1));
            break;
        }
        let mut exp = 0;
        while x % p == 0 {
            exp += 1;
            x /= p;
        }
        factors.push((p, exp));
    }

    factors
}

fn for_each_divisor(factors: &[(u32, u8)], mut visitor: impl FnMut(u32)) {
    let mut stack = vec![(1, 0u32)];
    while let Some((mut d, i)) = stack.pop() {
        if i as usize == factors.len() {
            visitor(d);
        } else {
            let (p, exp) = factors[i as usize];
            for _ in 0..=exp {
                stack.push((d, i + 1));
                d *= p;
            }
        }
    }
}

pub mod algebra {
    use std::ops::*;
    pub trait Unsigned:
        Copy
        + Default
        + SemiRing
        + Div<Output = Self>
        + Rem<Output = Self>
        + RemAssign
        + PartialEq
        + Eq
        + PartialOrd
        + Ord
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

        fn pow<U: Unsigned>(&self, exp: U) -> Self {
            let mut res = Self::one();
            let mut base = self.clone();
            let mut exp = exp;
            while exp > U::from(0u8) {
                if exp % U::from(2u8) == U::from(1u8) {
                    res *= base.clone();
                }
                base *= base.clone();
                exp = exp / U::from(2);
            }
            res
        }
    }

    pub trait CommRing: SemiRing + Neg<Output = Self> {}

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
}

pub mod mint {
    use super::algebra::*;
    use std::ops::*;

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct MInt<M: ModSpec>(M::U);

    impl<M: ModSpec> MInt<M> {
        pub fn new(s: M::U) -> Self {
            Self(s % M::MODULUS)
        }
    }

    macro_rules! impl_modspec {
        ($wrapper:ident $spec:ident $u:ty) => {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $spec<const M: $u>;

            impl<const MOD: $u> ModSpec for $spec<MOD> {
                type U = $u;
                const MODULUS: $u = MOD;
            }

            pub type $wrapper<const M: $u> = MInt<$spec<M>>;
        };
    }
    impl_modspec!(M32 __ByU32 u32);
    impl_modspec!(M64 __ByU64 u64);
    impl_modspec!(M128 __ByU128 u128);

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
            self.0 *= rhs.0;
            self.0 %= M::MODULUS;
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
            Self(1.into())
        }
    }
    impl<M: ModSpec> CommRing for MInt<M> {}

    impl<M: ModSpec> Field for MInt<M> {
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

    impl<U, S, M> From<S> for MInt<M>
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
                impl<M: ModSpec<U = $u>> From<MInt<M>> for $u {
                    fn from(n: MInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl<U: std::fmt::Debug, M: ModSpec<U = U>> std::fmt::Debug for MInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::fmt::Display, M: ModSpec<U = U>> std::fmt::Display for MInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::str::FromStr, M: ModSpec<U = U>> std::str::FromStr for MInt<M> {
        type Err = <U as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(|x| MInt::new(x))
        }
    }
}

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
}

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

pub mod linalg {
    use crate::algebra::SemiRing;

    use super::algebra::Field;
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Matrix<T> {
        pub r: usize,
        pub c: usize,
        elem: Vec<T>,
    }

    // impl<T> std::ops::Index<usize> for Matrix<T> {
    //     type Output = [T];

    //     #[inline(always)]
    //     fn index(&self, index: usize) -> &Self::Output {
    //         &self.elem[index * self.c..][..self.c]
    //     }
    // }

    // impl<T> std::ops::IndexMut<usize> for Matrix<T> {
    //     #[inline(always)]
    //     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    //         &mut self.elem[index * self.c..][..self.c]
    //     }
    // }

    impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
        type Output = T;

        #[inline(always)]
        fn index(&self, index: [usize; 2]) -> &Self::Output {
            &self.elem[index[0] * self.c + index[1]]
        }
    }

    impl<T> std::ops::IndexMut<[usize; 2]> for Matrix<T> {
        #[inline(always)]
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            &mut self.elem[index[0] * self.c + index[1]]
        }
    }

    impl<T> Matrix<T> {
        pub fn new(r: usize, elem: impl IntoIterator<Item = T>) -> Self {
            let elem: Vec<_> = elem.into_iter().collect();
            let c = if r == 0 { 0 } else { elem.len() / r };
            assert_eq!(r * c, elem.len());

            Self { r, c, elem }
        }

        pub fn map<S>(self, f: impl FnMut(T) -> S) -> Matrix<S> {
            Matrix {
                r: self.r,
                c: self.c,
                elem: self.elem.into_iter().map(f).collect(),
            }
        }

        pub fn swap_rows(&mut self, i: usize, j: usize) {
            for k in 0..self.c {
                self.elem.swap(i * self.c + k, j * self.c + k);
            }
        }
    }

    impl<T: SemiRing + Copy> Matrix<T> {
        pub fn with_size(r: usize, c: usize) -> Self {
            Self {
                r,
                c,
                elem: vec![T::zero(); r * c],
            }
        }

        pub fn apply(&self, rhs: &[T]) -> Vec<T> {
            assert_eq!(self.c, rhs.len());

            let mut res = vec![T::zero(); self.c];
            for i in 0..self.r {
                for j in 0..self.c {
                    res[i] += self[[i, j]] * rhs[j];
                }
            }
            res
        }
    }

    // Gaussian elimination
    pub fn rref<T: Field + PartialEq + Copy>(mat: &Matrix<T>) -> (usize, T, Matrix<T>) {
        let (r, c) = (mat.r, mat.c);
        let mut mat = mat.clone();
        let mut det = T::one();

        let mut rank = 0;
        for j0 in 0..c {
            let Some(pivot) = (rank..r).find(|&j| mat[[j, j0]] != T::zero()) else {
                continue;
            };
            // let Some(pivot) = (rank..n_rows)
            //     .filter(|&j| jagged[j][c] != Frac64::zero())
            //     .max_by_key(|&j| jagged[j][c].abs())
            // else {
            //     continue;
            // };

            if pivot != rank {
                mat.swap_rows(rank, pivot);
                det = -det;
            }

            det *= mat[[rank, j0]];
            let inv_x = mat[[rank, j0]].inv();
            for j in 0..c {
                mat[[rank, j]] *= inv_x;
            }

            for i in 0..r {
                if i == rank {
                    continue;
                }

                let coeff = mat[[i, j0]];
                for j in 0..c {
                    let f = mat[[rank, j]] * coeff;
                    mat[[i, j]] -= f;
                }
            }
            rank += 1;
        }

        if rank != mat.r {
            det = T::zero();
        };

        (rank, det, mat)
    }
}

use algebra::SemiRing;
use linalg::Matrix;
// type M = mint_dyn::M64;
type M<const P: u32> = mint_mont::M32<P>;

// Solve Ax = B
fn solve_mod_p<T: algebra::Field + From<u32> + Into<u32> + PartialEq + Copy>(
    mat: &Matrix<u32>,
    rhs: &[u32],
) -> Option<(Vec<u32>, impl FnOnce() -> Vec<Vec<u32>>)> {
    // M::set_modulus(p as u64);
    let (r, c) = (mat.r, mat.c);
    let mut ext = Matrix::<T>::with_size(r, c + 1);
    for i in 0..r {
        for j in 0..c {
            ext[[i, j]] = T::from(mat[[i, j]]);
        }
        ext[[i, c]] = T::from(rhs[i]);
    }

    let (_, _, rref) = linalg::rref(&ext);

    let mut sol = vec![0u32; c];
    let mut is_pivot_col = vec![false; c];
    for i in 0..r {
        if let Some(j) = (0..c).find(|&j| rref[[i, j]] != T::zero()) {
            is_pivot_col[j] = true;
            sol[j] = rref[[i, c]].into();
        } else if rref[[i, c]] != T::zero() {
            return None;
        }
    }

    let ker_basis = move || {
        let mut res = vec![];

        for f in 0..c {
            if is_pivot_col[f] {
                continue;
            }

            let mut bs = vec![0u32; c];
            bs[f] = 1;

            let mut pi = 0;
            for pj in 0..f {
                if !is_pivot_col[pj] {
                    continue;
                }
                bs[pj] = (-rref[[pi, f]]).into();

                pi += 1;
            }
            res.push(bs);
        }

        res
    };

    Some((sol, ker_basis))
}

fn ker_mod_p_pow<const P: u32, T: algebra::Field + From<u32> + Into<u32> + PartialEq + Copy>(
    p: u32,
    mat: &Matrix<u32>,
    e: u8,
) -> Vec<Vec<u32>> {
    assert!(e >= 1);

    let mut ker_basis = solve_mod_p::<T>(&mat, &vec![0; mat.c]).unwrap().1();
    let mut p_pow = p;

    // Lifting
    for _k in 2..=e {
        let mut t_basis = None;
        for b in std::mem::take(&mut ker_basis) {
            let mut r = mat.apply(&b);
            for ri in &mut r {
                assert!(*ri % p_pow == 0);
                *ri = *ri / p_pow;
                *ri = (p - *ri % p) % p;
            }

            if let Some((t, ker_lazy)) = solve_mod_p::<T>(&mat, &r) {
                ker_basis.push((0..mat.c).map(|i| b[i] + t[i] * p_pow).collect());

                if t_basis.is_none() {
                    t_basis = Some(ker_lazy());
                }
            }
        }

        if let Some(t_basis) = t_basis {
            for mut t in t_basis {
                for ti in &mut t {
                    *ti *= p_pow;
                }
                ker_basis.push(t);
            }
        }

        p_pow *= p;
    }

    // for b in &ker_basis {
    //     assert!(mat
    //         .apply(b)
    //         .iter()
    //         .map(|&x| x % p_pow)
    //         .eq((0..mat.r).map(|_| 0)))
    // }

    ker_basis
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: u32, b: u32) -> (u32, i32, i32) {
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - (q as i32) * x.1));
        y = (y.1, (y.0 - (q as i32) * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}

// Find (d, x, y) satisfying d = gcd(abs(a), abs(b)) and a * x + b * y = d
fn egcd_i32(a: i32, b: i32) -> (i32, i32, i32) {
    let (d, x, y) = egcd(a.abs() as u32, b.abs() as u32);
    (d as i32, x as i32 * a.signum(), y as i32 * b.signum())
}

fn crt(a1: u32, m1: u32, a2: u32, m2: u32) -> Option<(u32, u32)> {
    let (d, x, _y) = egcd(m1, m2);
    let m = m1 / d * m2;
    let da = ((a2 as i32 - a1 as i32) % m as i32 + m as i32) as u32 % m;
    if da % d != 0 {
        return None;
    }
    let mut x = ((x % m as i32) + m as i32) as u32 % m;
    x = (da / d % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}

fn span_ker_mod_p_pow<
    const P: u32,
    T: algebra::Field + From<u32> + Into<u32> + PartialEq + Copy,
>(
    p: u32,
    mat: &Matrix<u32>,
    e: u8,
) -> Vec<Vec<u32>> {
    let m = p.pow(e as u32);
    let basis = ker_mod_p_pow::<P, T>(p, &mat, e);

    let mut res = vec![vec![0; mat.c]];
    for b in basis {
        let ord = m / b.iter().fold(m, |acc, &x| gcd(acc, x));
        for prev in std::mem::take(&mut res) {
            for c in 0..ord {
                res.push((0..mat.c).map(|i| (prev[i] + c * b[i]) % m).collect());
            }
        }
        res.sort_unstable();
        res.dedup();
    }

    res
}

fn crt_vec(as1: &[u32], m1: u32, as2: &[u32], m2: u32) -> Vec<u32> {
    as1.iter()
        .zip(as2)
        .map(|(&a1, &a2)| crt(a1, m1, a2, m2).unwrap().0)
        .collect()
}

macro_rules! monomorphize_primes {
    ($($p:expr),* $(,)?) => {
        fn monomorphized(p: u32, mat: &Matrix<u32>, e: u8) -> Vec<Vec<u32>> {
            match p {
                $($p => {
                    if $p == 2 {
                        span_ker_mod_p_pow::<$p, mint::M32<$p>>(p, mat, e)
                    } else if $p <= 30 {
                        span_ker_mod_p_pow::<$p, mint_mont::M32<$p>>(p, mat, e)
                    } else {
                        mint_dyn::M64::set_modulus(p as u64);
                        span_ker_mod_p_pow::<0, mint_dyn::M64>($p, mat, e)
                    }
                })*
                _ => panic!(),
            }
        }
    };
}
monomorphize_primes!(
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199
);

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let (mpf, _) = linear_sieve(n as u32);
    let factors = factorize(&mpf, n as u32);

    let mut mat = Matrix::<u32>::with_size(n, n);
    for i in 0..n {
        let mut pow_i = 1;
        for j in 0..n {
            mat[[i, j]] = pow_i as u32;
            pow_i = pow_i * i % n;
        }

        mat[[i, i]] = (mat[[i, i]] + n as u32 - 1) % n as u32;
    }

    let mut res = vec![vec![0; n]];
    let mut m = 1;
    for (p, e) in factors {
        let p_pow = p.pow(e as u32);
        let vs = monomorphized(p, &mat, e);

        for prev in std::mem::take(&mut res) {
            for v in &vs {
                res.push(crt_vec(&prev, m, &v, p_pow));
            }
        }

        m *= p_pow;
    }

    for v in &mut res {
        v.reverse();
    }
    res.sort_unstable();

    for v in res {
        for x in v {
            write!(output, "{} ", x).unwrap();
        }
        writeln!(output).unwrap();
    }
}
