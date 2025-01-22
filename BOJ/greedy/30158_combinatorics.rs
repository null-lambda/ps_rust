use std::{collections::HashMap, io::Write};

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

pub mod num_mod_static {
    use std::ops::*;

    pub trait Unsigned:
        Copy
        + Default
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
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

    pub trait CommRing:
        Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Neg<Output = Self>
        + Mul<Output = Self>
        + MulAssign
        + Default
        + Clone
    {
        fn zero() -> Self {
            Self::default()
        }
        fn one() -> Self;
    }

    pub trait PowBy<E> {
        fn pow(&self, exp: E) -> Self;
    }

    macro_rules! impl_powby {
        ($($exp:ty)+) => {
            $(
                impl<R: CommRing> PowBy<$exp> for R {
                    fn pow(&self, exp: $exp) -> R {
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
            )+
        };
    }
    impl_powby!(u16 u32 u64 u128);

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
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
    impl_modspec!(ByU16 u16, ByU32 u32, ByU64 u64, ByU128 u128);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NaiveModInt<M: ModSpec>(M::U);

    impl<M: ModSpec> AddAssign for NaiveModInt<M> {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<M: ModSpec> Add for NaiveModInt<M> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            let mut res = self.clone();
            res += rhs;
            res
        }
    }

    impl<M: ModSpec> SubAssign for NaiveModInt<M> {
        fn sub_assign(&mut self, rhs: Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<M: ModSpec> Sub for NaiveModInt<M> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            let mut res = self.clone();
            res -= rhs;
            res
        }
    }

    impl<M: ModSpec> Neg for NaiveModInt<M> {
        type Output = Self;
        fn neg(self) -> Self {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            Self(res)
        }
    }

    impl<M: ModSpec> MulAssign for NaiveModInt<M> {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = self.0 * rhs.0 % M::MODULUS;
        }
    }

    impl<M: ModSpec> Mul for NaiveModInt<M> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            let mut res = self.clone();
            res *= rhs;
            res
        }
    }

    impl<M: ModSpec> Default for NaiveModInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> CommRing for NaiveModInt<M> {
        fn one() -> Self {
            Self(1.into())
        }
    }

    macro_rules! impl_from {
        ($($u:ty)+) => {
            $(
                impl<M: ModSpec<U = $u>> From<$u> for NaiveModInt<M> {
                    fn from(n: $u) -> Self {
                        Self(n % M::MODULUS)
                    }
                }

                impl<M: ModSpec<U = $u>> From<NaiveModInt<M>> for $u {
                    fn from(n: NaiveModInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_from!(u16 u32 u64 u128);

    pub mod montgomery {
        use super::{CommRing, Unsigned};
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

        macro_rules! impl_modspec {
            ($($t:ident, $t_impl:ident, U = $single:ty, D = $double:ty, EXP = $exp:expr, LOG2_EXP = $log2_exp: expr);+) => {
                $(
                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                    pub struct $t<const M: $single>;

                    impl<const MOD: $single> ModSpec for $t<MOD> {
                        type U = $single;
                        type D = $double;
                        const MODULUS: $single = MOD;
                        const M_INV: $single = $t_impl::eval_m_inv(MOD);
                        const R2: $single = $t_impl::eval_r2(MOD);

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

                    mod $t_impl {
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
                )+
            };
        }
        impl_modspec!(
            ByU32, u32_impl, U = u32, D = u64, EXP = 32, LOG2_EXP = 5;
            ByU64, u64_impl, U = u64, D = u128, EXP = 64, LOG2_EXP = 6
        );

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct Montgomery<M: ModSpec>(M::U);

        impl<M: ModSpec> AddAssign for Montgomery<M> {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
                if self.0 >= M::MODULUS {
                    self.0 -= M::MODULUS;
                }
            }
        }

        impl<M: ModSpec> Add for Montgomery<M> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                let mut res = self.clone();
                res += rhs;
                res
            }
        }

        impl<M: ModSpec> SubAssign for Montgomery<M> {
            fn sub_assign(&mut self, rhs: Self) {
                if self.0 < rhs.0 {
                    self.0 += M::MODULUS;
                }
                self.0 -= rhs.0;
            }
        }

        impl<M: ModSpec> Sub for Montgomery<M> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                let mut res = self.clone();
                res -= rhs;
                res
            }
        }

        impl<M: ModSpec> Neg for Montgomery<M> {
            type Output = Self;
            fn neg(self) -> Self {
                let mut res = M::MODULUS - self.0;
                if res == M::MODULUS {
                    res = 0.into();
                }
                Self(res)
            }
        }

        impl<M: ModSpec> MulAssign for Montgomery<M> {
            fn mul_assign(&mut self, rhs: Self) {
                self.0 = M::reduce_double(M::to_double(self.0) * M::to_double(rhs.0));
            }
        }

        impl<M: ModSpec> Mul for Montgomery<M> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                let mut res = self.clone();
                res *= rhs;
                res
            }
        }

        impl<M: ModSpec> Default for Montgomery<M> {
            fn default() -> Self {
                Self(M::U::default())
            }
        }

        impl<M: ModSpec> CommRing for Montgomery<M> {
            fn one() -> Self {
                Self(1.into()) * Self(M::R2)
            }
        }

        macro_rules! impl_from {
            ($($u:ty)+) => {
                $(
                    impl<M: ModSpec<U = $u>> From<$u> for Montgomery<M> {
                        fn from(x: $u) -> Self {
                            Self(x) * Self(M::R2)
                        }
                    }

                    impl<M: ModSpec<U = $u>> From<Montgomery<M>> for $u {
                        fn from(x: Montgomery<M>) -> Self {
                            M::reduce_double(M::to_double(x.0))
                        }
                    }
                )+
            };
        }
        impl_from!(u32 u64);

        pub type MontgomeryU32<const M: u32> = Montgomery<ByU32<M>>;
        pub type MontgomeryU64<const M: u64> = Montgomery<ByU64<M>>;
    }
}

const P: u32 = 1_000_000_007;
type ModP = num_mod_static::montgomery::MontgomeryU32<P>;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    const N_MAX: usize = 100_000;
    let fact: Vec<ModP> = std::iter::once(1.into())
        .chain((1..=N_MAX).scan(1.into(), |acc, i| {
            *acc *= ModP::from(i as u32);
            Some(*acc)
        }))
        .collect();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let mut freq: HashMap<u32, u32> = Default::default();
        for _ in 0..n {
            *freq.entry(input.value()).or_default() += 1;
        }

        let mut coeff_freq: HashMap<i32, u32> = Default::default();
        for f in freq.into_values() {
            for i in 0..f as i32 {
                *coeff_freq.entry(2 * i - (f - 1) as i32).or_default() += 1;
            }
        }

        let mut ans = ModP::from(1);
        for (_, f) in coeff_freq {
            ans *= fact[f as usize];
        }
        writeln!(output, "{}", u32::from(ans)).unwrap();
    }
}
