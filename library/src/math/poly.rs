pub mod algebra {
    use std::ops::*;
    pub trait Unsigned:
        Copy
        + Default
        + SemiRing
        + Div<Output = Self>
        + Rem<Output = Self>
        + RemAssign
        + Eq
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
        + Eq
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

pub mod mint_mont {
    use crate::algebra::*;
    use std::ops::*;

    pub trait ModSpec: Copy + Eq {
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

pub mod ntt {
    use super::algebra::*;

    fn bit_reversal_perm<T>(xs: &mut [T]) {
        let n = xs.len();
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        if n == 1 {
            return;
        }

        for i in 0..n as u32 {
            let rev = i.reverse_bits() >> (u32::BITS - n_log2);
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }
    }

    pub fn run<T: CommRing>(proot: T, xs: &mut [T]) {
        if xs.len() <= 20 {
            naive(proot, xs);
        } else {
            radix4(proot, xs);
        }
    }

    // naive O(n^2)
    pub fn naive<T: CommRing>(proot: T, xs: &mut [T]) {
        let n = xs.len().next_power_of_two();
        let proot_pow: Vec<T> =
            std::iter::successors(Some(T::one()), |acc| Some(acc.clone() * &proot))
                .take(n)
                .collect();
        let res: Vec<_> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| xs[j].clone() * &proot_pow[(i * j) % n])
                    .fold(T::zero(), |acc, x| acc + x)
            })
            .collect();
        for (r, x) in res.into_iter().zip(xs) {
            *x = r;
        }
    }

    pub fn radix4<T: CommRing>(proot: T, xs: &mut [T]) {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        bit_reversal_perm(xs);

        let base: Vec<_> = (0..n_log2)
            .scan(proot.clone(), |acc, _| {
                let prev = acc.clone();
                *acc *= acc.clone();
                Some(prev)
            })
            .collect();

        let mut proot_pow: Vec<T> = vec![T::zero(); n];
        proot_pow[0] = T::one();

        let quartic_root = proot.pow(n as u32 / 4);

        let update_proot_pow = |proot_pow: &mut [T], k: u32| {
            let step = 1 << k;
            let base = base[(n_log2 - k - 1) as usize].clone();
            for i in (0..step).rev() {
                proot_pow[i * 2 + 1] = proot_pow[i].clone() * base.clone();
                proot_pow[i * 2] = proot_pow[i].clone();
            }
        };

        let mut k = 0;
        if n_log2 % 2 == 1 {
            let step = 1 << k;
            // radix-2 butterfly
            update_proot_pow(&mut proot_pow, k);
            for t in xs.chunks_exact_mut(step * 2) {
                let (t0, t1) = t.split_at_mut(step);
                for (a0, a1) in t0.into_iter().zip(t1) {
                    let b0 = a0.clone();
                    let b1 = a1.clone();
                    *a0 = b0.clone() + b1.clone();
                    *a1 = b0 - b1;
                }
            }
            k += 1;
        }
        while k < n_log2 {
            let step = 1 << k;
            // radix-4 butterfly
            update_proot_pow(&mut proot_pow, k);
            update_proot_pow(&mut proot_pow, k + 1);

            for t in xs.chunks_exact_mut(step * 4) {
                let (t0, rest) = t.split_at_mut(step);
                let (t1, rest) = rest.split_at_mut(step);
                let (t2, t3) = rest.split_at_mut(step);

                for ((((a0, a1), a2), a3), pow1) in
                    t0.into_iter().zip(t1).zip(t2).zip(t3).zip(&proot_pow)
                {
                    let pow2 = pow1.clone() * pow1;
                    let pow1_shift = pow1.clone() * &quartic_root;

                    let b0 = a0.clone();
                    let b1 = a1.clone() * &pow2;
                    let b2 = a2.clone();
                    let b3 = a3.clone() * &pow2;

                    let c0 = b0.clone() + &b1;
                    let c1 = b0.clone() - &b1;
                    let c2 = (b2.clone() + &b3) * pow1;
                    let c3 = (b2 - b3) * pow1_shift;

                    *a0 = c0.clone() + &c2;
                    *a1 = c1.clone() + &c3;
                    *a2 = c0.clone() - &c2;
                    *a3 = c1.clone() - &c3;
                }
            }
            k += 2;
        }
    }

    pub trait NTTSpec:
        SemiRing + From<u32> + From<u64> + From<u128> + Into<u32> + std::fmt::Debug
    {
        fn try_nth_proot(_n: u32) -> Option<Self> {
            None
        }
    }

    // TODO: replace with const fn
    pub mod sample {
        // Check:
        // https://oeis.org/A039687
        // https://oeis.org/A050526
        // https://oeis.org/A300407
        use super::NTTSpec;
        use crate::algebra::SemiRing;
        use crate::mint_mont::*;
        pub mod p13631489 {
            use super::*;
            pub const P: u32 = 13631489;
            pub const GEN: u32 = 15;
            pub type M = M32<P>;
            impl NTTSpec for M {
                fn try_nth_proot(n: u32) -> Option<Self> {
                    assert!((P - 1) % n as u32 == 0);
                    M::from(GEN).pow((P - 1) / n as u32).into()
                }
            }
        }

        pub mod p104857601 {
            use super::*;
            pub const P: u32 = 104857601;
            pub const GEN: u32 = 3;
            pub type M = M32<P>;
            impl NTTSpec for M {
                fn try_nth_proot(n: u32) -> Option<Self> {
                    assert!((P - 1) % n as u32 == 0);
                    M::from(GEN).pow((P - 1) / n as u32).into()
                }
            }
        }

        pub mod p167772161 {
            use super::*;
            pub const P: u32 = 167772161;
            pub const GEN: u32 = 3;
            pub type M = M32<P>;
            impl NTTSpec for M {
                fn try_nth_proot(n: u32) -> Option<Self> {
                    assert!((P - 1) % n as u32 == 0);
                    M::from(GEN).pow((P - 1) / n as u32).into()
                }
            }
        }

        pub mod p998244353 {
            use super::*;
            pub const P: u32 = 998244353;
            pub const GEN: u32 = 3;
            pub type M = M32<P>;
            impl NTTSpec for M {
                fn try_nth_proot(n: u32) -> Option<Self> {
                    assert!((P - 1) % n as u32 == 0);
                    M::from(GEN).pow((P - 1) / n as u32).into()
                }
            }
        }

        pub mod p1092616193 {
            use super::*;
            pub const P: u32 = 1092616193;
            pub const GEN: u32 = 3;
            pub type M = M32<P>;
            impl NTTSpec for M {
                fn try_nth_proot(n: u32) -> Option<Self> {
                    assert!((P - 1) % n as u32 == 0);
                    M::from(GEN).pow((P - 1) / n as u32).into()
                }
            }
        }
    }
}

pub mod poly {
    use crate::algebra::*;
    use crate::ntt::{self, NTTSpec};
    use std::cell::OnceCell;
    use std::collections::VecDeque;
    use std::ops::*;

    // TODO: move convolution part to the trait `NTTSpec` (or rename it Conv)
    // and implement const function gen_proot for mint_mont
    //
    // TODO: cast modint from signed integers
    //
    // TODO: add from_const for modints (remove TLS variable)

    // shouldn't belong here
    fn crt3_coeff_u64<
        T0: NTTSpec + Field,
        T1: NTTSpec + Field,
        T2: NTTSpec + Field,
        TR: NTTSpec + Field,
    >(
        ps: [u64; 3],
    ) -> ([u64; 3], (T0, T1, T2)) {
        let qs = [ps[1] * ps[2], ps[0] * ps[2], ps[0] * ps[1]];
        let rs = (
            T0::from(qs[0]).inv(),
            T1::from(qs[1]).inv(),
            T2::from(qs[2]).inv(),
        );
        (qs, rs)
    }

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Poly<T>(pub Vec<T>);

    impl<T: SemiRing> Poly<T> {
        pub fn new(coeffs: Vec<T>) -> Self {
            Self(coeffs)
        }
        pub fn zero() -> Self {
            Self(vec![])
        }
        pub fn one() -> Self {
            Self(vec![T::one()])
        }
        pub fn pop_zeros(&mut self) {
            while self.0.last().filter(|&c| c == &T::zero()).is_some() {
                self.0.pop();
            }
        }
        pub fn len(&self) -> usize {
            self.0.len()
        }
        pub fn is_zero(&mut self) -> bool {
            self.pop_zeros();
            self.0.is_empty()
        }

        pub fn degree(&mut self) -> usize {
            self.pop_zeros();
            self.0.len().saturating_sub(1)
        }
        pub fn leading_coeff(&self) -> T {
            self.0.last().cloned().unwrap_or(T::zero())
        }
        pub fn coeff(&self, i: usize) -> T {
            self.0.get(i).cloned().unwrap_or_default()
        }
        pub fn eval(&self, x: T) -> T {
            let mut res = T::zero();
            for c in self.0.iter().rev() {
                res *= &x;
                res += c;
            }
            res
        }
        pub fn reverse(&mut self) {
            self.0.reverse()
        }
        pub fn mod_xk(mut self, k: usize) -> Self {
            if self.degree() >= k {
                self.0.truncate(k);
            }
            self
        }
        pub fn mul_xk(mut self, k: usize) -> Self {
            ((0..k).map(|_| T::zero()))
                .chain(std::mem::take(&mut self.0))
                .collect()
        }
        pub fn div_xk(&self, k: usize) -> Self {
            Self(self.0[k.min(self.0.len())..].to_vec())
        }
        pub fn factor_out_xk(&self) -> (usize, Self) {
            if let Some(k) = self.0.iter().position(|x| x != &T::zero()) {
                let q = self.0[k..].to_vec();
                (k, Self::new(q))
            } else {
                (0, Self::zero())
            }
        }
    }
    impl<T> FromIterator<T> for Poly<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self(iter.into_iter().collect())
        }
    }
    impl<T: NTTSpec + Field> Poly<T> {
        pub fn prod(xs: impl IntoIterator<Item = Self>) -> Self {
            let mut factors: VecDeque<_> = xs.into_iter().collect();

            while factors.len() >= 2 {
                let mut lhs = factors.pop_front().unwrap();
                let rhs = factors.pop_front().unwrap();
                lhs *= rhs;
                factors.push_back(lhs);
            }

            factors.pop_front().unwrap_or(Self::one())
        }
        pub fn sum_frac(fs: impl IntoIterator<Item = (Self, Self)>) -> (Self, Self) {
            let mut factors: VecDeque<_> = fs.into_iter().collect();

            while factors.len() >= 2 {
                let (n0, d0) = factors.pop_front().unwrap();
                let (n1, d1) = factors.pop_front().unwrap();
                factors.push_back((n0 * d1.clone() + n1 * d0.clone(), d0 * d1));
            }

            factors.pop_front().unwrap_or((Self::zero(), Self::one()))
        }
        pub fn interp(ps: impl IntoIterator<Item = (T, T)>) -> Self {
            // Potential optimization: Reduce redundant computation of the polynomial tree of `f`
            // from three times to once - one in multipoint_eval, and the other in sum_frac.
            let ps: Vec<_> = ps.into_iter().collect();
            let f = Self::prod(
                ps.iter()
                    .map(|(x, _)| Poly::new(vec![-x.clone(), T::one()])),
            );
            let df_dx = f.deriv().multipoint_eval(ps.iter().map(|(x, _)| x.clone()));
            let (d, _n) = Poly::sum_frac(ps.into_iter().zip(df_dx).map(|((x, y), m)| {
                (
                    Poly::from(y * m.inv()),
                    Poly::new(vec![-x.clone(), T::one()]),
                )
            }));
            d
        }
        // Transposed version of mul(Rev[f], -) (Tellegen's principle)
        pub fn mul_rev_t(self, rhs: Self) -> Self {
            let shift = self.len().saturating_sub(1);
            let old = rhs.len();
            let mut prod = self * rhs;
            prod.0.truncate(old);
            prod = prod.div_xk(shift);
            prod
        }
        // Transposed version of mul(f, -) (Tellegen's principle)
        pub fn mul_t(mut self, rhs: Self) -> Self {
            self.reverse();
            self.mul_rev_t(rhs)
        }
        pub fn multipoint_eval(&self, ps: impl IntoIterator<Item = T>) -> Vec<T> {
            let ps = ps.into_iter().collect::<Vec<_>>();
            let mut divisors: Vec<_> = ps
                .iter()
                .map(|p| Poly::new(vec![-p.clone(), T::one()]))
                .collect();
            if divisors.is_empty() {
                return vec![];
            }

            let n = divisors.len();
            for i in 0..n - 1 {
                divisors.push(divisors[i << 1].clone() * divisors[i << 1 | 1].clone());
            }

            // Transposed version of $\sum_i c_i/(1-a_i x)$ (Tellegen's principle)
            let mut remainders = vec![Poly::zero(); 2 * n - 1];

            let mut f = self.clone();
            let k = f.len();

            let mut d = std::mem::take(&mut divisors[2 * n - 2]);
            d.reverse();
            d = d.recip_mod_xk(k);
            d.0.resize(k, T::zero());

            f.0.resize(n + k - 1, T::zero());
            f = d.mul_t(f);
            f.0.resize(n, T::zero());
            remainders[2 * n - 2] = f;

            for i in (0..n - 1).rev() {
                let old = std::mem::take(&mut remainders[i + n]);
                let mut l = std::mem::take(&mut divisors[i << 1]);
                let mut r = std::mem::take(&mut divisors[i << 1 | 1]);

                l = l.mul_rev_t(old.clone());
                r = r.mul_rev_t(old);

                remainders[i << 1] = r;
                remainders[i << 1 | 1] = l;
            }
            (0..n).map(|i| remainders[i].coeff(0)).collect()
        }
        pub fn pow_mod_xk(&self, mut exp: u64, k: usize) -> Self {
            let mut res = Self::one().mod_xk(k);
            let mut base = self.clone().mod_xk(k);
            while exp > 0 {
                if exp & 1 == 1 {
                    res *= base.clone();
                    res = res.mod_xk(k);
                }
                base *= base.clone();
                base = base.mod_xk(k);
                exp >>= 1;
            }
            res.mod_xk(k)
        }
        pub fn integrate(&self) -> Self {
            if self.0.is_empty() {
                return self.clone();
            }

            Self(
                std::iter::once(T::zero())
                    .chain((1u32..).zip(&self.0).map(|(i, x)| T::from(i).inv() * x))
                    .collect(),
            )
        }
        pub fn recip_mod_xk(&self, k: usize) -> Self {
            assert!(self.0[0] != T::zero(), "");
            let mut res = Poly::from(self.0[0].inv());
            let mut i = 1;
            let two = Poly::from(T::from(2u32));
            while i < k {
                i <<= 1;
                let r = self.clone().mod_xk(i);
                res *= two.clone() - &(res.clone() * r);
                res = res.mod_xk(i);
            }
            res.mod_xk(k)
        }
        pub fn ln_mod_xk(&self, k: usize) -> Self {
            assert!(self.0[0] != T::zero(), "");

            let mut deriv_ln = self.deriv();
            deriv_ln *= self.clone().recip_mod_xk(k);
            deriv_ln.mod_xk(k.saturating_sub(1)).integrate()
        }
        pub fn exp_mod_xk(&self, k: usize) -> Self {
            assert!(self.0.is_empty() || self.0[0] == T::zero(), "");
            let one = Poly::from(T::from(1u32));
            let mut res = one.clone();
            let mut i = 1;
            while i < k {
                i <<= 1;
                res *= one.clone() + &self.clone().mod_xk(i) - res.ln_mod_xk(i);
                res = res.mod_xk(i);
            }

            res = res.mod_xk(k);
            res
        }
        // sqrt (1 + x f(x)) mod x^k
        pub fn sqrt_1p_mx_mod_xk(&self, k: usize) -> Self {
            let mut f = self.clone();
            f = f.mul_xk(1);
            f += &Self::one();

            let mut res = Self::one();
            let mut i = 1;
            let inv2 = T::from(2u32).inv();
            while i < k {
                i <<= 1;
                let mut p = f.clone().mod_xk(i);
                p *= res.recip_mod_xk(i);
                res += &p.mod_xk(i);
                res *= inv2.clone();
                res = res.mod_xk(i);
            }
            res.mod_xk(k)
        }
        pub fn taylor_shift(&self, k: usize, ifc: &[T]) -> Self {
            todo!()
        }
        // Bostan-Mori, O(L log L log N)
        pub fn nth_of_frac(numer: Self, denom: Self, mut n: u64) -> T {
            let mut p = numer.mod_xk(n as usize + 1);
            let mut q = denom.mod_xk(n as usize + 1);
            while n > 0 {
                let mut q_neg = q.clone();
                for i in (1..q_neg.0.len()).step_by(2) {
                    q_neg.0[i] = -q_neg.0[i].clone();
                }

                let u = p * q_neg.clone();
                let v = q * q_neg;

                p = u.0.into_iter().skip((n % 2) as usize).step_by(2).collect();
                q = v.0.into_iter().step_by(2).collect();

                n /= 2;
            }
            p.coeff(0) / q.coeff(0)
        }
        fn pad_chunks(&mut self, w_src: usize, w_dest: usize) {
            // Helper for kronecker substitution
            assert!(w_src <= w_dest);

            let mut res = Poly::new(vec![]);
            for r in self.0.chunks(w_src) {
                res.0.extend(r.iter().cloned());
                res.0.extend((0..w_dest - r.len()).map(|_| T::zero()))
            }
            *self = res
        }
        // [x^n] g(x)/(1-y f(x)) mod y^{n+1}
        pub fn power_proj(f: &Self, g: &Self, n: usize) -> Self {
            if f.0.is_empty() || g.0.is_empty() || n == 0 {
                return Poly::zero();
            };

            let f0 = f.0[0].clone();
            if f0 != T::zero() {
                unimplemented!("f(0) != 0. Do shift yourself")
            }

            let mut nc = n;
            let np = n + 1;
            let mut w = 2;
            let mut p = Poly::new(vec![T::zero(); np * w]);
            let mut q = Poly::new(vec![T::zero(); np * w]);
            for i in 0..np.min(g.0.len()) {
                p.0[i * w + 0] = g.0[i].clone();
            }
            q.0[0 * w + 0] = T::one();
            for i in 0..np.min(f.0.len()) {
                q.0[i * w + 1] = -f.0[i].clone();
            }
            while nc > 0 {
                let w_prev = w;
                w = w_prev * 2 - 1;
                p.pad_chunks(w_prev, w);
                q.pad_chunks(w_prev, w);

                let mut q_nx = q.clone();
                for r in q_nx.0.chunks_mut(w).skip(1).step_by(2) {
                    for x in r {
                        *x = -x.clone();
                    }
                }

                let u = p * q_nx.clone();
                let v = q * q_nx;

                p =
                    u.0.chunks(w)
                        .skip(nc % 2)
                        .step_by(2)
                        .take(nc / 2 + 1)
                        .flatten()
                        .cloned()
                        .collect();
                q =
                    v.0.chunks(w)
                        .step_by(2)
                        .take(nc / 2 + 1)
                        .flatten()
                        .cloned()
                        .collect();

                nc /= 2;
            }

            (p.mod_xk(n + 1) * q.mod_xk(n + 1).recip_mod_xk(n + 1)).mod_xk(n + 1)
        }
        pub fn comp_inv_mod_xk(&self, k: usize, fc: &[T], ifc: &[T]) -> Self {
            // Power projection & Lagrange inv.
            assert!(self.0.len() >= 2 && self.0[1] != T::zero());
            assert!(self.0[0] == T::zero(), "Check algebraic generating series");
            if k <= 1 {
                return Poly::zero();
            } else if k == 2 {
                return Poly::new(vec![T::zero(), self.0[1].inv()]);
            }

            let n = k - 1;
            let mut p = Poly::power_proj(self, &Poly::one(), n);
            p.0.resize(n + 1, T::zero());

            for i in 1..n + 1 {
                p.0[i] *= ifc[i].clone() * fc[i - 1].clone();
            }
            p.reverse();
            p *= p.0[0].inv();

            p = p.ln_mod_xk(n);
            p *= -T::from(n as u32).inv();
            p = p.exp_mod_xk(n);
            p *= self.0[1].inv();
            p.mul_xk(1)
        }
        pub fn comp_mod_xk(&self, other: &Self, k: usize) -> Self {
            // Kinoshita-Li composition
            todo!()
        }
    }
    impl<T: NTTSpec + From<u32> + Field> Poly<T> {
        pub fn deriv(&self) -> Self {
            Self(
                ((1u32..).zip(&self.0[1..]))
                    .map(|(i, x)| T::from(i) * x)
                    .collect(),
            )
        }
    }
    impl<T: SemiRing> From<T> for Poly<T> {
        fn from(c: T) -> Self {
            Self(vec![c])
        }
    }
    impl<T: SemiRing> MulAssign<&'_ T> for Poly<T> {
        fn mul_assign(&mut self, rhs: &T) {
            self.0.iter_mut().for_each(|c| c.mul_assign(rhs.clone()));
        }
    }
    impl<T: SemiRing> MulAssign<T> for Poly<T> {
        fn mul_assign(&mut self, rhs: T) {
            self.mul_assign(&rhs);
        }
    }
    impl<T: SemiRing> AddAssign<&'_ Self> for Poly<T> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0.resize_with(self.len().max(rhs.len()), T::zero);
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.add_assign(b));
        }
    }
    impl<T: SemiRing> Add<&'_ Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, rhs: &Self) -> Self {
            self += rhs;
            self
        }
    }
    impl<T: SemiRing> Add<Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self {
            self += &rhs;
            self
        }
    }
    impl<T: SemiRing> SubAssign<&'_ Self> for Poly<T> {
        fn sub_assign(&mut self, rhs: &Self) {
            self.0.resize_with(self.len().max(rhs.len()), T::zero);
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.sub_assign(b));
        }
    }
    impl<T: SemiRing> Sub<&'_ Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, rhs: &Self) -> Self {
            self -= rhs;
            self
        }
    }
    impl<T: SemiRing> Sub<Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, rhs: Self) -> Self {
            self -= &rhs;
            self
        }
    }
    impl<T: NTTSpec + Field> MulAssign<Self> for Poly<T> {
        fn mul_assign(&mut self, mut rhs: Self) {
            self.pop_zeros();
            rhs.pop_zeros();
            if self.len() == 0 || rhs.len() == 0 {
                self.0.clear();
                return;
            }

            let mut lhs = std::mem::take(self);
            let n = lhs.len() + rhs.len() - 1;

            if lhs.len() < rhs.len() {
                std::mem::swap(&mut lhs, &mut rhs);
            }
            if rhs.len() <= 20 {
                self.0 = vec![T::zero(); n];
                for (i, x) in lhs.0.into_iter().enumerate() {
                    for j in 0..rhs.len() {
                        self.0[i + j] += rhs.0[j].clone() * &x;
                    }
                }
                return;
            }

            let n_padded = n.next_power_of_two();

            lhs.0.resize(n_padded, T::zero());
            rhs.0.resize(n_padded, T::zero());

            if let Some(proot) = T::try_nth_proot(n_padded as u32) {
                ntt::run(proot.clone(), &mut lhs.0);
                ntt::run(proot.clone(), &mut rhs.0);
                lhs.0.iter_mut().zip(&rhs.0).for_each(|(a, b)| *a *= b);
                ntt::run(proot.inv(), &mut lhs.0);
                let n_inv = T::from(n_padded as u32).inv();
                lhs.0.iter_mut().for_each(|c| c.mul_assign(&n_inv));

                lhs.0.truncate(n);
                *self = lhs;
            } else {
                use ntt::sample::p104857601 as p0;
                use ntt::sample::p167772161 as p1;
                use ntt::sample::p998244353 as p2;

                let into_u64 = |x: &T| -> u64 {
                    let x: u32 = x.clone().into();
                    x as u64
                };

                let lhs_u64 = || lhs.0.iter().map(into_u64);
                let rhs_u64 = || rhs.0.iter().map(into_u64);

                let h0 = Poly::from_iter(lhs_u64().map(p0::M::from))
                    * Poly::from_iter(rhs_u64().map(p0::M::from));
                let h1 = Poly::from_iter(lhs_u64().map(p1::M::from))
                    * Poly::from_iter(rhs_u64().map(p1::M::from));
                let h2 = Poly::from_iter(lhs_u64().map(p2::M::from))
                    * Poly::from_iter(rhs_u64().map(p2::M::from));

                let (q, ms) = {
                    thread_local! {
                        static COEFF: OnceCell<(u128, [u128; 3])> =
                            OnceCell::new();
                    }
                    COEFF.with(|coeff| {
                        *coeff.get_or_init(|| {
                            let q = p0::P as u128 * p1::P as u128 * p2::P as u128;
                            let (qs, rs) = crt3_coeff_u64::<p0::M, p1::M, p2::M, T>([
                                p0::P as u64,
                                p1::P as u64,
                                p2::P as u64,
                            ]);
                            let ms = [
                                qs[0] as u128 * u32::from(rs.0) as u128,
                                qs[1] as u128 * u32::from(rs.1) as u128,
                                qs[2] as u128 * u32::from(rs.2) as u128,
                            ];
                            (q, ms)
                        })
                    })
                };

                let m = h0.len().max(h1.len()).max(h2.len());
                self.0 = (h0.0.into_iter().chain(std::iter::once(SemiRing::zero())))
                    .zip(h1.0.into_iter().chain(std::iter::once(SemiRing::zero())))
                    .zip(h2.0.into_iter().chain(std::iter::once(SemiRing::zero())))
                    .take(m)
                    .map(|((x0, x1), x2)| {
                        let a = ms[0] as u128 * u32::from(x0) as u128
                            + ms[1] as u128 * u32::from(x1) as u128
                            + ms[2] as u128 * u32::from(x2) as u128;
                        T::from(a % q)
                    })
                    .collect();
            }
        }
    }
    impl<T: NTTSpec + Field> Mul<Self> for Poly<T> {
        type Output = Self;
        fn mul(mut self, rhs: Self) -> Self {
            self *= rhs;
            self
        }
    }
    impl<T: NTTSpec + Field> DivAssign<Self> for Poly<T> {
        fn div_assign(&mut self, mut rhs: Self) {
            assert!(!rhs.is_zero());
            self.pop_zeros();
            rhs.pop_zeros();
            if self.degree() < rhs.degree() {
                self.0.clear();
                return;
            }

            let n = self.degree();
            let m = rhs.degree();
            let l = n - m + 1;

            self.reverse();
            *self = std::mem::take(self).mod_xk(l);
            rhs.reverse();
            rhs = rhs.mod_xk(l);

            *self *= rhs.recip_mod_xk(l);
            self.0.resize(l, T::zero());
            self.reverse();
        }
    }
    impl<T: NTTSpec + Field> RemAssign<Self> for Poly<T> {
        fn rem_assign(&mut self, rhs: Self) {
            let mut q = self.clone();
            q /= rhs.clone();
            q *= rhs;
            *self -= &q;
            self.pop_zeros();
        }
    }
    impl<T: CommRing> Neg for &Poly<T> {
        type Output = Poly<T>;
        fn neg(self) -> Poly<T> {
            Poly(self.0.iter().map(|c| -c.clone()).collect())
        }
    }
    impl<T: CommRing> Neg for Poly<T> {
        type Output = Self;
        fn neg(self) -> Self {
            -&self
        }
    }
}

pub mod linear_recurrence {
    use crate::algebra::Field;

    use super::algebra::CommRing;
    use super::ntt::NTTSpec;
    use super::poly::Poly;

    pub fn berlekamp_massey<T: CommRing>(_seq: &[T]) -> Vec<T> {
        unimplemented!()
    }

    pub fn next<T: CommRing + Copy>(recurrence: &[T], init: &[T]) -> T {
        let l = recurrence.len();
        let n = init.len();
        assert!(n >= l);
        let mut value = recurrence[0] * init[n - 1];
        for i in 1..l {
            value += recurrence[i] * init[n - 1 - i];
        }
        value
    }

    pub fn nth_by_ntt<T: NTTSpec + Field + Clone>(recurrence: &[T], init: &[T], n: u64) -> T {
        let l = recurrence.len();
        assert!(l >= 1 && l == init.len());

        let mut q = Vec::with_capacity(l + 1);
        q.push(T::one());
        for c in recurrence.iter().cloned() {
            q.push(-c);
        }
        let q = Poly::new(q);
        let p = (Poly::new(init.to_vec()) * q.clone()).mod_xk(l);

        Poly::nth_of_frac(p, q, n)
    }
}

use algebra::SemiRing;
use poly::Poly;

use crate::algebra::Field;

pub mod p1000000007 {
    pub const P: u32 = 1000000007;
    pub const GEN: u32 = 3;
    pub type M = super::mint_mont::M32<P>;
    impl super::ntt::NTTSpec for M {
        fn try_nth_proot(_n: u32) -> Option<Self> {
            None
        }
    }
}

type M = ntt::sample::p998244353::M;
// type M = p1000000007::M;
