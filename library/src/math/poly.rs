// TODO: cast modint from signed integers
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
            $(impl Unsigned for $t {
                fn one() -> Self {
                    1
                }
            })+
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
            $(impl SemiRing for $t {
                fn one() -> Self {
                    1
                }
            })+
        };
    }
    macro_rules! impl_commring {
        ($($t:ty)+) => { $(impl CommRing for $t {})+ };
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
        fn reduce(value: Self::D) -> Self::U;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct MInt<M: ModSpec>(M::U);

    impl<M: ModSpec> MInt<M> {
        pub const MODULUS: M::U = M::MODULUS;
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
                fn reduce(x: Self::D) -> Self::U {
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
            self.0 = M::reduce(M::to_double(self.0) * M::to_double(rhs.0));
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
                    M::reduce(M::to_double(x.0))
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

pub mod conv {
    use super::algebra::*;

    pub struct PreCalc<T> {
        rs: Vec<T>,
        irs: Vec<T>,
    }
    impl<T: Field> PreCalc<T> {
        fn new(n_log: usize, proot: T) -> Self {
            let mut pow = vec![T::one(); n_log + 1];
            let mut ipow = vec![T::one(); n_log + 1];
            pow[n_log] = proot.clone();
            ipow[n_log] = proot.clone().inv();
            for i in (0..n_log).rev() {
                pow[i] = pow[i + 1].clone() * pow[i + 1].clone();
                ipow[i] = ipow[i + 1].clone() * ipow[i + 1].clone();
            }

            let mut rs = vec![T::one(); n_log];
            let mut irs = vec![T::one(); n_log];
            let mut p = T::one();
            let mut ip = T::one();
            for i in 0..n_log - 1 {
                rs[i] = pow[i + 2].clone() * p.clone();
                irs[i] = ipow[i + 2].clone() * ip.clone();
                p *= ipow[i + 2].clone();
                ip *= pow[i + 2].clone();
            }
            Self { rs, irs }
        }
    }

    pub fn permute_bitrev<T>(xs: &mut [T]) {
        let n = xs.len();
        if n == 0 {
            return;
        }
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        for i in 0..n as u32 {
            let rev = i.reverse_bits() >> u32::BITS - n_log2;
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }
    }
    pub fn bf2<T: CommRing>(x1: &mut T, x2: &mut T, w: T) {
        let y1 = x1.clone();
        let y2 = x2.clone() * w;
        *x1 = y1.clone() + y2.clone();
        *x2 = y1 - y2;
    }
    pub fn bf2_t<T: CommRing>(x1: &mut T, x2: &mut T, w: T) {
        let y1 = x1.clone();
        let y2 = x2.clone();
        *x1 = y1.clone() + y2.clone();
        *x2 = (y1 - y2) * w;
    }
    fn chunks2<T>(xs: &mut [T], w: usize) -> impl Iterator<Item = (&mut [T], &mut [T])> {
        xs.chunks_exact_mut(w * 2).map(move |t| t.split_at_mut(w))
    }
    pub fn ntt_radix2<T: Field>(cx: &PreCalc<T>, xs: &mut [T]) {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log = u32::BITS - (n as u32).leading_zeros() - 1;
        for e in (0..n_log).rev() {
            let mut w = T::one();
            for (it, (t0, t1)) in chunks2(xs, 1 << e).enumerate() {
                (t0.iter_mut().zip(t1)).for_each(|(x0, x1)| bf2(x0, x1, w.clone()));
                w *= cx.rs[it.trailing_ones() as usize].clone();
            }
        }
    }
    pub fn intt_radix2<T: Field>(cx: &PreCalc<T>, xs: &mut [T]) {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log = u32::BITS - (n as u32).leading_zeros() - 1;
        for e in 0..n_log {
            let mut w = T::one();
            for (it, (t0, t1)) in chunks2(xs, 1 << e).enumerate() {
                (t0.iter_mut().zip(t1)).for_each(|(x0, x1)| bf2_t(x0, x1, w.clone()));
                w *= cx.irs[it.trailing_ones() as usize].clone();
            }
        }
    }

    const P_GENS_32: [[u32; 2]; 4] = [
        [998244353, 3],
        [167772161, 3],
        [104857601, 3],
        [13631489, 15],
    ];
    pub const fn try_gen_32(p: u32) -> Option<u32> {
        let mut i = 0;
        while i < P_GENS_32.len() {
            if P_GENS_32[i][0] == p {
                return Some(P_GENS_32[i][1]);
            }
            i += 1;
        }
        None
    }
    pub fn try_proot_m32<const P: u32>(n: usize) -> Option<crate::mint_mont::M32<P>> {
        let g = try_gen_32(P)?;
        if (P - 1) % n as u32 != 0 {
            return None;
        }
        Some(crate::mint_mont::M32::from(g).pow((P - 1) / n as u32))
    }

    fn conv_naive<T: SemiRing>(mut lhs: Vec<T>, mut rhs: Vec<T>) -> Vec<T> {
        if lhs.len() == 0 || rhs.len() == 0 {
            return lhs;
        }
        if lhs.len() > rhs.len() {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        let n = lhs.len() + rhs.len() - 1;
        let mut res = vec![T::zero(); n];
        for i in 0..lhs.len() {
            for j in 0..rhs.len() {
                res[i + j] += lhs[i].clone() * rhs[j].clone();
            }
        }
        res
    }

    fn conv_with_proot<T: Field + From<u32>>(
        mut lhs: Vec<T>,
        mut rhs: Vec<T>,
        gen_proot: fn(usize) -> Option<T>,
    ) -> Result<Vec<T>, (Vec<T>, Vec<T>)> {
        if lhs.len().min(rhs.len()) <= 20 {
            return Ok(conv_naive(lhs, rhs));
        }

        let n = lhs.len() + rhs.len() - 1;
        let n_pad = n.next_power_of_two();
        let proot = match gen_proot(n_pad) {
            Some(proot) => proot,
            None => return Err((lhs, rhs)),
        };
        let n_log = (u32::BITS - (n_pad as u32).leading_zeros() - 1) as usize;

        let cx = PreCalc::new(n_log, proot.clone());

        lhs.resize(n_pad, T::zero());
        rhs.resize(n_pad, T::zero());
        ntt_radix2(&cx, &mut lhs);
        ntt_radix2(&cx, &mut rhs);
        lhs.iter_mut().zip(&rhs).for_each(|(x, y)| *x *= y);
        intt_radix2(&cx, &mut lhs);
        let n_inv = T::from(n_pad as u32).inv();
        lhs.iter_mut().for_each(|x| *x *= n_inv.clone());
        lhs.truncate(n);
        Ok(lhs)
    }
    fn conv_with_crt_32<T: SemiRing + Into<u32> + From<u128>>(lhs: Vec<T>, rhs: Vec<T>) -> Vec<T> {
        const fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
            let mut res = 1;
            while exp > 0 {
                if exp % 2 == 1 {
                    res = res * base % m;
                }
                base = base * base % m;
                exp >>= 1;
            }
            res
        }
        const fn crt3_coeff_u32(ps: [u32; 3]) -> (u128, [u128; 3]) {
            let ps = [ps[0] as u64, ps[1] as u64, ps[2] as u64];
            let q = ps[0] as u128 * ps[1] as u128 * ps[2] as u128;
            let qs = [ps[1] * ps[2], ps[0] * ps[2], ps[0] * ps[1]];
            let rs = [
                mod_pow(qs[0] % ps[0], ps[0] - 2, ps[0]),
                mod_pow(qs[1] % ps[1], ps[1] - 2, ps[1]),
                mod_pow(qs[2] % ps[2], ps[2] - 2, ps[2]),
            ];
            let ms = [
                qs[0] as u128 * rs[0] as u128,
                qs[1] as u128 * rs[1] as u128,
                qs[2] as u128 * rs[2] as u128,
            ];
            (q, ms)
        }
        use crate::conv;
        type M<const P: u32> = crate::mint_mont::M32<P>;
        const P0: u32 = conv::P_GENS_32[0][0];
        const P1: u32 = conv::P_GENS_32[1][0];
        const P2: u32 = conv::P_GENS_32[2][0];
        const Q: u128 = crt3_coeff_u32([P0, P1, P2]).0;
        const MS: [u128; 3] = crt3_coeff_u32([P0, P1, P2]).1;

        fn forward<T: Into<u32> + Clone, const P: u32>(lhs: &[T], rhs: &[T]) -> Vec<M<P>> {
            let c = |x: &T| <M<P> as From<u32>>::from(x.clone().into());
            let lhs = lhs.iter().map(c).collect();
            let rhs = rhs.iter().map(c).collect();
            conv_with_proot(lhs, rhs, try_proot_m32).unwrap()
        }
        let h0: Vec<M<P0>> = forward(&lhs, &rhs);
        let h1: Vec<M<P1>> = forward(&lhs, &rhs);
        let h2: Vec<M<P2>> = forward(&lhs, &rhs);
        (h0.into_iter().zip(h1).zip(h2))
            .map(|((x0, x1), x2)| {
                let a = MS[0] * u32::from(x0) as u128
                    + MS[1] * u32::from(x1) as u128
                    + MS[2] * u32::from(x2) as u128;
                T::from(a % Q)
            })
            .collect()
    }

    pub trait Conv: SemiRing {
        fn conv(lhs: Vec<Self>, rhs: Vec<Self>) -> Vec<Self>;
    }
    impl<const P: u32> Conv for crate::mint_mont::M32<P> {
        fn conv(lhs: Vec<Self>, rhs: Vec<Self>) -> Vec<Self> {
            match conv_with_proot(lhs, rhs, try_proot_m32) {
                Ok(res) => res,
                Err((lhs, rhs)) => conv_with_crt_32(lhs, rhs),
            }
        }
    }
}

pub mod comb {
    pub struct Comb<M> {
        pub fc: Vec<M>,
        pub ifc: Vec<M>,
    }
    impl<T: crate::algebra::Field + From<u32>> Comb<T> {
        pub fn new(bound: usize) -> Self {
            assert!(bound >= 1);

            let mut fc = vec![T::one()];
            for i in 1..=bound {
                fc.push(fc[i as usize - 1].clone() * T::from(i as u32));
            }

            let mut ifc = vec![T::one(); bound as usize + 1];
            ifc[bound as usize] = fc[bound as usize].inv();
            for i in (2..=bound).rev() {
                ifc[i as usize - 1] = ifc[i as usize].clone() * T::from(i as u32);
            }

            Self { fc, ifc }
        }
        pub fn inv(&self, i: usize) -> T {
            self.ifc[i].clone() * self.fc[i - 1].clone()
        }
        pub fn binom(&self, n: usize, k: usize) -> T {
            self.fc[n].clone() * self.ifc[k].clone() * self.ifc[n - k].clone()
        }
    }
}

pub mod poly {
    use crate::algebra::*;
    use crate::comb::Comb;
    use crate::conv::Conv;
    use std::collections::VecDeque;
    use std::ops::*;

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Poly<T>(pub Vec<T>);

    impl<T: CommRing> Poly<T> {
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
        pub fn clone_mod_xk(&self, k: usize) -> Self {
            Self(self.0[..self.0.len().min(k)].to_vec())
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
    impl<T: From<u32> + Field> Poly<T> {
        pub fn deriv(&self) -> Self {
            Self(
                ((1u32..).zip(&self.0[1..]))
                    .map(|(i, x)| T::from(i) * x)
                    .collect(),
            )
        }
    }
    impl<T: CommRing> From<T> for Poly<T> {
        fn from(c: T) -> Self {
            Self(vec![c])
        }
    }
    impl<T: CommRing> MulAssign<T> for Poly<T> {
        fn mul_assign(&mut self, rhs: T) {
            self.0.iter_mut().for_each(|c| c.mul_assign(rhs.clone()));
        }
    }
    impl<T: CommRing> Mul<T> for Poly<T> {
        type Output = Self;
        fn mul(mut self, rhs: T) -> Self::Output {
            self *= rhs;
            self
        }
    }
    impl<T: CommRing> AddAssign<&'_ Self> for Poly<T> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0.resize(self.len().max(rhs.len()), T::zero());
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.add_assign(b));
        }
    }
    impl<T: CommRing> Add<&'_ Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, rhs: &Self) -> Self {
            self += rhs;
            self
        }
    }
    impl<T: CommRing> Add<Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, mut rhs: Self) -> Self {
            if self.len() < rhs.len() {
                std::mem::swap(&mut self, &mut rhs);
            }
            self += &rhs;
            self
        }
    }
    impl<T: CommRing> SubAssign<&'_ Self> for Poly<T> {
        fn sub_assign(&mut self, rhs: &Self) {
            self.0.resize(self.len().max(rhs.len()), T::zero());
            self.0.iter_mut().zip(&rhs.0).for_each(|(x, y)| *x -= y);
        }
    }
    impl<T: CommRing> Sub<&'_ Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, rhs: &Self) -> Self {
            self -= rhs;
            self
        }
    }
    impl<T: CommRing> Sub<Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, mut rhs: Self) -> Self {
            if self.len() >= rhs.len() {
                self.0.iter_mut().zip(&rhs.0).for_each(|(x, y)| *x -= y);
                self
            } else {
                std::mem::swap(&mut self, &mut rhs);
                self.0.iter_mut().zip(&rhs.0).for_each(|(x, y)| *x -= y);
                -self
            }
        }
    }
    impl<T: Conv + Field> Mul<Self> for Poly<T> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(Conv::conv(self.0, rhs.0))
        }
    }
    impl<T: Conv + Field> MulAssign<Self> for Poly<T> {
        fn mul_assign(&mut self, rhs: Self) {
            let lhs = std::mem::take(self);
            *self = lhs * rhs;
        }
    }
    impl<T: Conv + Field + From<u32>> DivAssign<Self> for Poly<T> {
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

            *self *= rhs.inv_mod_xk(l);
            self.0.resize(l, T::zero());
            self.reverse();
        }
    }
    impl<T: Conv + Field + From<u32>> RemAssign<Self> for Poly<T> {
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
    impl<T: Conv + Field + From<u32>> Poly<T> {
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
            d = d.inv_mod_xk(k);
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
        pub fn integrate(&self, cx: &Comb<T>) -> Self {
            std::iter::once(T::zero())
                .chain(self.0.iter().enumerate().map(|(i, x)| cx.inv(i + 1) * x))
                .collect()
        }
        pub fn inv_mod_xk(&self, k: usize) -> Self {
            assert!(self.0[0] != T::zero(), "");
            let mut res = Poly::from(self.0[0].inv());
            let mut i = 1;
            let two = Poly::from(T::from(2u32));
            while i < k {
                i = (i << 1).min(k);
                res *= two.clone() - res.clone() * self.clone_mod_xk(i);
                res = res.mod_xk(i);
            }
            res.mod_xk(k)
        }
        pub fn ln_mod_xk(&self, cx: &Comb<T>, k: usize) -> Self {
            assert!(self.0[0] != T::zero(), "");
            let mut q = self.deriv();
            q *= self.clone().inv_mod_xk(k);
            q.mod_xk(k.saturating_sub(1)).integrate(cx)
        }
        pub fn exp_mod_xk(&self, cx: &Comb<T>, k: usize) -> Self {
            assert!(self.0.is_empty() || self.0[0] == T::zero(), "");
            let mut f = Poly::one();
            let mut i = 1;
            while i < k {
                i = (i << 1).min(k);
                let inv_f = f.clone().inv_mod_xk(i - 1);
                f *= Poly::one() + self.clone_mod_xk(i)
                    - (f.deriv() * inv_f.clone()).mod_xk(i - 1).integrate(cx);
                f = f.mod_xk(i);
            }
            f.mod_xk(k)
        }
        pub fn sqrt_1p_mx_mod_xk(&self, k: usize) -> Self {
            // sqrt (1 + x f(x)) mod x^k
            let mut f = self.clone();
            f = f.mul_xk(1);
            f += &Self::one();

            let mut res = Self::one();
            let mut i = 1;
            let inv2 = T::from(2u32).inv();
            while i < k {
                i = (i << 1).min(k);
                let q = res.inv_mod_xk(i);
                res = (res + (f.clone_mod_xk(i) * q).mod_xk(i)) * inv2.clone();
            }
            res.mod_xk(k)
        }
        pub fn taylor_shift(&self, cx: &Comb<T>, k: usize) -> Self {
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
        // Helps kronecker substitution
        fn pad_chunks(&mut self, w_src: usize, w_dest: usize) {
            assert!(w_src <= w_dest);
            let mut res = Poly::new(vec![]);
            for r in self.0.chunks(w_src) {
                res.0.extend(r.iter().cloned());
                res.0.extend((0..w_dest - r.len()).map(|_| T::zero()))
            }
            *self = res
        }
        // Kinoshita-Li power projection
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
            (p.mod_xk(n + 1) * q.mod_xk(n + 1).inv_mod_xk(n + 1)).mod_xk(n + 1)
        }
        pub fn comp_inv_mod_xk(&self, cx: &Comb<T>, k: usize) -> Self {
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
                p.0[i] *= cx.inv(i);
            }
            p.reverse();
            p *= p.0[0].inv();

            p = (p.ln_mod_xk(cx, n) * (-T::from(n as u32).inv())).exp_mod_xk(cx, n);
            (p * self.0[1].inv()).mul_xk(1)
        }
        // Kinoshita-Li composition
        pub fn comp_mod_xk(&self, other: &Self, k: usize) -> Self {
            todo!()
        }
    }
}

pub mod linear_rec {
    use super::poly::Poly;
    use crate::algebra::{CommRing, Field};
    use crate::conv::Conv;

    pub fn berlekamp_massey<T: CommRing>(_seq: &[T]) -> Vec<T> {
        unimplemented!()
    }

    pub fn next<T: CommRing>(recurrence: &[T], init: &[T]) -> T {
        let l = recurrence.len();
        let n = init.len();
        assert!(n >= l);
        let mut value = recurrence[0].clone() * init[n - 1].clone();
        for i in 1..l {
            value += recurrence[i].clone() * init[n - 1 - i].clone();
        }
        value
    }

    pub fn nth_by_ntt<T: Conv + Field + From<u32>>(recurrence: &[T], init: &[T], n: u64) -> T {
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

use poly::Poly;

type M = mint_mont::M32<998244353>;
// type M = mint_mont::M32<1000000007>;
