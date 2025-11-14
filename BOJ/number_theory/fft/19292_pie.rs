// We love genfunc
//
// ## Solution 1
//
// Special case: single color, ans = L.
//
// Insert colors one by one.
// fix color of the first element (time complexity multiplies by N), and insert that color first.
//
// we should keep track of statistic for equality on adjacent colors.
// state: (n_non_eq_holes, n_eq_holes, lengths) ~ $x^h y^k t_1^{c_1} ... t_n^{c_n}$
//
// single-component: $f(t) = \sum_l l t^l$
// squared: $g(t) = \sum_l l^2 t^l$
//
//
// - initial state: $h = 0$.
// first we determine the position of the block containing the first position (g(t)),
// and we insert the list of remaining blocks.
// $p_1(x,y) = [t_1^{c_1}] (y g(t_1))/(1-y f(t_1))$
//
//
//
// -transition:
// for each hole, insert subpartittion of blocks with new color (allowing empty partition).
// non-empty single partition: $f(t_i)/(1-y f(t_i)) $
//
// single non-equal outer hole: $b(x,y,t_i) = x + x^2 f(t_i)/(1-y f(t_i))$
// single equal outer hole: $c(x,y,t_i) = y + x^2 f(t_i)/(1-y f(t_i))$
//
// $x^h y^k$ transforms to: $[t_i^{c_i}] (b(x,y,t_i))^{h} (c(x,y,t_i))^k $. Therefore
// $p_i(x,y) = [t_i^{c_i}] p_{i-1}(b(x,y,t_i), c(x,y,t_i))$
//
//
// - final state: $k=0$.  $Ans = sum_h [x^h y^0] p_n(x,y)$
//
// - summary
// $f(t) = \sum_l l t^l = (tD_t) (1/(1-t)) = t/(1-t)^2$
// $g(t) = \sum_l l^2 t^l = (t D_t)^2 (1/(1-t)) = (t+t^2)/(1-t)^3$
// $h(x,y,t) = x^2 f(t)/(1-y f(t)) = x^2 \sum_c t^c sum_{k \le c-1} y^k binom(k+c, 2k+1)$
// $b(x,y,t) = x + h(x,y,t)$
// $c(x,y,t) = y + h(x,y,t)$
// $p_1(x,y) = [t_1^{c_1}] (y g(t_1))/(1-y f(t_1)) = sum_{k} y^{k+1} ([k\le c - 1]binom(k+c+1, 2k+2) +[k \le c - 2] binom(k+c, 2k+2))$
// $p_i(x,y) = [t_i^{c_i}]p_{i-1}(b(x,y,t_i), c(x,y,t_i))$
// $[\text{Ans for a single color}] = sum_h [x^h y^0] p_n(x,y) = p_n(1,0)$
//
//
// too slow
//
//
//
// ## Solution 2
//
// As in the first solution, we fix the position of the first color. The building blocks remain the same.
// $$f(t) = \sum_l l t^l = (tD_t) (1/(1-t)) = t/(1-t)^2$$
// $$g(t) = \sum_l l^2 t^l = (t D_t)^2 (1/(1-t)) = (t+t^2)/(1-t)^3$$
//
// We construct a linear partition which forbids color 1 at both ends, and seal it into a circular configuration with $g(t_1)$.
// Using the approach of (Atcoder FPS 24 - F editorial)[https://atcoder.jp/contests/fps-24/editorial/14463],
// we encode the last color as the index variable of a sequence vector.
// $$S = A_1 + ... + A_n$$
// $$A_i = f(t_i) (1 - \delta_{i1} + S - A_i)$$
//
// Define $h_i \equiv h(t_i) := f(t_i)/(1+f(t_i))$, $H = \sum_i h_i$. Then
// $$A_i = h_i (1-\delta_{i1} + S)$$
// $$S = (H - h_1)/(1-H)$$
// $$[\text{Ans for a single color}] = [t_1^{c_1}...t_n^{c_n}] g_1 (S - A_1)
// = [t_1^{c_1}...t_n^{c_n}] g_1(1-h_1)^2/(1-H)
//
// Apply recursion with $H = h_1 + E_2$, $E_2 = h_2 + E_3$, ... , $E_{n+1}=0$.
// $$[t_1^{c_1}...t_n^{c_n}] g_1(1-h_1)^2/(1-H)
// = [t_2^{c_2}...t_n^{c_n}] \sum_k [t_1^{c_1}] g_1(1-h_1)^2 h_1^k (1-E_2)^{-k-1}
// \equiv  \sum_k s^{(c_1)}_{k} [t_2^{c_2}...t_n^{c_n}] E_2^k$$
// $$[t_2^{c_2}...t_n^{c_n}] E_2^i
// =  \sum_k {i \choose k} [t_2^{c_2}] h_2^{i-k} [t_3^{c_3}...t_n^{c_n}] E_3^k
// \equiv \sum_k  r^{(c_2)}_{ik} [t_3^{c_3}...t_n^{c_n}] E_3^k$$
// $$...$$
// $$E_n^k = \delta_{k0}$$
//
// The result is a matrix product, with dimension bounded by $c_1$ (the max exponent is $c_1-1$)
// $$s^{(c_1)} \cdot r^{(c_2)} ... r^{(c_n)} e_0$$
//
// Observation: r^{(c)} and r^{(d)} commutes.
// Thus if the matrix $r^{(c)} is invertible for $c>0$, Then
// $$Ans = \sum_i s^{(c_i)} (r^{(c_i)})^{-1} \cdot r^{(c_1)} ... r^{(c_n)} e_0$$
//
// r is not really an invertible matrix, but it can be interpreted as polynomial multiplication.
// So we can do inversion in the restricted sense.
//
// The remaining step is to compute $[t^c] h(t)^k$ and $[t^c] g(t) (1-h(t)) h(t)^k$.
//  Transform it to $p(t,y) = 1/(1-y h(t))$ and $q(t,y)= g(t) (1-h(t))/(1-y h(t)) = g(t) (1-h(t)) p(t,y)$.
//  $$h(t) = f(t)/(1+f(t)) = t/(1-t+t^2)$$
//  $$(1- y h(t)) p(t,y) = 1$$
//  $$(1-t+t^2 - yt) p(t,y) = 1-t+t^2$$
//  $$(1-t+t^2 - yt) q(t,y) = g(t) (1-h(t)) (1-t+t^2)$$
//  $$(1-t+t^2 - yt)(1-t) q(t,y) = t+t^2$$
//  Translate this into a O(1) recurrence.
//
//
// By the way, $h = \sum_{k \ge 1} (-1)^{k-1} f^k$ has a pretty nice combinatorial interpretation:
// It represents partition of a monochromatic block, with a PIE factor -1 for each pair of adjacent blocks of the same color.
// Then $List[H]=1/(1-H)=1/(1-\sum_i h_i)$ corresponds to a linear partition of many-colored blocks.
//
//
//
// + By gradually optimizing the code, I found that the answer is a product of polynomials,
// an OGF in terms of $t_i$'s and EGF in terms of $y$. But I haven't found reasonable inituition for
// the EGF. What is the combinatorial meaning of $y$, and what species should be labeled?
//
// <= Just do a multinomial expansion on $H^k$ and extract $[t_1^{c_1} ... t_n^{c_n}$ immediately.
// The factorials are just a mixing factor of blocks, grouped by color. We even do not need
// a partial expansion of $(h_i+E_{i+1})^k$ or weird matrix operations.
//
// $$r(t,y) = g(t) (1-h(t))^2 /(1-y h(t))$$
// $$(1-t+t^2 - ty)(1-t+t^2) r(t,y) = t - t^3
//

use std::io::Write;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
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

    impl<T: SemiRing + PartialEq> Poly<T> {
        pub fn new(coeffs: Vec<T>) -> Self {
            Self(coeffs)
        }

        pub fn from_iter(coeffs: impl IntoIterator<Item = T>) -> Self {
            Self(coeffs.into_iter().collect())
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

        pub fn mod_xk_in_place(&mut self, k: usize) {
            if self.degree() >= k {
                self.0.truncate(k);
            }
        }

        pub fn mod_xk(mut self, k: usize) -> Self {
            self.mod_xk_in_place(k);
            self
        }

        pub fn mul_xk_in_place(&mut self, k: usize) {
            self.0 = ((0..k).map(|_| T::zero()))
                .chain(std::mem::take(&mut self.0))
                .collect();
        }

        pub fn mul_xk(mut self, k: usize) -> Self {
            self.mul_xk_in_place(k);
            self
        }

        pub fn div_xk(&self, k: usize) -> Self {
            Self(self.0[k..].to_vec())
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

    impl<T: NTTSpec + PartialEq + Field> Poly<T> {
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
            let ps: Vec<_> = ps.into_iter().collect();

            // Potential optimization: Reduce redundant computation of the polynomial tree of `f`
            // from three times to once - one in multipoint_eval, and the other in sum_frac.
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

        pub fn multipoint_eval(&self, ps: impl IntoIterator<Item = T>) -> Vec<T> {
            let mut divisors: Vec<_> = ps
                .into_iter()
                .map(|p| Poly::new(vec![-p, T::one()]))
                .collect();
            if divisors.is_empty() {
                return vec![];
            }

            let n = divisors.len();
            for i in 0..n - 1 {
                divisors.push(divisors[i << 1].clone() * divisors[i << 1 | 1].clone());
            }

            let mut remainders = vec![Poly::zero(); 2 * n - 1];
            remainders[2 * n - 2] = self.clone();
            for i in (0..n - 1).rev() {
                remainders[i + n] %= std::mem::take(&mut divisors[i + n]);
                remainders[i << 1] = remainders[i + n].clone();
                remainders[i << 1 | 1] = std::mem::take(&mut remainders[i + n]);
            }
            (0..n)
                .map(|i| remainders[i].eval(-divisors[i].0[0].clone()))
                .collect()
        }

        pub fn pow_mod_xk(&self, mut exp: u64, k: usize) -> Self {
            let mut res = Self::one().mod_xk(k);
            let mut base = self.clone().mod_xk(k);
            while exp > 0 {
                if exp & 1 == 1 {
                    res *= base.clone();
                    res.mod_xk_in_place(k);
                }
                base *= base.clone();
                base.mod_xk_in_place(k);
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

        pub fn inv_mod_xk(&self, k: usize) -> Self {
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
            deriv_ln *= self.clone().inv_mod_xk(k);
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
            f.mul_xk_in_place(1);
            f += &Self::one();

            let mut res = Self::one();
            let mut i = 1;
            let inv2 = T::from(2u32).inv();
            while i < k {
                i <<= 1;
                let mut p = f.clone().mod_xk(i);
                p *= res.inv_mod_xk(i);
                res += &p.mod_xk(i);
                res *= inv2.clone();
                res = res.mod_xk(i);
            }
            res.mod_xk(k)
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

    impl<T: SemiRing + PartialEq> MulAssign<&'_ T> for Poly<T> {
        fn mul_assign(&mut self, rhs: &T) {
            self.0.iter_mut().for_each(|c| c.mul_assign(rhs.clone()));
        }
    }

    impl<T: SemiRing + PartialEq> MulAssign<T> for Poly<T> {
        fn mul_assign(&mut self, rhs: T) {
            self.mul_assign(&rhs);
        }
    }

    impl<T: SemiRing + PartialEq> AddAssign<&'_ Self> for Poly<T> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0.resize_with(self.len().max(rhs.len()), T::zero);
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.add_assign(b));
        }
    }

    impl<T: SemiRing + PartialEq> Add<&'_ Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, rhs: &Self) -> Self {
            self += rhs;
            self
        }
    }

    impl<T: SemiRing + PartialEq> Add<Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self {
            self += &rhs;
            self
        }
    }

    impl<T: SemiRing + PartialEq> SubAssign<&'_ Self> for Poly<T> {
        fn sub_assign(&mut self, rhs: &Self) {
            self.0.resize_with(self.len().max(rhs.len()), T::zero);
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.sub_assign(b));
        }
    }

    impl<T: SemiRing + PartialEq> Sub<&'_ Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, rhs: &Self) -> Self {
            self -= rhs;
            self
        }
    }

    impl<T: SemiRing + PartialEq> Sub<Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, rhs: Self) -> Self {
            self -= &rhs;
            self
        }
    }

    impl<T: NTTSpec + PartialEq + Field> MulAssign<Self> for Poly<T> {
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

    impl<T: NTTSpec + PartialEq + Field> Mul<Self> for Poly<T> {
        type Output = Self;
        fn mul(mut self, rhs: Self) -> Self {
            self *= rhs;
            self
        }
    }

    impl<T: NTTSpec + PartialEq + Field> DivAssign<Self> for Poly<T> {
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
            self.mod_xk_in_place(l);
            rhs.reverse();
            rhs.mod_xk_in_place(l);

            *self *= rhs.inv_mod_xk(l);
            self.0.resize(l, T::zero());
            self.reverse();
        }
    }

    impl<T: NTTSpec + PartialEq + Field> RemAssign<Self> for Poly<T> {
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

use poly::Poly;

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

// type M = ntt::sample::p998244353::M;
type M = p1000000007::M;

fn gen_factorials<T: algebra::Field + Clone + From<u32> + std::fmt::Debug>(
    n_bound: u32,
) -> (Vec<T>, Vec<T>) {
    assert!(n_bound >= 1);

    let mut fac = vec![T::one()];
    for i in 1..=n_bound {
        fac.push(fac[i as usize - 1].clone() * T::from(i));
    }

    let mut ifac = vec![T::one(); n_bound as usize + 1];
    ifac[n_bound as usize] = fac[n_bound as usize].inv();
    for i in (2..=n_bound).rev() {
        ifac[i as usize - 1] = ifac[i as usize].clone() * T::from(i - 1);
    }

    (fac, ifac)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let m0 = M::from(0u32);
    let m1 = M::from(1u32);
    let m2 = M::from(2u32);
    let m3 = M::from(2u32);

    const C_BOUND: usize = 100;
    let (fc, ifc) = gen_factorials::<M>(1e2 as u32 + 10);
    // let binom = |n: usize, k: usize| fc[n] * ifc[k] * ifc[n - k];

    let mut p = vec![vec![m0; C_BOUND]; C_BOUND];
    p[0][0] = m1;
    p[1][0] = -m1;
    p[2][0] = m1;
    for c in 1..C_BOUND {
        for k in 0..C_BOUND {
            p[c][k] = p[c][k] + p[c - 1][k];
            if c >= 2 {
                p[c][k] = p[c][k] - p[c - 2][k];
            }
            if k >= 1 {
                p[c][k] = p[c][k] + p[c - 1][k - 1];
            }
        }
    }

    let mut r = vec![vec![m0; C_BOUND]; C_BOUND];
    r[1][0] = m1;
    r[3][0] = -m1;
    for c in 1..C_BOUND {
        for k in 0..C_BOUND {
            r[c][k] = r[c][k] - m2 * r[c - 1][k];
            if c >= 2 {
                r[c][k] = r[c][k] + m3 * r[c - 2][k];
                if c >= 3 {
                    r[c][k] = r[c][k] - m3 * r[c - 3][k];
                    if c >= 4 {
                        r[c][k] = r[c][k] + m3 * r[c - 4][k];
                    }
                }
            }
            if k >= 1 {
                r[c][k] = r[c][k] + r[c - 1][k - 1];
                if c >= 2 {
                    r[c][k] = r[c][k] - r[c - 2][k - 1];
                    if c >= 3 {
                        r[c][k] = r[c][k] + r[c - 3][k - 1];
                    }
                }
            }
        }
    }

    for c in 0..C_BOUND {
        for k in 0..p[c].len() {
            p[c][k] *= ifc[k];
        }
    }
    for c in 0..C_BOUND {
        for k in 0..r[c].len() {
            r[c][k] *= ifc[k];
        }
    }

    let n: usize = input.value();
    let cs: Vec<usize> = (0..n).map(|_| input.value()).collect();

    let mut ans = m0;
    if n == 1 {
        ans = M::from(cs[0] as u32);
    } else {
        let (f, _) = Poly::sum_frac(
            cs.iter()
                .map(|&c| (Poly::from_iter(r[c].clone()), Poly::new(p[c].clone()))),
        );

        for i in 0..f.0.len() {
            ans += f.0[i] * fc[i];
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
