use std::cell::OnceCell;
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

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T>
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().ok()
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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
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

const P: u32 = 1_000_000_007;
type M = mint_mont::M32<P>;

use std::ops::{Add, Mul, Neg, Sub};

use algebra::{Field, SemiRing};

#[derive(Default, Clone, Debug)]
struct Poly(Vec<M>);

thread_local! {
    static FACTORIALS: OnceCell<(Vec<M>, Vec<M>)> = OnceCell::new();
}

fn gen_factorials(n_bound: u32) {
    assert!(n_bound >= 1);

    FACTORIALS.with(|cell| {
        cell.get_or_init(|| {
            let mut fac = vec![M::one()];
            for i in 1..=n_bound {
                fac.push(fac[i as usize - 1].clone() * M::from(i));
            }

            let mut ifac = vec![M::one(); n_bound as usize + 1];
            ifac[n_bound as usize] = fac[n_bound as usize].inv();
            for i in (2..=n_bound).rev() {
                ifac[i as usize - 1] = ifac[i as usize].clone() * M::from(i);
            }

            (fac, ifac)
        });
    })
}

fn fac(n: u32) -> M {
    gen_factorials(300);
    FACTORIALS.with(|cell| {
        let (fac, _) = cell.get().unwrap();
        fac[n as usize]
    })
}

fn ifac(n: u32) -> M {
    gen_factorials(300);
    FACTORIALS.with(|cell| {
        let (_, ifac) = cell.get().unwrap();
        ifac[n as usize]
    })
}

fn binom(n: u32, k: u32) -> M {
    if n < k {
        return M::zero();
    }
    fac(n) * ifac(k) * ifac(n - k)
}

impl Poly {
    const fn zero() -> Self {
        Poly(vec![])
    }

    fn one() -> Self {
        Poly(vec![M::one()])
    }

    fn eval(&self, x: M) -> M {
        let mut res = M::zero();
        for i in (0..self.0.len()).rev() {
            res = res * x + self.0[i];
        }
        res
    }

    fn shift_arg(&self, a: M) -> Self {
        let mut res = vec![M::zero(); self.0.len()];
        for i in 0..self.0.len() {
            let mut a_pow = M::one();
            for j in 0..=i {
                res[i - j] += binom(i as u32, j as u32) * a_pow * self.0[i];
                a_pow *= a;
            }
        }
        Poly(res)
    }

    fn deriv(&self) -> Self {
        if self.0.is_empty() {
            return Self::zero();
        }

        let mut res = vec![M::zero(); self.0.len() - 1];
        for i in 1..self.0.len() {
            res[i - 1] = self.0[i] * M::from(i as u32);
        }
        Poly(res)
    }

    fn integrate(&self) -> Self {
        if self.0.is_empty() {
            return Self::zero();
        }

        let mut res = vec![M::zero(); self.0.len() + 1];
        for i in 0..self.0.len() {
            res[i + 1] = self.0[i] * M::from(i as u32 + 1).inv();
        }
        Poly(res)
    }
}

impl Add for Poly {
    type Output = Self;
    fn add(mut self, mut rhs: Self) -> Self::Output {
        if self.0.len() < rhs.0.len() {
            std::mem::swap(&mut self, &mut rhs);
        }
        for i in 0..rhs.0.len() {
            self.0[i] += rhs.0[i];
        }
        self
    }
}

impl Sub for Poly {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<M> for Poly {
    type Output = Self;
    fn sub(mut self, rhs: M) -> Self::Output {
        if self.0.is_empty() {
            return Poly(vec![-rhs]);
        }
        self.0[0] -= rhs;
        self
    }
}

impl Mul for Poly {
    type Output = Self;
    fn mul(mut self, mut rhs: Self) -> Self::Output {
        if self.0.len() < rhs.0.len() {
            std::mem::swap(&mut self, &mut rhs);
        }
        if rhs.0.is_empty() {
            return Poly::zero();
        }

        let mut res = vec![M::zero(); self.0.len() + rhs.0.len() - 1];
        for j in 0..rhs.0.len() {
            for i in 0..self.0.len() {
                res[i + j] += self.0[i] * rhs.0[j];
            }
        }
        Poly(res)
    }
}

impl Neg for Poly {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        for c in &mut self.0 {
            *c = -(*c);
        }
        self
    }
}

// Piecewise polynomial with integer breakpoints
// Analytically continues from the endpoints of the both sides
#[derive(Default, Clone, Debug)]
struct PWPoly {
    x_lower: i32,  // Leftmost breakpoint
    ps: Vec<Poly>, // Value on intervals [x0 + i, x0 + i + 1)
}

impl PWPoly {
    fn zero() -> Self {
        Self::default()
    }

    fn get_current_poly(&self, x: i32) -> Poly {
        if self.ps.is_empty() {
            return Poly::zero();
        }
        let i = (x - self.x_lower).max(0).min(self.ps.len() as i32 - 1);
        self.ps[i as usize].clone()
    }

    fn eval(&self, x: i32) -> M {
        self.get_current_poly(x)
            .eval(M::from((P as i32 + x) as u32))
    }

    fn shift_arg(&self, a: i32) -> Self {
        Self {
            x_lower: self.x_lower - a,
            ps: self
                .ps
                .iter()
                .map(|p| p.shift_arg(M::from((P as i32 + a) as u32)))
                .collect(),
        }
    }

    fn deriv(&self) -> Self {
        Self {
            x_lower: self.x_lower,
            ps: self.ps.iter().map(|p| p.deriv()).collect(),
        }
    }

    fn integrate(&self) -> Self {
        let mut res = Self {
            x_lower: self.x_lower,
            ps: self.ps.iter().map(|p| p.integrate()).collect(),
        };

        let mut shift = vec![M::zero(); res.ps.len()];
        for i in 1..res.ps.len() {
            let x = M::from((P as i32 + res.x_lower + i as i32) as u32);
            shift[i] = shift[i - 1] + res.ps[i - 1].eval(x) - res.ps[i].eval(x);
        }

        for i in 0..res.ps.len() {
            res.ps[i] = std::mem::take(&mut res.ps[i]) + Poly(vec![shift[i]]);
        }

        let delta = res.eval(0);
        for p in &mut res.ps {
            *p = std::mem::take(p) - Poly(vec![delta]);
        }

        res
    }

    fn build_binary_skeleton(&self, other: &Self) -> Self {
        let x_lower = self.x_lower.min(other.x_lower);
        let x_upper =
            (self.x_lower + self.ps.len() as i32).max(other.x_lower + other.ps.len() as i32);
        Self {
            x_lower: self.x_lower.min(other.x_lower),
            ps: vec![Poly::zero(); (x_upper - x_lower) as usize],
        }
    }
}

impl Add for PWPoly {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut res = self.build_binary_skeleton(&rhs);
        for i in 0..res.ps.len() {
            let x = res.x_lower + i as i32;
            res.ps[i] = self.get_current_poly(x) + rhs.get_current_poly(x);
        }
        res
    }
}

impl Add<Poly> for PWPoly {
    type Output = Self;
    fn add(mut self, rhs: Poly) -> Self::Output {
        for c in &mut self.ps {
            *c = std::mem::take(c) + rhs.clone();
        }
        self
    }
}

impl Sub for PWPoly {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<M> for PWPoly {
    type Output = Self;
    fn sub(mut self, rhs: M) -> Self::Output {
        for c in &mut self.ps {
            *c = std::mem::take(c) - rhs;
        }
        self
    }
}

impl Mul for PWPoly {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = self.build_binary_skeleton(&rhs);
        for i in 0..res.ps.len() {
            let x = res.x_lower + i as i32;
            res.ps[i] = self.get_current_poly(x) * rhs.get_current_poly(x);
        }
        res
    }
}

impl Mul<Poly> for PWPoly {
    type Output = Self;
    fn mul(mut self, rhs: Poly) -> Self::Output {
        for c in &mut self.ps {
            *c = std::mem::take(c) * rhs.clone();
        }
        self
    }
}

impl Mul<M> for PWPoly {
    type Output = Self;
    fn mul(self, rhs: M) -> Self::Output {
        self * Poly(vec![rhs])
    }
}

impl Neg for PWPoly {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        for c in &mut self.ps {
            *c = -std::mem::take(c);
        }
        self
    }
}

pub mod reroot {
    // O(n) rerooting dp for trees with combinable, non-invertible pulling operation. (Monoid action)
    // Technically, its a static, offline variant of the top tree.
    // https://codeforces.com/blog/entry/124286
    // https://github.com/koosaga/olympiad/blob/master/Library/codes/data_structures/all_direction_tree_dp.cpp
    pub trait AsBytes<const N: usize> {
        unsafe fn as_bytes(self) -> [u8; N];
        unsafe fn decode(bytes: [u8; N]) -> Self;
    }

    #[macro_export]
    macro_rules! impl_as_bytes {
        ($T:ty, $N: ident) => {
            const $N: usize = std::mem::size_of::<$T>();

            impl crate::reroot::AsBytes<$N> for $T {
                unsafe fn as_bytes(self) -> [u8; $N] {
                    std::mem::transmute::<$T, [u8; $N]>(self)
                }

                unsafe fn decode(bytes: [u8; $N]) -> $T {
                    std::mem::transmute::<[u8; $N], $T>(bytes)
                }
            }
        };
    }
    pub use impl_as_bytes;

    impl_as_bytes!((), __N_UNIT);
    impl_as_bytes!(u32, __N_U32);
    impl_as_bytes!(u64, __N_U64);
    impl_as_bytes!(i32, __N_I32);
    impl_as_bytes!(i64, __N_I64);

    pub trait DpSpec {
        type E: Clone; // edge weight
        type V: Clone; // Subtree aggregate on a node
        type F: Clone; // pulling operation (edge dp)
        fn lift_to_action(&self, node: &Self::V, weight: &Self::E) -> Self::F;
        fn id_action(&self) -> Self::F;
        fn rake_action(&self, node: &Self::V, lhs: &mut Self::F, rhs: &Self::F);
        fn apply(&self, node: &mut Self::V, action: &Self::F);
        fn finalize(&self, node: &mut Self::V);
    }

    fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
        for (x, y) in xs.iter_mut().zip(&ys) {
            *x ^= *y;
        }
    }

    const UNSET: u32 = !0;

    fn for_each_in_list(xor_links: &[u32], start: u32, mut visitor: impl FnMut(u32)) -> u32 {
        let mut u = start;
        let mut prev = UNSET;
        loop {
            visitor(u);
            let next = xor_links[u as usize] ^ prev;
            if next == UNSET {
                return u;
            }
            prev = u;
            u = next;
        }
    }

    pub fn run<'a, const N: usize, R>(
        cx: &R,
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, R::E)>,
        data: &mut [R::V],
        mut yield_edge_dp: impl FnMut(usize, &R::F, &R::F, &R::E),
    ) where
        R: DpSpec,
        R::E: Default + AsBytes<N>,
    {
        let root = 0;
        let mut degree = vec![0; n_verts];
        let mut xor_neighbors: Vec<(u32, [u8; N])> = vec![(0u32, [0u8; N]); n_verts];
        for (u, v, w) in edges
            .into_iter()
            .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
        {
            degree[u as usize] += 1;
            xor_neighbors[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor_neighbors[u as usize].1, unsafe {
                AsBytes::as_bytes(w)
            });
        }

        // Upward propagation
        let data_orig = data.to_owned();
        let mut action_upward = vec![cx.id_action(); n_verts];
        let mut topological_order = vec![];
        let mut first_child = vec![UNSET; n_verts];
        let mut xor_siblings = vec![UNSET; n_verts];
        degree[root] += 2;
        for mut u in 0..n_verts {
            while degree[u as usize] == 1 {
                let (p, w_encoded) = xor_neighbors[u as usize];
                degree[u as usize] = 0;
                degree[p as usize] -= 1;
                xor_neighbors[p as usize].0 ^= u as u32;
                xor_assign_bytes(&mut xor_neighbors[p as usize].1, w_encoded);
                let w = unsafe { AsBytes::decode(w_encoded) };

                let c = first_child[p as usize];
                xor_siblings[u as usize] = c ^ UNSET;
                if c != UNSET {
                    xor_siblings[c as usize] ^= u as u32 ^ UNSET;
                }
                first_child[p as usize] = u as u32;

                let mut sum_u = data[u as usize].clone();
                cx.finalize(&mut sum_u);
                action_upward[u as usize] = cx.lift_to_action(&sum_u, &w);
                cx.apply(&mut data[p as usize], &action_upward[u as usize]);

                topological_order.push((u as u32, p, w));
                u = p as usize;
            }
        }
        topological_order.push((root as u32, UNSET, R::E::default()));
        cx.finalize(&mut data[root]);

        // Downward propagation
        let mut action_exclusive = vec![cx.id_action(); n_verts];
        for (u, p, w) in topological_order.into_iter().rev() {
            let action_from_parent;
            if p != UNSET {
                let mut sum_exclusive = data_orig[p as usize].clone();
                cx.apply(&mut sum_exclusive, &action_exclusive[u as usize]);
                cx.finalize(&mut sum_exclusive);
                action_from_parent = cx.lift_to_action(&sum_exclusive, &w);

                let sum_u = &mut data[u as usize];
                cx.apply(sum_u, &action_from_parent);
                cx.finalize(sum_u);
                yield_edge_dp(
                    u as usize,
                    &action_upward[u as usize],
                    &action_from_parent,
                    &w,
                );
            } else {
                action_from_parent = cx.id_action();
            }

            if first_child[u as usize] != UNSET {
                let mut prefix = action_from_parent.clone();
                let last = for_each_in_list(&xor_siblings, first_child[u as usize], |v| {
                    action_exclusive[v as usize] = prefix.clone();
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut prefix,
                        &action_upward[v as usize],
                    );
                });

                let mut postfix = cx.id_action();
                for_each_in_list(&xor_siblings, last, |v| {
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut action_exclusive[v as usize],
                        &postfix,
                    );
                    cx.rake_action(
                        &data_orig[u as usize],
                        &mut postfix,
                        &action_upward[v as usize],
                    );
                });
            }
        }
    }
}

struct RandomDiameter;

impl reroot::DpSpec for RandomDiameter {
    type E = ();
    type V = PWPoly;
    type F = (PWPoly, PWPoly);

    fn lift_to_action(&self, node: &Self::V, _: &Self::E) -> Self::F {
        let f = (node.clone() - node.shift_arg(-1)).integrate();
        (node.clone(), f.clone() - f.eval(f.x_lower))
    }

    fn id_action(&self) -> Self::F {
        let poly_0 = || Poly(vec![]);
        let poly_1 = || Poly(vec![M::one()]);
        let cdf_kronecker = PWPoly {
            x_lower: -1,
            ps: vec![poly_0(), poly_1()],
        };

        (
            cdf_kronecker,
            PWPoly {
                x_lower: 0,
                ps: vec![poly_1()],
            },
        )
    }

    fn rake_action(&self, _: &Self::V, lhs: &mut Self::F, rhs: &Self::F) {
        self.apply(&mut lhs.1, rhs);
    }

    fn apply(&self, node: &mut Self::V, action: &Self::F) {
        *node = std::mem::take(node) * action.1.clone();
    }

    fn finalize(&self, _: &mut Self::V) {}
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let poly_0 = || Poly(vec![]);
    let poly_1 = || Poly(vec![M::one()]);
    let poly_x = || Poly(vec![M::zero(), M::one()]);
    let poly_x2 = || Poly(vec![M::zero(), M::zero(), M::one()]);
    let inv2 = M::from(2u8).inv();

    let int_full = |p: &PWPoly| {
        let c = p.integrate();
        let lower = p.x_lower;
        let upper = p.x_lower + p.ps.len() as i32;
        c.eval(upper) - c.eval(lower)
    };
    let narrow = |p: &PWPoly| p.clone() - p.shift_arg(-1);

    let cdf_kronecker = PWPoly {
        x_lower: -1,
        ps: vec![poly_0(), poly_1()],
    };

    let n: usize = input.value();
    let edges = (0..n - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1, ()));
    let mut node_cdfs = vec![cdf_kronecker; n];
    let mut ans = M::zero();
    reroot::run(
        &RandomDiameter,
        n,
        edges,
        &mut node_cdfs,
        |_, (ce, _), (cf, _), _| {
            let mut ce = ce.clone();
            let mut cf = cf.clone();

            // PDF's
            // ill-defined for delta distribution
            let mut pe = ce.deriv();
            let mut pf = cf.deriv();

            let mut ke = int_full(&pe);
            let mut kf = int_full(&pf);
            assert!(ke == M::zero() || ke == M::one());
            assert!(kf == M::zero() || kf == M::one());

            if kf == M::zero() {
                std::mem::swap(&mut ce, &mut cf);
                std::mem::swap(&mut pe, &mut pf);
                std::mem::swap(&mut ke, &mut kf);
            }

            if kf == M::zero() {
                debug_assert!(ke == M::zero());
                ans += inv2;
            } else if ke == M::zero() {
                let h = pf.clone() * Poly(vec![inv2, M::one(), -M::from(3u8) * inv2]);
                let r = h.integrate();
                ans += r.eval(1) - r.eval(0);
                // println!("end 120m {:?}", (r.eval(1) - r.eval(0)) * M::from(120u8));
            } else {
                for _ in 0..2 {
                    let int_1 = narrow(&cf);
                    let int_f = narrow(&(pf.clone() * poly_x()).integrate());
                    let int_f2 = narrow(&(pf.clone() * poly_x2()).integrate());

                    let h = int_f2 * inv2
                        + int_f * Poly(vec![M::one(), M::one()])
                        + int_1 * Poly(vec![inv2, M::one(), -M::from(3u8) * inv2]);
                    // println!(
                    //     "120m {:?}",
                    //     int_full(&(h.clone() * pe.clone())) * M::from(120u8)
                    // );
                    ans += int_full(&(h * pe.clone()));

                    std::mem::swap(&mut ce, &mut cf);
                    std::mem::swap(&mut pe, &mut pf);
                }
            }
        },
    );

    // println!("60ans {:?}", ans * M::from(60u8));
    writeln!(output, "{}", ans).unwrap();
}
