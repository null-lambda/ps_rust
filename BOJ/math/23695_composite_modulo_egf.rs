// Typical functional graph counting
// Composite modulus -> FFT  impossible. Do some binomial convolution.
//
// $$T(x) =x e^T(x)$$
// $$B(x) = \int T(x)/x =\sum_{n \ge 1} n^{n-2} \frac{x^n}{n!}$$
// $$(k \ge 3) C_k(x,t) = \frac{T(x)^k}{2k}  t^k$$
// $$D(x,t) = B+C_3+C_4+...$$
// $$Ans = n! [x^n] \partial_t e^{D(x,t)}|_{t=1} = n![x^n] \partial_t D(x,1) e^{D(x,1)}$$
//
// $$R(x) = -\ln (1-x) - x-\frac{x^2}{2}$$
// $$S(x) = \frac{x^3}{1-x}$$
// $$D(x,1) = B(x)  + \frac{1}{2} R(T(x))$$
// $$\partial_t D(x,1) =\frac{1}{2} S(T(x))$$

use buffered_io::BufReadExt;
use std::io::Write;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
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

use algebra::*;
type M = mint_dyn::M64;

const N: usize = 3001;

struct Cx {
    binom: Vec<[u32; N]>,
}

impl Cx {
    fn new(p: u64) -> Self {
        let mut binom = vec![[0u32; N]; N * 2];
        for n in 0..N * 2 {
            binom[n][0] = 1;
            for k in 1..=n / 2 {
                binom[n][k] = ((binom[n - 1][k.min(n - 1 - k)] as u64
                    + binom[n - 1][(k - 1).min(n - k)] as u64)
                    % p) as u32;
            }
        }
        Self { binom }
    }

    fn binom(&self, n: usize, k: usize) -> M {
        self.binom[n][k.min(n - k)].into()
    }
    fn binom_div(&self, n: usize, k: usize, a: u64) -> M {
        assert!(self.binom[n][k] as u64 % a == 0);
        (self.binom[n][k.min(n - k)] as u64 / a).into()
    }
}

fn add(mut xs: Vec<M>, mut ys: Vec<M>) -> Vec<M> {
    if xs.len() < ys.len() {
        std::mem::swap(&mut xs, &mut ys);
    }
    for (x, y) in xs.iter_mut().zip(ys) {
        *x += y;
    }
    xs
}

fn neg(mut xs: Vec<M>) -> Vec<M> {
    for x in &mut xs {
        *x = -*x;
    }
    xs
}

fn smul(mut xs: Vec<M>, a: M) -> Vec<M> {
    for x in &mut xs {
        *x *= a;
    }
    xs
}

fn try_div(x: M, a: u64) -> Option<M> {
    let inner = u64::from(x);
    (inner % a == 0).then(|| M::from(inner / a))
}

fn sdiv(mut xs: Vec<M>, a: u64) -> Vec<M> {
    for x in &mut xs {
        *x = try_div(*x, a).unwrap();
    }
    xs
}

fn sdiv2(xs: Vec<M>) -> Vec<M> {
    sdiv(xs, 2)
}

// Binomial conv.
fn mul(cx: &Cx, xs: &[M], ys: &[M], n: usize) -> Vec<M> {
    let mut res = vec![M::zero(); n];
    for i in 0..xs.len() {
        for j in 0..ys.len().min(n.saturating_sub(i)) {
            res[i + j] += cx.binom(i + j, j) * xs[i] * ys[j];
        }
    }
    res
}

// Binomial conv.
fn sq_div2(cx: &Cx, xs: &[M]) -> Vec<M> {
    let n = xs.len();
    let mut res = vec![M::zero(); n];

    assert!(xs[0] == M::zero());
    for i in 0..xs.len() {
        for j in 0..xs.len().min(n.saturating_sub(i)).min(i) {
            res[i + j] += cx.binom(i + j, j) * xs[i] * xs[j];
        }
        if i >= 1 && i + i < n {
            res[i + i] += cx.binom_div(i + i, i, 2) * xs[i] * xs[i];
        }
    }
    res
}

// 1/(1-f)
fn inv_1m(cx: &Cx, xs: &[M]) -> Vec<M> {
    assert!(xs[0] == M::zero());
    let n = xs.len();
    let mut res = vec![M::zero(); n];
    res[0] = M::one();
    for i in 1..n {
        for k in 0..i {
            res[i] = res[i] + cx.binom(i, k) * res[k] * xs[i - k];
        }
    }
    res
}

fn ln(cx: &Cx, xs: &[M]) -> Vec<M> {
    assert!(xs[0] == M::one());
    let n = xs.len();
    let mut res = vec![M::zero(); n];
    for i in 1..n {
        res[i] = xs[i];
        for k in 1..i {
            res[i] = res[i] - cx.binom(i - 1, k - 1) * res[k] * xs[i - k];
        }
    }
    res
}

fn exp(cx: &Cx, xs: &[M]) -> Vec<M> {
    assert!(xs[0] == M::zero());
    let n = xs.len();
    let mut res = vec![M::zero(); n];
    res[0] = M::one();
    for i in 1..n {
        for k in 0..i {
            res[i] = res[i] + cx.binom(i - 1, k) * res[k] * xs[i - k];
        }
    }
    res
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let q: usize = input.value();
    let p: u64 = input.value();

    let f;

    let p_ex = if p % 2 == 0 { p * 2 } else { p * 4 };
    let cx = Cx::new(p_ex);
    M::set_modulus(p_ex);

    // for n in 0..=10 {
    //     for k in 0..=10 {
    //         eprint!("{:?} ", cx.binom(n, k));
    //     }
    //     eprintln!();
    // }

    let mut t = vec![M::zero(); N];
    let mut b = vec![M::zero(); N];
    b[1] = M::one();
    t[1] = M::one();
    for i in 2..N {
        b[i] = M::from(i as u32).pow(i - 2);
        t[i] = b[i] * M::from(i as u32);
    }

    let t2_d2 = sq_div2(&cx, &t);
    let t3_d2 = mul(&cx, &t, &t2_d2, N);
    let s_comp_t_d2 = mul(&cx, &t3_d2, &inv_1m(&cx, &t), N);

    let mut nt = neg(t.clone());
    nt[0] += M::one();
    let mut r_comp_t = ln(&cx, &nt);
    r_comp_t = add(r_comp_t, t);
    r_comp_t = add(r_comp_t, t2_d2);
    r_comp_t = neg(r_comp_t);

    // eprintln!("{:?}", r_comp_t);

    let u = exp(&cx, &add(b, sdiv2(r_comp_t)));
    f = mul(&cx, &s_comp_t_d2, &u, N);

    for _ in 0..q {
        let i: usize = input.value();

        let mut ans: u64 = f[i].into();
        ans %= p;
        writeln!(output, "{}", ans).unwrap();
        //
    }
}
