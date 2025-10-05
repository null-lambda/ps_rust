use std::io::Write;

use algebra::{Field, SemiRing};
use buffered_io::BufReadExt;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    #[allow(unused)]
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

mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Option<Self> {
            let mut seed = 0;
            unsafe { (std::arch::x86_64::_rdrand64_step(&mut seed) == 1).then(|| Self(seed)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            debug_assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            if n == 0 {
                return;
            }

            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
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

pub struct EmulatedInteractor {
    zs: Vec<i64>,
    n_query_calls: u32,
}

impl EmulatedInteractor {
    pub fn new(n: usize, rng: &mut rand::SplitMix64) -> Self {
        let zs: Vec<_> = (0..n).map(|_| rng.range_u64(0..1_000_001) as i64).collect();
        Self {
            zs,
            n_query_calls: 0,
        }
    }

    pub fn query(&mut self, xs: impl IntoIterator<Item = usize>) -> i64 {
        let xs = xs.into_iter().collect::<Vec<_>>();
        let xs_set = std::collections::HashSet::<usize>::from_iter(xs.iter().copied());
        assert_eq!(xs.len(), xs_set.len());

        self.n_query_calls += 1;
        xs.into_iter().map(|x| self.zs[x]).sum()
    }

    pub fn terminate(self, ans: i64) {
        assert_eq!(ans, self.zs.iter().sum());
    }
}

pub struct Interactor<'a, R, W> {
    input: &'a mut R,
    output: &'a mut W,
    n_query_calls: u32,
}

impl<'a, R: buffered_io::BufReadExt, W: std::io::Write> Interactor<'a, R, W> {
    pub fn new(input: &'a mut R, output: &'a mut W) -> Self {
        Self {
            input,
            output,
            n_query_calls: 0,
        }
    }

    pub fn query(&mut self, xs: impl IntoIterator<Item = usize>) -> i64 {
        self.n_query_calls += 1;
        write!(self.output, "? ").unwrap();
        for x in xs {
            write!(self.output, "{} ", x + 1).unwrap();
        }
        writeln!(self.output).unwrap();
        self.output.flush().unwrap();

        self.input.value()
    }

    pub fn terminate(self, ans: i64) {
        writeln!(self.output, "! {}", ans).unwrap();
        self.output.flush().unwrap();
    }
}

const P: u32 = 1000000007;
type M = mint_mont::M32<P>;

fn solve_tridiagonal_m1(mut bs: Vec<M>, mut rhs: Vec<M>) -> M {
    // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    let n = bs.len();
    for i in (0..n - 1).rev() {
        let z = -bs[i + 1].inv();
        bs[i] += z;
        rhs[i] = rhs[i] - z * rhs[i + 1];
    }
    rhs[0] / bs[0]
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    // let mut rng = rand::SplitMix64::from_entropy().unwrap();
    // for n in 1..=1000 {
    //     for k in 1..=n {
    //         eprintln!("{:?}", (n, k));
    // let mut itr = EmulatedInteractor::new(n, &mut rng);
    let mut itr = Interactor::new(&mut input, &mut output);

    let mut rem = n;
    let mut w = k;
    let mut hole = vec![];

    let mut history = vec![];
    loop {
        let mut i = 0;
        let mut s = 0;

        let mut a = 0;
        let count = rem / w;
        for b in 0..count {
            let xs = (i..i + w).chain(hole.iter().copied());
            let ds = itr.query(xs);

            s += ds;

            if b == 0 {
                a = ds;
            }

            i += w;
        }

        history.push((s as i64, a as i64, count as i64));

        let delta_hole = rem / w * w..rem;
        if delta_hole.len() == 0 {
            break;
        }
        hole.extend(delta_hole);

        rem = w;
        w = k - hole.len();
    }

    let (s0, _, _0) = history[0];
    let mut ans = s0;
    if history.len() >= 2 {
        let c = history.len();
        let r = c - 1;
        {
            // let mut mat = vec![R::from(0); r * c];
            // for i in 1..r {
            //     mat[i * c + (i - 1)] = R::from(-1);
            //     mat[(i - 1) * c + i] = R::from(-1);
            // }
            // for i in 0..r {
            //     let (_, a, _) = history[i];
            //     let (s, _, count) = history[i + 1];
            //     mat[i * c + i] = R::from(count + (i + 1 != r) as i64);
            //     mat[i * c + r] = R::from(s - a);
            // }

            // let mat = linalg::Matrix {
            //     n_rows: r,
            //     n_cols: c,
            //     data: mat,
            // };

            // // println!("{:?}", mat);
            // // let (_rank, _det, rref) = linalg::rref(&mat);
            // // println!("{:?}", rref);
            // let h0 = rref.data[0 * c + r].numer();
            // ans += h0;
        }

        let mut bs = vec![M::zero(); r];
        let mut rhs = vec![M::zero(); r];
        for i in 0..r {
            let (_, a, _) = history[i];
            let (s, _, count) = history[i + 1];
            bs[i] = M::from(count as u64 + (i + 1 != r) as u64);
            rhs[i] = M::from((s - a).rem_euclid(P as i64) as u32);
        }

        // Key fact: every hole-sum <= 1e6 * n <= 1e9
        // => apply modp with some prime p > 1e9 to bypass overflow problem
        let h0: u32 = solve_tridiagonal_m1(bs, rhs).into();
        ans += h0 as i64;
    }

    itr.terminate(ans);
    // }
    // }
}
