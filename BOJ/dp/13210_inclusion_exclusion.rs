use std::io::Write;

use num_mod_static::{CommRing, PowBy};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
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

const P: u64 = 1_000_000_007;
type ModP = num_mod_static::NaiveModInt<num_mod_static::ByU64<P>>;

fn count_rect_in_histogram(hs: &[u16], count_delta: &mut [ModP]) {
    let m = hs.len();
    let mut stack = vec![(0, 0)];
    for &h in hs.iter().chain(Some(&0)) {
        let mut width = 0;
        loop {
            let (h_top, w_top) = *stack.last().unwrap();
            if h_top <= h {
                break;
            }
            width += w_top;
            stack.pop();

            let (h_left, _) = stack.last().copied().unwrap();
            count_delta[h_top as usize * (m + 1) + width] += 1.into();
            count_delta[h_left.max(h) as usize * (m + 1) + width] -= 1.into();
        }
        width += 1;
        stack.push((h, width));
    }
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();
    let mut grid = vec![0u16; n * m];
    for i in 0..n {
        let row = input.token().as_bytes();
        for j in 0..m {
            grid[i * m + j] = 1 << row[j] - b'1';
        }
    }

    let mut sum_k_subset = vec![vec![ModP::zero(); (n + 1) * (m + 1)]; k + 1];
    for k_curr in 1..=k {
        let mut sum_le_k_subset_delta = vec![ModP::zero(); (n + 1) * (m + 1)];
        for mask in 1u16..1 << k {
            if mask.count_ones() != k_curr as u32 {
                continue;
            }

            let mut prev_row = vec![0u16; m];
            for src in grid.chunks(m) {
                let mut row = vec![0u16; m];
                for ((c, p), f) in row.iter_mut().zip(&prev_row).zip(src) {
                    *c = (*f & mask == *f) as u16;
                    *c *= 1 + *p;
                }

                count_rect_in_histogram(&row, &mut sum_le_k_subset_delta);

                prev_row = row;
            }
        }

        {
            let mut sum_le_k_subset = sum_le_k_subset_delta;
            for j in 0..=m {
                for i in (0..n).rev() {
                    let temp = sum_le_k_subset[(i + 1) * (m + 1) + j];
                    sum_le_k_subset[i * (m + 1) + j] += temp;
                }
            }
            for _ in 0..2 {
                for i in 0..=n {
                    for j in (0..m).rev() {
                        let temp = sum_le_k_subset[i * (m + 1) + j + 1];
                        sum_le_k_subset[i * (m + 1) + j] += temp;
                    }
                }
            }

            let mut factor = ModP::one();
            for l in k_curr..=k {
                if l != k_curr {
                    factor *= -ModP::from((k - l + 1) as u64)
                        * ModP::from((l - k_curr) as u64).pow(P - 2);
                }
                for (a, c) in sum_k_subset[l].iter_mut().zip(&sum_le_k_subset) {
                    *a += *c * factor;
                }
            }
        }
    }

    let mut ans = ModP::one();
    for k_curr in 1..=k {
        for i in 1..=n {
            for j in 1..=m {
                let c = sum_k_subset[k_curr][i * (m + 1) + j];
                ans *= c + ModP::from((k_curr as u64) * (i as u64) * (j as u64));
            }
        }
    }
    writeln!(output, "{}", u64::from(ans)).unwrap();
}
