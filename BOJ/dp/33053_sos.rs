use std::io::Write;

use algebra::{Field, SemiRing};

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

pub mod dp_sos {
    // A pair of subset Zeta and Mobius transforms and their variants.
    use std::ops::{AddAssign, SubAssign};

    pub trait CommGroup:
        Default + for<'a> AddAssign<&'a Self> + for<'a> SubAssign<&'a Self>
    {
    }
    impl<T: Default + for<'a> AddAssign<&'a Self> + for<'a> SubAssign<&'a Self>> CommGroup for T {}

    fn unit_bits(pow2: usize) -> impl Iterator<Item = usize> {
        assert!(pow2.is_power_of_two());
        (0..pow2.trailing_zeros()).map(|i| 1 << i)
    }

    fn chunks_exact_mut_paired<T>(
        xs: &mut [T],
        block_size: usize,
    ) -> impl Iterator<Item = (&mut [T], &mut [T])> {
        xs.chunks_exact_mut(block_size * 2)
            .map(move |block| block.split_at_mut(block_size))
    }

    fn propagate_subset_sum<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
        let n = xs.len();
        for e in unit_bits(n) {
            for (zs, os) in chunks_exact_mut_paired(xs, e) {
                for (z, o) in zs.iter_mut().zip(os) {
                    modifier(z, o);
                }
            }
        }
    }

    pub fn subset_sums<T: CommGroup>(xs: &mut [T]) {
        propagate_subset_sum(xs, |z, o| *o += z);
    }

    pub fn inv_subset_sums<T: CommGroup>(xs: &mut [T]) {
        propagate_subset_sum(xs, |z, o| *o -= z);
    }

    pub fn superset_sums<T: CommGroup>(xs: &mut [T]) {
        propagate_subset_sum(xs, |z, o| *z += o);
    }

    pub fn inv_superset_sums<T: CommGroup>(xs: &mut [T]) {
        propagate_subset_sum(xs, |z, o| *z -= o);
    }
}

type M = mint::M64<1000000007>;

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    let mut fs: Vec<_> = (0..1u32 << k)
        .map(|i| M::from(i.count_ones() as u64).inv())
        .collect();
    dp_sos::inv_superset_sums(&mut fs);

    let xs: Vec<u32> = (0..n).map(|_| input.u32()).collect();

    let mut cs0 = vec![M::zero(); 1 << k];
    for &x in &xs {
        cs0[x as usize] += M::one();
    }

    let mut cs1: Vec<_> = cs0
        .iter()
        .enumerate()
        .map(|(i, x)| M::from(i.count_ones()) * x)
        .collect();

    let t = |cs: &mut [M]| {
        dp_sos::subset_sums(cs);
        for (c, f) in cs.iter_mut().zip(&fs) {
            *c *= f;
        }
        dp_sos::superset_sums(cs);
    };

    t(&mut cs0);
    t(&mut cs1);

    for &x in &xs {
        let ans = cs0[x as usize] * M::from(x.count_ones()) + cs1[x as usize] - M::from(n as u32);
        writeln!(output, "{}", ans).unwrap();
    }
}
