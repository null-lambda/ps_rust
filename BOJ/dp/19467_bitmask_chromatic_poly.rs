use std::io::Write;

use algebra::SemiRing;

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

    pub trait CommGroup: Default + AddAssign + SubAssign + Clone {}
    impl<T: Default + AddAssign + SubAssign + Clone> CommGroup for T {}

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

    fn kronecker_prod<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
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
        kronecker_prod(xs, |z, o| *o += z.clone());
    }

    pub fn inv_subset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *o -= z.clone());
    }

    pub fn superset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *z += o.clone());
    }

    pub fn inv_superset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *z -= o.clone());
    }

    // Fast Walsh-Hadamard Transform.
    // For an inverse transform, do fwht and then divide by 2^n == xs.len().
    pub fn fwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }
}

type M = mint::M64<998244353>;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let mut adj = vec![0u32; n];
        for i in 0..n {
            let row = input.token().as_bytes();
            for j in 0..n {
                if row[j] == b'1' {
                    adj[i] |= 1 << j;
                }
            }
        }

        let mut indep = vec![false; 1 << n];
        indep[0] = true;
        for mask in 0..1 << n {
            if !indep[mask] {
                continue;
            }

            for j in 0..n {
                if adj[j] as usize & mask == 0 {
                    indep[mask | (1 << j)] = true;
                }
            }
        }
        let mut indep = indep
            .into_iter()
            .map(|x| M::from(x as u32))
            .collect::<Vec<_>>();
        dp_sos::subset_sums(&mut indep);

        const UNSET: u8 = u8::MAX;
        let mut pow_indep = indep.clone();
        let mut n_chrom = vec![UNSET; 1 << n];
        n_chrom[0] = 0;
        for i in 1..=n as u8 {
            let mut chrom_k = pow_indep.clone();
            dp_sos::inv_subset_sums(&mut chrom_k);
            for mask in 0..1 << n {
                if n_chrom[mask] == UNSET && chrom_k[mask] != M::zero() {
                    n_chrom[mask] = i;
                }
            }

            for (x, y) in pow_indep.iter_mut().zip(&indep) {
                *x *= y;
            }
        }

        let mut ans = 0u32;
        let mut pow = 1u32;
        for mask in 0..1 << n {
            ans = ans.wrapping_add((n_chrom[mask] as u32).wrapping_mul(pow));
            pow = pow.wrapping_mul(233);
        }

        writeln!(output, "{}", ans).unwrap();
    }
}
