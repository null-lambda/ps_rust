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
  ($($t:ty)+) => { $(impl Unsigned for $t { fn one() -> Self { 1 } })+ };
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
    ($($t:ty)+) => { $(impl SemiRing for $t { fn one() -> Self { 1 } })+ };
 }
    macro_rules! impl_commring { ($($t:ty)+) => { $(impl CommRing for $t {})+ }; }
    impl_semiring!(u8 u16 u32 u64 u128 usize);
    impl_semiring!(i8 i16 i32 i64 i128 isize);
    impl_commring!(i8 i16 i32 i64 i128 isize);
}

pub mod mint_mont {
    use crate::algebra::*;
    use std::ops::*;

    macro_rules! forward_binop {
        ($mint:ident $U:ty, $OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
            impl<const M: $U> $OpAssign for $mint<M> {
                fn $op_assign(&mut self, rhs: Self) {
                    self.$op_assign(&rhs);
                }
            }
            impl<const M: $U> $Op<&'_ Self> for $mint<M> {
                type Output = Self;
                fn $op(mut self, rhs: &Self) -> Self {
                    self.$op_assign(rhs);
                    self
                }
            }
            impl<const M: $U> $Op for $mint<M> {
                type Output = $mint<M>;
                fn $op(self, rhs: Self) -> Self::Output {
                    self.clone().$op(&rhs)
                }
            }
        };
    }

    macro_rules! impl_modint {
 ($mint:ident, U = $U:ty, D = $D:ty, EXP = $exp:expr) => {
  #[repr(transparent)]
  #[derive(Clone, Copy, PartialEq, Eq)]
  pub struct $mint<const M: $U>($U);
  impl<const M: $U> $mint<M> {
   pub const NEG_M_INV: $U = {
    assert!(M % 2 == 1 && M < 1 << 31, "invalid modulus");
    let mut m_inv: $U = 1;
    let two: $U = 2;

    let mut iter = 0;
    let log2_exp = u32::trailing_zeros($exp);
    while iter < log2_exp {
     m_inv = m_inv.wrapping_mul(two.wrapping_sub(M.wrapping_mul(m_inv)));
     iter += 1;
    }
    m_inv.wrapping_neg()
   };
   pub const R2: $U = {
    let r = M.wrapping_neg() % M;
    (r as $D * r as $D % M as $D) as $U
   };
   pub const fn reduce(x: $D) -> $U {
    debug_assert!(x < (M as $D) * (M as $D));
    let q = (x as $U).wrapping_mul(Self::NEG_M_INV);
    let mut res = ((q as $D * M as $D + x) >> $exp) as $U;
    if res >= M {
     res -= M;
    }
    res
   }
   pub const fn new(x: $U) -> Self {
    Self(Self::reduce(x as $D * Self::R2 as $D))
   }
   pub const fn into_unsigned(self) -> $U {
    Self::reduce(self.0 as $D)
   }
  }

  impl<const M: $U> AddAssign<&'_ Self> for $mint<M> { fn add_assign(&mut self, rhs: &Self) { self.0 += rhs.0; if self.0 >= M { self.0 -= M; } } }
  impl<const M: $U> SubAssign<&'_ Self> for $mint<M> { fn sub_assign(&mut self, rhs: &Self) { if self.0 < rhs.0 { self.0 += M; } self.0 -= rhs.0; } }
  impl<const M: $U> MulAssign<&'_ Self> for $mint<M> { fn mul_assign(&mut self, rhs: &Self) { self.0 = Self::reduce(self.0 as $D * rhs.0 as $D); } }
  impl<const M: $U> DivAssign<&'_ Self> for $mint<M> { fn div_assign(&mut self, rhs: &Self) { self.mul_assign(&rhs.inv()); } }
  forward_binop!($mint $U, AddAssign add_assign, Add add);
  forward_binop!($mint $U, SubAssign sub_assign, Sub sub);
  forward_binop!($mint $U, MulAssign mul_assign, Mul mul);
  forward_binop!($mint $U, DivAssign div_assign, Div div);

  impl<const M: $U> Neg for &'_ $mint<M> { type Output = $mint<M>; fn neg(self) -> $mint<M> { let mut res = M - self.0; if res == M { res = 0; } $mint(res) } }
  impl<const M: $U> Neg for $mint<M> { type Output = Self; fn neg(self) -> Self::Output { (&self).neg() } }
  impl<const M: $U> Default for $mint<M> { fn default() -> Self { Self::new(0) } }
  impl<const M: $U> SemiRing for $mint<M> { fn one() -> Self { Self::new(1) } }
  impl<const M: $U> CommRing for $mint<M> {}
  impl<const M: $U> Field for $mint<M> { fn inv(&self) -> Self { self.pow(M - 2) } }
  impl<const M: $U> std::fmt::Debug for $mint<M> { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.into_unsigned()) } }
  impl<const M: $U> std::fmt::Display for $mint<M> { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.into_unsigned()) } }
  impl<const M: $U> std::str::FromStr for $mint<M> { type Err = <$U as std::str::FromStr>::Err; fn from_str(s: &str) -> Result<Self, Self::Err> { s.parse::<$U>().map(|x| $mint::new(x)) } }
 };
 }
    impl_modint!(M32, U = u32, D = u64, EXP = 32);
    impl_modint!(M64, U = u64, D = u128, EXP = 64);

    macro_rules! impl_ucast {
  (@eq $mint:ident $U:ident) => {
   impl<const M: $U> From<$U> for $mint<M> { fn from(x: $U) -> Self { Self::new(x) } }
   impl<const M: $U> From<$mint<M>> for $U { fn from(x: $mint<M>) -> Self { x.into_unsigned() } }
  };
  (@lt $mint:ident $U:ident $src:ident) => {
   impl<const M: $U> From<$src> for $mint<M> { fn from(x: $src) -> Self { Self::new(x as $U) } }
  };
  (@gt $mint:ident $U:ident $src:ident) => {
   impl<const M: $U> From<$src> for $mint<M> { fn from(x: $src) -> Self { Self::new((x % M as $src) as $U) } }
   impl<const M: $U> From<$mint<M>> for $src { fn from(x: $mint<M>) -> Self { x.into_unsigned() as $src } }
  };
  ($mint:ident, $($lower:ident)*, $U:ident, $($upper:ident)*) => {
   $(impl_ucast!(@lt $mint $U $lower);)*
   impl_ucast!(@eq $mint $U);
   $(impl_ucast!(@gt $mint $U $upper);)*
  };
  () => {};
 }
    macro_rules! impl_icast {
  (@lt $mint:ident $U:ident $I:ident $src:ident) => {
   impl<const M: $U> From<$src> for $mint<M> { fn from(x: $src) -> Self { Self::new((x as $I).rem_euclid(M as $I) as $U) } }
  };
  (@ge $mint:ident $U:ident $src:ident) => {
   impl<const M: $U> From<$src> for $mint<M> { fn from(x: $src) -> Self { Self::new((x.rem_euclid(M as $src)) as $U) } }
  };
  ($mint:ident, $($lower:ident)*, $U:ident $I:ident, $($upper:ident)*) => {
   $(impl_icast!(@lt $mint $U $I $lower);)*
   impl_icast!(@ge $mint $U $I);
   $(impl_icast!(@ge $mint $U $upper);)*
  };
  () => {};
 }
    impl_ucast!(M32, u8 u16, u32, u64 u128);
    impl_ucast!(M64, u8 u16 u32, u64, u128);
    impl_icast!(M32, i8 i16, u32 i32, i64 i128);
    impl_icast!(M64, i8 i16 i32, u64 i64, i128);
}

#[rustfmt::skip]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod mint_mont_simd {
 #[cfg(target_arch = "x86")]
 use core::arch::x86::*;
 #[cfg(target_arch = "x86_64")]
 use core::arch::x86_64::*;

 use crate::{algebra::*, mint_mont::*};

 #[repr(transparent)]
 #[derive(Clone, Copy)]
 pub struct M32x8<const M: u32>(pub __m256i);

 impl<const M: u32> M32x8<M> {
  #[target_feature(enable = "avx")]
  pub unsafe fn loadu(ptr: *const M32<M>) -> Self { unsafe { Self(_mm256_loadu_si256(ptr as *const _)) } }
  #[target_feature(enable = "avx")]
  pub unsafe fn storeu(&self, ptr: *mut M32<M>) { unsafe { _mm256_storeu_si256(ptr as *mut _, self.0) } }
  #[target_feature(enable = "avx")]
  pub unsafe fn msplat(x: u32) -> __m256i { _mm256_set1_epi32(x as i32) }
  #[target_feature(enable = "avx")]
  pub unsafe fn splat(x: M32<M>) -> Self { unsafe { Self(Self::msplat(std::mem::transmute(x))) } }
  #[target_feature(enable = "avx")]
  pub unsafe fn from_array(xs: [M32<M>; 8]) -> Self {
   unsafe {
    let xs: [i32; 8] = std::mem::transmute(xs);
    Self(_mm256_setr_epi32(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]))
   }
  }
  #[target_feature(enable = "avx")]
  pub unsafe fn mod_x8() -> __m256i { unsafe { Self::msplat(M) } }
  #[target_feature(enable = "avx")]
  pub unsafe fn neg_m_inv() -> __m256i { unsafe { Self::msplat(M32::<M>::NEG_M_INV) } }
  #[target_feature(enable = "avx")]
  pub unsafe fn r2() -> __m256i { unsafe { Self::msplat(M32::<M>::R2) } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn shrink(x: __m256i) -> __m256i { unsafe { _mm256_min_epu32(x, _mm256_sub_epi32(x, Self::mod_x8())) } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn recover(x: __m256i) -> __m256i { unsafe { _mm256_min_epu32(x, _mm256_add_epi32(x, Self::mod_x8())) } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn reduce_x8(x0246: __m256i, x1357: __m256i) -> __m256i {
   unsafe {
    let z0246 = _mm256_mul_epu32(x0246, Self::neg_m_inv());
    let z1357 = _mm256_mul_epu32(x1357, Self::neg_m_inv());
    let z0246 = _mm256_add_epi64(x0246, _mm256_mul_epu32(z0246, Self::mod_x8()));
    let z1357 = _mm256_add_epi64(x1357, _mm256_mul_epu32(z1357, Self::mod_x8()));
    let z = _mm256_or_si256(_mm256_bsrli_epi128::<4>(z0246), z1357);
    Self::shrink(z)
   }
  }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn add(self, other: Self) -> Self { unsafe { Self(Self::shrink(_mm256_add_epi32(self.0, other.0))) } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn sub(self, other: Self) -> Self { unsafe { Self(Self::recover(_mm256_sub_epi32(self.0, other.0))) } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn neg(self) -> Self { unsafe { Self(Self::shrink(Self(Self::mod_x8()).sub(self).0)) } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn mul(self, other: Self) -> Self {
   unsafe {
    let x0246 = _mm256_mul_epu32(self.0, other.0);
    let x1357 = _mm256_mul_epu32(_mm256_bsrli_epi128::<4>(self.0), _mm256_bsrli_epi128::<4>(other.0));
    Self(Self::reduce_x8(x0246, x1357))
   }
  }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn loadu_vec(xs: Vec<M32<M>>) -> Vec<Self> { unsafe { xs.chunks_exact(8).map(|t| M32x8::loadu(t.as_ptr())).collect() } }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn storeu_vec(xs: Vec<Self>) -> Vec<M32<M>> {
   let mut res = vec![M32::zero(); xs.len() * 8];
   unsafe {
    for (v, x) in res.chunks_exact_mut(8).zip(&xs) {
     x.storeu(v.as_mut_ptr())
    }
   }
   res
  }
 }
}

pub mod sps {
    // Variants of R[x_1, ..., x_n]/(q(x_1), ..., q(x_n)),
    // including Set Power Series (q(x) = x^2)
    use crate::algebra::{CommRing, Field};
    use std::ops::*;

    // LOJ submission (rust ed 2024 when??????)
    trait ILOG2 {
        fn ilog2(self) -> u32;
    }
    impl ILOG2 for usize {
        fn ilog2(self) -> u32 {
            usize::BITS - usize::leading_zeros(self) - 1
        }
    }

    pub trait CommGroup: Default + AddAssign + SubAssign + Clone {}
    impl<T: Default + AddAssign + SubAssign + Clone> CommGroup for T {}

    fn kronecker_prod<T: CommGroup>(f: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommGroup>(f: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
            let n = f.len().ilog2() as usize;
            for b in 0..n {
                for t in f.chunks_exact_mut(1 << b + 1) {
                    let (zs, os) = t.split_at_mut(1 << b);
                    zs.iter_mut().zip(os).for_each(|(z, o)| modifier(z, o));
                }
            }
        }
        unsafe { inner(f, modifier) }
    }
    pub fn subset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *o += z.clone());
    }
    pub fn inv_subset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *o -= z.clone());
    }
    pub fn superset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *z += o.clone());
    }
    pub fn inv_superset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *z -= o.clone());
    }
    pub fn fwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }
    pub fn ifwht<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }

    // Group terms of $\sum_{S \subset [n]} f_S x^S$ by $|S|$
    pub fn chop<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        let n = f.len().ilog2() as usize;
        let mut res = vec![T::zero(); (n + 1) * (1 << n)];
        for (i, x) in f.iter().cloned().enumerate() {
            res[((i.count_ones() as usize) << n) + i] = x;
        }
        res
    }
    pub fn unchop<T: CommRing>(n: usize, f: &[T]) -> Vec<T> {
        assert_eq!(f.len(), n + 1 << n);
        let mut res = vec![T::zero(); 1 << n];
        for i in 0..1 << n {
            res[i] = f[((i.count_ones() as usize) << n) + i].clone();
        }
        res
    }
    pub fn conv<T: CommRing>(f: &[T], g: &[T]) -> Vec<T> {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommRing>(f: &[T], g: &[T]) -> Vec<T> {
            assert!(f.len().is_power_of_two());
            assert_eq!(f.len(), g.len());
            let n = f.len().ilog2() as usize;

            let mut xs = chop(&f);
            let mut ys = chop(&g);
            xs.chunks_exact_mut(1 << n).for_each(|bs| subset_sums(bs));
            ys.chunks_exact_mut(1 << n).for_each(|bs| subset_sums(bs));

            let mut zs = vec![T::default(); 1 << n];
            for k in 0..=n {
                let mut zr = vec![T::default(); 1 << n];
                for i in 0..=k {
                    let xr = &xs[i << n..][..1 << n];
                    let yr = &ys[k - i << n..][..1 << n];
                    for ((x, y), z) in xr.iter().zip(yr).zip(&mut zr) {
                        *z += x.clone() * y.clone();
                    }
                }
                inv_subset_sums(&mut zr);
                for (i, z) in zr.into_iter().enumerate() {
                    if i.count_ones() == k as u32 {
                        zs[i] += z;
                    }
                }
            }
            zs
        }
        unsafe { inner(f, g) }
    }
    fn inv_inner<T: CommRing>(f: &[T], inv_f0: T) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        let n = f.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        res[0] = inv_f0;
        let mut w = 1;
        while w < 1 << n {
            let mut ext = conv(&f[w..w * 2], &conv(&res[..w], &res[..w]));
            ext.iter_mut().for_each(|x| *x = -x.clone());
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn inv1<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f[0] == T::one());
        inv_inner(f, T::one())
    }
    pub fn inv<T: Field>(f: &[T]) -> Vec<T> {
        assert!(f[0] != T::zero());
        inv_inner(f, f[0].inv())
    }
    pub fn exp<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        assert!(f[0] == T::zero());
        let n = f.len().ilog2() as usize;

        let mut res = vec![T::one(); 1 << n];
        let mut w = 1;
        while w < 1 << n {
            let ext = conv(&f[w..w * 2], &res[..w]);
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn ln<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        assert!(f[0] == T::one());
        let n = f.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        let mut w = 1;
        let f_inv = inv1(&f[..1 << n - 1]);
        while w < 1 << n {
            let ext = conv(&f[w..w * 2], &f_inv[..w]);
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn comp<T: CommRing + From<u32>>(f: &[T], g: &[T]) -> Vec<T> {
        assert!(g.len().is_power_of_two());
        let n = g.len().ilog2() as usize;

        if f.is_empty() {
            return vec![T::zero(); 1 << n];
        }

        // $(d^k f) \circ xs$
        let mut layers = vec![T::zero(); (n + 1) * 1];
        {
            let mut dk_f = f.to_vec();
            for k in 0..n + 1 {
                if k >= 1 {
                    dk_f = dk_f.into_iter().skip(1).collect();
                    for i in 0..dk_f.len() {
                        dk_f[i] *= T::from(i as u32 + 1);
                    }
                }
                let mut pow = T::one();
                for i in 0..dk_f.len() {
                    layers[k] += dk_f[i].clone() * pow.clone();
                    pow *= &g[0];
                }
            }
        }
        for b in 1..=n {
            let w = 1 << b - 1;
            let prev = layers;
            layers = vec![T::zero(); (n - b + 1) * (w * 2)];
            for c in 0..=n - b {
                let p0 = &prev[c * w..][..w];
                let p1 = &prev[c * w + w..][..w];
                layers[c * w * 2..][..w].clone_from_slice(p0);
                layers[c * w * 2 + w..][..w].clone_from_slice(&conv(p1, &g[w..w * 2]));
            }
        }
        layers
    }
    // $[x_i^e] f(x_1, .., x_n)$.
    // In particular, $e=0$ gives $f(x_i = 0)$ and $e=1$ gives $\partial_i f$.
    pub fn extract_axis<T: Clone, const E: usize>(f: &[T], i: usize) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        let n = f.len().ilog2() as usize;
        assert!(i < n);
        (f.chunks_exact(1 << i).skip(E).step_by(2).flatten())
            .cloned()
            .collect()
    }
    // overwrite [x_i^e] f
    pub fn overwrite_axis<T: Clone, const E: usize>(f: &mut [T], sub: &[T], i: usize) {
        assert!(f.len().is_power_of_two());
        assert!(f.len() == sub.len() * 2);
        let n = f.len().ilog2() as usize;
        assert!(i < n);
        (f.chunks_exact_mut(1 << i).skip(E).step_by(2).flatten())
            .zip(sub)
            .for_each(|(x, y)| *x = y.clone());
    }
    // todo: power projection
}

use algebra::SemiRing;

type M = mint_mont::M32<998244353>;
// type M = mint_mont::M32<1000000007>;

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u8>() - 1;
        let v = input.value::<u8>() - 1;
        edges.push((1 << u) | (1 << v));
    }

    let mut es = vec![0u32; 1 << n];
    for s in 0..1 << n {
        for &e in &edges {
            if s & e == e {
                es[s] += 1;
            }
        }
    }

    let mut pow2 = vec![M::one(); n * (n - 1) / 2 + 1];
    for i in 1..pow2.len() {
        pow2[i] = pow2[i - 1] + pow2[i - 1];
    }

    let mut graph = vec![M::zero(); 1 << n];
    for i in 0..1 << n {
        graph[i] = pow2[es[i] as usize];
    }

    let conn = sps::ln(&graph);

    let mut two_vc = conn;
    if n >= 2 {
        for i in 0..n {
            let t1 = sps::extract_axis::<_, 1>(&two_vc, i);
            sps::overwrite_axis::<_, 1>(&mut two_vc, &sps::ln(&t1), i);
        }
    }

    let mut two_ec = two_vc;
    for i in 0..n {
        for j in i + 1..n {
            let s = (1 << i) | (1 << j);
            two_ec[s] = M::zero();
        }
    }

    if n >= 2 {
        for i in 0..n {
            let t1 = sps::extract_axis::<_, 1>(&two_ec, i);
            sps::overwrite_axis::<_, 1>(&mut two_ec, &sps::exp(&t1), i);
        }
    }

    let ans = two_ec[(1 << n) - 1];
    writeln!(output, "{}", ans).unwrap();
}
