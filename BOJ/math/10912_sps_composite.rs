use std::io::Write;

#[allow(unused)]
mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    unsafe extern "C" {
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

pub mod set_power_series {
    use crate::algebra::{CommRing, Field};
    use std::ops::*;

    pub trait CommGroup: Default + AddAssign + SubAssign + Clone {}
    impl<T: Default + AddAssign + SubAssign + Clone> CommGroup for T {}

    fn kronecker_prod<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
            let n = xs.len().ilog2() as usize;
            for b in 0..n {
                for t in xs.chunks_exact_mut(1 << b + 1) {
                    let (zs, os) = t.split_at_mut(1 << b);
                    zs.iter_mut().zip(os).for_each(|(z, o)| modifier(z, o));
                }
            }
        }
        unsafe { inner(xs, modifier) }
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
    pub fn fwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }
    pub fn ifwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }

    // Group terms of $\sum_{S \subset [n]} a_S x^S$ by $|S|$
    pub fn chop<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        let n = xs.len().ilog2() as usize;
        let mut res = vec![T::zero(); (n + 1) * (1 << n)];
        for (i, x) in xs.iter().cloned().enumerate() {
            res[((i.count_ones() as usize) << n) + i] = x;
        }
        res
    }
    pub fn unchop<T: CommRing>(n: usize, xs: &[T]) -> Vec<T> {
        assert_eq!(xs.len(), n + 1 << n);
        let mut res = vec![T::zero(); 1 << n];
        for i in 0..1 << n {
            res[i] = xs[((i.count_ones() as usize) << n) + i].clone();
        }
        res
    }
    pub fn subset_conv<T: CommRing>(xs: &[T], ys: &[T]) -> Vec<T> {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommRing>(xs: &[T], ys: &[T]) -> Vec<T> {
            assert!(xs.len().is_power_of_two());
            assert_eq!(xs.len(), ys.len());
            let n = xs.len().ilog2() as usize;

            let mut xs = chop(&xs);
            let mut ys = chop(&ys);
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
        unsafe { inner(xs, ys) }
    }
    fn subset_inv_inner<T: CommRing>(xs: &[T], inv_xs0: T) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        let n = xs.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        res[0] = inv_xs0;
        let mut w = 1;
        while w < 1 << n {
            let mut ext = subset_conv(&xs[w..w * 2], &subset_conv(&res[..w], &res[..w]));
            ext.iter_mut().for_each(|x| *x = -x.clone());
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn subset_inv1<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs[0] == T::one());
        subset_inv_inner(xs, T::one())
    }
    pub fn subset_inv<T: Field>(xs: &[T]) -> Vec<T> {
        assert!(xs[0] != T::zero());
        subset_inv_inner(xs, xs[0].inv())
    }
    pub fn subset_exp<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        assert!(xs[0] == T::zero());
        let n = xs.len().ilog2() as usize;

        let mut res = vec![T::one(); 1 << n];
        let mut w = 1;
        while w < 1 << n {
            let ext = subset_conv(&xs[w..w * 2], &res[..w]);
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn subset_ln<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        assert!(xs[0] == T::one());
        let n = xs.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        let mut w = 1;
        while w < 1 << n {
            let ext = subset_conv(&xs[w..w * 2], &subset_inv1(&xs[..w]));
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn subset_comp<T: CommRing + From<u32> + std::fmt::Debug>(f: &[T], xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        let n = xs.len().ilog2() as usize;

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
                    pow *= &xs[0];
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
                layers[c * w * 2 + w..][..w].clone_from_slice(&subset_conv(p1, &xs[w..w * 2]));
            }
        }
        layers
    }
    // todo: power projection
}

use algebra::SemiRing;

// type M = mint_mont::M32<998244353>;
type M = mint_mont::M32<1000000007>;

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let m0 = M::zero();
    let m1 = M::one();

    fn binom_small(n: u64, k: u64) -> M {
        let mut numer = M::one();
        let mut denom = M::one();
        for i in 0..k {
            numer *= M::from(n - i);
            denom *= M::from(i + 1);
        }
        numer / denom
    }

    let t: u64 = input.value();
    let n: usize = input.value();
    let a: M = input.value();

    let s: usize = 10;
    let mut trans = vec![m0; 1 << s];
    for _ in 0..n {
        let p = input.value::<M>() / a;
        let mut prod = vec![m1; 1 << s];
        for b in 0..s {
            let w = 1 << b;
            let c: M = input.value();
            let (ls, rs) = prod[..w * 2].split_at_mut(w);
            for (l, r) in ls.iter_mut().zip(rs) {
                *r = *l * c;
            }
        }
        for (x, y) in trans.iter_mut().zip(&mut prod) {
            *x += *y * p;
        }
    }
    trans[0] = m0;

    let mut ans;
    {
        // // O(N^3 2^N)
        // let mut trans_pow = vec![m1; 1 << s];
        // ans = m1;
        // for i in 1..=t.min(10) {
        //     trans_pow = set_power_series::subset_conv(&trans_pow, &trans);
        //     ans += trans_pow[(1 << s) - 1] * binom_small(t, i);
        // }
        // ans *= a.pow(t);
    }
    {
        // O(N^2 2^N)
        let mut f = vec![M::zero(); t.min(10) as usize + 1];
        for i in 0..f.len() {
            f[i] = binom_small(t, i as u64);
        }

        let p = set_power_series::subset_comp(&f, &trans);
        ans = p.iter().fold(M::zero(), |x, y| x + y);
        ans *= a.pow(t);
    }

    writeln!(output, "{}", ans).unwrap();
}
