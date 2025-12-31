#[rustfmt::skip]
pub mod algebra {
 use std::ops::*;
 pub trait Unsigned: Copy + Default + SemiRing + Div<Output = Self> + Rem<Output = Self> + RemAssign + Eq + Ord + From<u8> {
  fn zero() -> Self { Self::default() }
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
  fn zero() -> Self { Self::default() }
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
 pub trait Field: CommRing + Div<Output = Self> + DivAssign + for<'a> Div<&'a Self, Output = Self> + for<'a> DivAssign<&'a Self> {
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

#[rustfmt::skip]
pub mod mint_mont {
 use std::ops::*;
 use crate::algebra::*;

 macro_rules! forward_binop {
  ($mint:ident $U:ty, $OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
   impl<const M: $U> $OpAssign for $mint<M> {
    fn $op_assign(&mut self, rhs: Self) { self.$op_assign(&rhs); }
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
    fn $op(self, rhs: Self) -> Self::Output { self.clone().$op(&rhs) }
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

#[rustfmt::skip]
pub mod conv {
 use super::algebra::*;
 use crate::mint_mont::M32;
 const P_GENS_32: [[u32; 2]; 4] = [[998244353, 3], [167772161, 3], [104857601, 3], [13631489, 15]];
 pub const fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
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
 pub const fn crt3_coeff_u32(ps: [u32; 3]) -> (u128, [u128; 3]) {
  let ps = [ps[0] as u64, ps[1] as u64, ps[2] as u64];
  let q = ps[0] as u128 * ps[1] as u128 * ps[2] as u128;
  let qs = [ps[1] * ps[2], ps[0] * ps[2], ps[0] * ps[1]];
  let rs = [mod_pow(qs[0] % ps[0], ps[0] - 2, ps[0]), mod_pow(qs[1] % ps[1], ps[1] - 2, ps[1]), mod_pow(qs[2] % ps[2], ps[2] - 2, ps[2])];
  let ms = [qs[0] as u128 * rs[0] as u128, qs[1] as u128 * rs[1] as u128, qs[2] as u128 * rs[2] as u128];
  (q, ms)
 }

 pub struct PreCalc<T> {
  rs: [T; 30],
  irs: [T; 30],
 }
 impl<T: Field> PreCalc<T> {
  // fn new(n_log: usize, proot: T) -> Self {
  //     let mut pow = vec![T::one(); 31];
  //     let mut ipow = vec![T::one(); 31];
  //     pow[n_log] = proot.clone();
  //     ipow[n_log] = proot.clone().inv();
  //     for i in (0..n_log).rev() {
  //         pow[i] = pow[i + 1].clone() * pow[i + 1].clone();
  //         ipow[i] = ipow[i + 1].clone() * ipow[i + 1].clone();
  //     }

  //     let mut rs = std::array::from_fn(|_| T::one());
  //     let mut irs = std::array::from_fn(|_| T::one());
  //     let mut p = T::one();
  //     let mut ip = T::one();
  //     for i in 0..n_log - 1 {
  //         rs[i] = pow[i + 2].clone() * p.clone();
  //         irs[i] = ipow[i + 2].clone() * ip.clone();
  //         p *= ipow[i + 2].clone();
  //         ip *= pow[i + 2].clone();
  //     }
  //     Self { rs, irs }
  // }
 }
 impl<const P: u32> PreCalc<crate::mint_mont::M32<P>> {
  const INST: Option<Self> = 'o: {
   let p64 = P as u64;
   let mut pow = [1u64; 31];
   let mut ipow = [1u64; 31];
   let v2 = (P - 1).trailing_zeros() as usize;
   let Some(proot) = try_gen_32(P) else {
    break 'o None;
   };
   let base = mod_pow(proot as u64, (p64 - 1) >> v2, p64);
   pow[v2] = base;
   ipow[v2] = mod_pow(base, p64 - 2, p64);
   let mut i = v2;
   while i > 0 {
    i -= 1;
    pow[i] = pow[i + 1] * pow[i + 1] as u64 % p64;
    ipow[i] = ipow[i + 1] * ipow[i + 1] as u64 % p64;
   }

   let mut rs = [1u64; 30];
   let mut irs = [1u64; 30];
   let mut w = 1;
   let mut iw = 1;
   let mut i = 0;
   while i < v2 - 1 {
    rs[i] = pow[i + 2] * w % p64;
    irs[i] = ipow[i + 2] * iw % p64;
    w = w * ipow[i + 2] % p64;
    iw = iw * pow[i + 2] % p64;
    i += 1;
   }
   const fn trans<const P: u32>(xs: [u64; 30]) -> [M32<P>; 30] {
    let mut res = [M32::new(0); 30];
    let mut i = 0;
    while i < 30 {
     res[i] = M32::new(xs[i] as u32);
     i += 1;
    }
    res
   }
   let rs = trans(rs);
   let irs = trans(irs);
   Some(Self { rs, irs })
  };
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
 pub fn bf2<T: CommRing, const TRANS: bool>(x1: &mut T, x2: &mut T, w: T) {
  if !TRANS {
   let y1 = x1.clone();
   let y2 = x2.clone() * w;
   *x1 = y1.clone() + y2.clone();
   *x2 = y1 - y2;
  } else {
   let y1 = x1.clone();
   let y2 = x2.clone();
   *x1 = y1.clone() + y2.clone();
   *x2 = (y1 - y2) * w;
  }
 }
 fn chunks2<T>(xs: &mut [T], w: usize) -> impl Iterator<Item = (&mut [T], &mut [T])> { xs.chunks_exact_mut(w * 2).map(move |t| t.split_at_mut(w)) }
 pub fn ntt_radix2<T: Field>(cx: &PreCalc<T>, xs: &mut [T]) {
  let n = xs.len();
  assert!(n.is_power_of_two());
  let n_log = u32::BITS - (n as u32).leading_zeros() - 1;
  for e in (0..n_log).rev() {
   let mut w = T::one();
   for (it, (t0, t1)) in chunks2(xs, 1 << e).enumerate() {
    (t0.iter_mut().zip(t1)).for_each(|(x0, x1)| bf2::<_, false>(x0, x1, w.clone()));
    w *= cx.rs[it.trailing_ones() as usize].clone();
   }
  }
 }
 pub fn intt_radix2<T: Field>(cx: &PreCalc<T>, xs: &mut [T]) {
  let n = xs.len();
  assert!(n.is_power_of_two());
  let n_log = n.trailing_zeros();
  for e in 0..n_log {
   let mut w = T::one();
   for (it, (t0, t1)) in chunks2(xs, 1 << e).enumerate() {
    (t0.iter_mut().zip(t1)).for_each(|(x0, x1)| bf2::<_, true>(x0, x1, w.clone()));
    w *= cx.irs[it.trailing_ones() as usize].clone();
   }
  }
 }
 fn conv_naive<T: SemiRing>(mut lhs: Vec<T>, mut rhs: Vec<T>) -> Vec<T> {
  if lhs.len() == 0 || rhs.len() == 0 {
   lhs.clear();
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
 fn conv_with_proot_32<const P: u32>(mut lhs: Vec<M32<P>>, mut rhs: Vec<M32<P>>) -> Result<Vec<M32<P>>, (Vec<M32<P>>, Vec<M32<P>>)> {
  if lhs.len().min(rhs.len()) <= 20 {
   return Ok(conv_naive(lhs, rhs));
  }
  let n = lhs.len() + rhs.len() - 1;
  let n_pad = n.next_power_of_two();
  if (P - 1) % n_pad as u32 != 0 {
   return Err((lhs, rhs));
  }
  let Some(cx) = &PreCalc::<M32<P>>::INST else {
   return Err((lhs, rhs));
  };
  // let proot = match try_proot_m32(n_pad) {
  //     Some(proot) => proot,
  //     None => return Err((lhs, rhs)),
  // };
  // let n_log = (u32::BITS - (n_pad as u32).leading_zeros() - 1) as usize;
  // let cx = PreCalc::new(n_log, proot.clone());

  lhs.resize(n_pad, M32::zero());
  rhs.resize(n_pad, M32::zero());
  ntt_radix2(&cx, &mut lhs);
  ntt_radix2(&cx, &mut rhs);
  lhs.iter_mut().zip(&rhs).for_each(|(x, y)| *x *= y);
  intt_radix2(&cx, &mut lhs);
  let n_inv = M32::from(n_pad as u32).inv();
  lhs.iter_mut().for_each(|x| *x *= n_inv.clone());
  lhs.truncate(n);
  Ok(lhs)
 }

 fn conv_with_crt_32<T: SemiRing + Into<u32> + From<u128>>(lhs: Vec<T>, rhs: Vec<T>) -> Vec<T> {
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
   conv_with_proot_32(lhs, rhs).unwrap()
  }
  let h0: Vec<M<P0>> = forward(&lhs, &rhs);
  let h1: Vec<M<P1>> = forward(&lhs, &rhs);
  let h2: Vec<M<P2>> = forward(&lhs, &rhs);
  (h0.into_iter().zip(h1).zip(h2))
   .map(|((x0, x1), x2)| {
    let a = MS[0] * u32::from(x0) as u128 + MS[1] * u32::from(x1) as u128 + MS[2] * u32::from(x2) as u128;
    T::from(a % Q)
   })
   .collect()
 }

 #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
 pub mod simd {
  #[cfg(target_arch = "x86")]
  use core::arch::x86::*;
  #[cfg(target_arch = "x86_64")]
  use core::arch::x86_64::*;

  use super::*;
  use crate::{algebra::SemiRing, mint_mont_simd::M32x8};

  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn ntt_radix2_32x8<const P: u32>(cx: &PreCalc<M32<P>>, xs: &mut [M32x8<P>]) {
   let n = xs.len() * 8;
   assert!(n.is_power_of_two());
   let n_log = n.trailing_zeros();
   unsafe {
    for e in (3..n_log).rev() {
     let mut w = M32::one();
     for (it, (t0, t1)) in chunks2(xs, 1 << e - 3).enumerate() {
      for (x0, x1) in t0.iter_mut().zip(t1) {
       let y0 = *x0;
       let y1 = x1.mul(M32x8::splat(w));
       *x0 = y0.add(y1);
       *x1 = y0.sub(y1);
      }
      w *= cx.rs[it.trailing_ones() as usize];
     }
    }
    let r0 = cx.rs[0];
    let r1 = cx.rs[1];
    let ra = r0;
    let rb = ra * r1;
    let rc = rb * r0;
    let m1 = M32::one();
    let mut w2 = M32x8::from_array([m1, m1, m1, m1, -m1, -m1, -m1, -m1]);
    let mut w1 = M32x8::from_array([m1, m1, -m1, -m1, ra, ra, -ra, -ra]);
    let mut w0 = M32x8::from_array([m1, -m1, ra, -ra, rb, -rb, rc, -rc]);
    for (it, x) in xs.iter_mut().enumerate() {
     let l = M32x8(_mm256_permute2x128_si256(x.0, x.0, 0));
     let r = M32x8(_mm256_permute2x128_si256(x.0, x.0, 0x11));
     *x = l.add(r.mul(w2));
     let l = M32x8(_mm256_permute4x64_epi64(x.0, 0xA0));
     let r = M32x8(_mm256_permute4x64_epi64(x.0, 0xF5));
     *x = l.add(r.mul(w1));
     let l = M32x8(_mm256_shuffle_epi32(x.0, 0xA0));
     let r = M32x8(_mm256_shuffle_epi32(x.0, 0xF5));
     *x = l.add(r.mul(w0));
     let t1 = it.trailing_ones() as usize;
     w2 = w2.mul(M32x8::splat(cx.rs[t1]));
     w1 = w1.mul(M32x8::splat(ra * cx.rs[t1 + 1]));
     w0 = w0.mul(M32x8::splat(rc * cx.rs[t1 + 2]));
    }
   }
  }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn intt_radix2_32x8<const P: u32>(cx: &PreCalc<M32<P>>, xs: &mut [M32x8<P>]) {
   let n = xs.len() * 8;
   assert!(n.is_power_of_two());
   let n_log = n.trailing_zeros();
   unsafe {
    let r0 = cx.irs[0];
    let r1 = cx.irs[1];
    let ra = r0;
    let rb = ra * r1;
    let rc = rb * r0;
    let m1 = M32::one();
    let mut w0 = M32x8::from_array([m1, m1, m1, ra, m1, rb, m1, rc]);
    let mut w1 = M32x8::from_array([m1, m1, m1, m1, m1, m1, ra, ra]);
    let mut w2 = M32x8::splat(m1);
    for (it, x) in xs.iter_mut().enumerate() {
     let l = M32x8(_mm256_shuffle_epi32(x.0, 0xA0));
     let r = M32x8(_mm256_shuffle_epi32(x.0, 0xF5));
     *x = M32x8(_mm256_blend_epi32(l.add(r).0, l.sub(r).mul(w0).0, 0xAA));
     let l = M32x8(_mm256_permute4x64_epi64(x.0, 0xA0));
     let r = M32x8(_mm256_permute4x64_epi64(x.0, 0xF5));
     *x = M32x8(_mm256_blend_epi32(l.add(r).0, l.sub(r).mul(w1).0, 0xCC));
     let l = M32x8(_mm256_permute2x128_si256(x.0, x.0, 0));
     let r = M32x8(_mm256_permute2x128_si256(x.0, x.0, 0x11));
     *x = M32x8(_mm256_blend_epi32(l.add(r).0, l.sub(r).mul(w2).0, 0xF0));
     let t1 = it.trailing_ones() as usize;
     w0 = w0.mul(M32x8::splat(rc * cx.irs[t1 + 2]));
     w1 = w1.mul(M32x8::splat(ra * cx.irs[t1 + 1]));
     w2 = w2.mul(M32x8::splat(cx.irs[t1]));
    }
    for e in 3..n_log {
     let mut w = M32::one();
     for (it, (t0, t1)) in chunks2(xs, 1 << e - 3).enumerate() {
      for (x0, x1) in t0.iter_mut().zip(t1) {
       let y0 = x0.clone();
       let y1 = x1.clone();
       *x0 = y0.add(y1);
       *x1 = y0.sub(y1).mul(M32x8::splat(w));
      }
      w *= cx.irs[it.trailing_ones() as usize];
     }
    }
   }
  }
  #[target_feature(enable = "avx", enable = "avx2")]
  pub unsafe fn conv_with_proot_32_avx2<const P: u32>(mut lhs: Vec<M32<P>>, mut rhs: Vec<M32<P>>) -> Result<Vec<M32<P>>, (Vec<M32<P>>, Vec<M32<P>>)> {
   if lhs.len().min(rhs.len()) <= 20 {
    return Ok(conv_naive(lhs, rhs));
   }
   let n = lhs.len() + rhs.len() - 1;
   let n_pad = n.next_power_of_two();
   if (P - 1) % n_pad as u32 != 0 {
    return Err((lhs, rhs));
   }
   let Some(cx) = &PreCalc::<M32<P>>::INST else {
    return Err((lhs, rhs));
   };
   unsafe {
    lhs.resize(n_pad, M32::zero());
    rhs.resize(n_pad, M32::zero());
    let mut lhs = M32x8::loadu_vec(lhs);
    let mut rhs = M32x8::loadu_vec(rhs);
    ntt_radix2_32x8(&cx, &mut lhs);
    ntt_radix2_32x8(&cx, &mut rhs);
    lhs.iter_mut().zip(&rhs).for_each(|(x, y)| *x = x.mul(*y));
    intt_radix2_32x8(&cx, &mut lhs);
    let n_inv = M32x8::splat(M32::from(n_pad as u32).inv());
    lhs.iter_mut().for_each(|x| *x = x.mul(n_inv));
    let mut lhs = M32x8::storeu_vec(lhs);
    lhs.truncate(n);
    Ok(lhs)
   }
  }
 }

 pub trait Conv: SemiRing {
  fn conv(lhs: Vec<Self>, rhs: Vec<Self>) -> Vec<Self>;
 }
 impl<const P: u32> Conv for crate::mint_mont::M32<P> {
  fn conv(lhs: Vec<Self>, rhs: Vec<Self>) -> Vec<Self> {
   #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
   let res = conv_with_proot_32(lhs, rhs);
   #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
   let res = if std::is_x86_feature_detected!("avx2") { unsafe { simd::conv_with_proot_32_avx2(lhs, rhs) } } else { conv_with_proot_32(lhs, rhs) };
   match res {
    Ok(res) => res,
    Err((lhs, rhs)) => conv_with_crt_32(lhs, rhs),
   }
  }
 }
}

#[rustfmt::skip]
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
  pub fn inv(&self, i: usize) -> T { self.ifc[i].clone() * self.fc[i - 1].clone() }
  pub fn binom(&self, n: usize, k: usize) -> T { self.fc[n].clone() * self.ifc[k].clone() * self.ifc[n - k].clone() }
 }
}

pub mod poly {
    use crate::{algebra::*, comb::Comb, conv::Conv};
    use std::{collections::VecDeque, ops::*};

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
    impl<T: CommRing + From<u32>> Poly<T> {
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
    impl<T: Conv> Mul<Self> for Poly<T> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(Conv::conv(self.0, rhs.0))
        }
    }
    impl<T: Conv> MulAssign<Self> for Poly<T> {
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
    impl<T: Conv + CommRing + From<u32>> Poly<T> {
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
    }
    impl<T: Conv + Field + From<u32>> Poly<T> {
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
        pub fn integrate(&self, cx: &Comb<T>) -> Self {
            std::iter::once(T::zero())
                .chain(self.0.iter().enumerate().map(|(i, x)| cx.inv(i + 1) * x))
                .collect()
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
        fn resize_chunks(&mut self, w_src: usize, w_dest: usize) {
            let mut res = Poly::new(vec![]);
            if w_src <= w_dest {
                for r in self.0.chunks(w_src) {
                    res.0.extend(r.iter().cloned());
                    res.0.extend((0..w_dest - r.len()).map(|_| T::zero()))
                }
            } else {
                for r in self.0.chunks(w_src) {
                    res.0.extend(r.iter().cloned().take(w_dest));
                }
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
            let mut w = 2;
            let mut p = Poly::new(vec![T::zero(); (n + 1) * w]);
            let mut q = Poly::new(vec![T::zero(); (n + 1) * w]);
            for i in 0..(n + 1).min(g.0.len()) {
                p.0[i * w + 0] = g.0[i].clone();
            }
            q.0[0 * w + 0] = T::one();
            for i in 0..(n + 1).min(f.0.len()) {
                q.0[i * w + 1] = -f.0[i].clone();
            }
            while nc > 0 {
                let w_prev = w;
                w = w_prev * 2 - 1;
                p.resize_chunks(w_prev, w);
                q.resize_chunks(w_prev, w);
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
        // \sum_{i=0...m-d} y^i [y^{d+i}] p(y)/q(x,y) mod x^k
        fn comp_rec(p: Self, mut q: Self, qw: usize, k: usize, d: usize, m: usize) -> Self {
            if k == 1 {
                let u = p * q.inv_mod_xk(m);
                return Poly::new(u.0[d.min(u.0.len())..m.min(u.0.len())].to_vec());
            }
            let qw_next = qw * 2 - 1;
            q.resize_chunks(qw, qw_next);
            let mut q_nx = q.clone();
            for r in q_nx.0.chunks_mut(qw_next).skip(1).step_by(2) {
                for x in r {
                    *x = -x.clone();
                }
            }
            let mut v = q * q_nx.clone();
            v =
                v.0.chunks(qw_next)
                    .step_by(2)
                    .take((k + 1) / 2)
                    .flatten()
                    .cloned()
                    .collect();
            let e = (d + 1).saturating_sub(qw);
            let mut b = Self::comp_rec(p, v, qw_next, (k + 1) / 2, e, m);
            let bw = m - e;
            let cw = bw + qw - 1;
            b.resize_chunks(bw, cw * 2);
            q_nx.resize_chunks(qw_next, cw);
            b = b * q_nx;
            b.0.chunks(cw)
                .take(k)
                .flat_map(|r| &r[(d - e).min(r.len())..(m - e).min(r.len())])
                .cloned()
                .collect()
        }
        // Kinoshita-Li composition
        // [y^{m-1}] Rev_{m}[g](y)/(1-y f(x)) mod x^k
        pub fn comp_mod_xk(&self, other: &Self, k: usize) -> Self {
            if k == 0 || self.0.is_empty() {
                return Poly::zero();
            }
            let qw = 2;
            let mut p = Poly::new(vec![T::zero(); k]);
            let m = p.0.len().min(k);
            for i in 0..m {
                p.0[i] = self.coeff(m - 1 - i);
            }
            let mut q = Poly::new(vec![T::zero(); k * qw]);
            q.0[0 * qw + 0] = T::one();
            for i in 0..k.min(other.0.len()) {
                q.0[i * qw + 1] = -other.0[i].clone();
            }
            Self::comp_rec(p, q, qw, k, m - 1, m).mod_xk(k)
        }
    }
}
