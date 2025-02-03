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
}

// Example
const P: u64 = 998_244_353;
type ModP = num_mod_static::NaiveModInt<num_mod_static::ByU64<P>>;

pub mod ntt {
    use crate::num_mod_static::{CommRing, PowBy};

    fn bit_reversal_perm<T>(xs: &mut [T]) {
        let n = xs.len();
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;

        for i in 0..n as u32 {
            let rev = i.reverse_bits() >> (u32::BITS - n_log2);
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }
    }

    pub fn radix4<T: CommRing + PowBy<u32>>(proot: T, xs: &mut [T])
    where
        T: Copy,
    {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        bit_reversal_perm(xs);

        let base: Vec<_> = (0..n_log2)
            .scan(proot, |acc, _| {
                let prev = *acc;
                *acc *= *acc;
                Some(prev)
            })
            .collect();

        let mut proot_pow: Vec<T> = vec![T::zero(); n]; // Cache-friendly twiddle factors
        proot_pow[0] = T::one();

        let quartic_root = proot.pow(n as u32 / 4);

        let update_proot_pow = |proot_pow: &mut [T], k: u32| {
            let step = 1 << k;
            let base = base[(n_log2 - k - 1) as usize];
            for i in (0..step).rev() {
                proot_pow[i * 2 + 1] = proot_pow[i] * base;
                proot_pow[i * 2] = proot_pow[i];
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
                    let b0 = *a0;
                    let b1 = *a1;
                    *a0 = b0 + b1;
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

                for ((((a0, a1), a2), a3), &pow1) in
                    t0.into_iter().zip(t1).zip(t2).zip(t3).zip(&proot_pow)
                {
                    let pow2 = pow1 * pow1;
                    let pow1_shift = pow1 * quartic_root;

                    let b0 = *a0;
                    let b1 = *a1 * pow2;
                    let b2 = *a2;
                    let b3 = *a3 * pow2;

                    let c0 = b0 + b1;
                    let c1 = b0 - b1;
                    let c2 = (b2 + b3) * pow1;
                    let c3 = (b2 - b3) * pow1_shift;

                    *a0 = c0 + c2;
                    *a1 = c1 + c3;
                    *a2 = c0 - c2;
                    *a3 = c1 - c3;
                }
            }
            k += 2;
        }
    }
}
