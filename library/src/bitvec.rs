pub mod bitset {
    // TODO: forward
    // TODO: empiricial test

    use std::ops::*;

    pub type B = u64;
    pub const BW: usize = 64;

    #[derive(Clone)]
    pub struct BitVec(pub Vec<B>);

    impl BitVec {
        pub fn zero_bits(n: usize) -> Self {
            Self(vec![0; n.div_ceil(BW)])
        }
        pub fn one_bits(n: usize) -> Self {
            let mut res = Self(vec![!0; n.div_ceil(BW)]);
            if n % BW != 0 {
                res.0[n / BW] = (1 << n % BW) - 1;
            }
            res
        }

        pub fn bitlen(&self) -> usize {
            self.0.len() * BW
        }

        pub fn bit_trunc(&mut self, n: usize) {
            let q = n.div_ceil(BW);
            if q > self.0.len() {
                return;
            }
            self.0.truncate(q);
            if n % BW != 0 {
                self.0[q - 1] &= (1 << n % BW) - 1;
            }
        }

        pub fn get(&self, i: usize) -> bool {
            let (b, s) = (i / BW, i % BW);
            (self.0[b] >> s) & 1 != 0
        }
        #[inline]
        pub fn set(&mut self, i: usize, value: bool) {
            if !value {
                self.0[i / BW] &= !(1 << i % BW);
            } else {
                self.0[i / BW] |= 1 << i % BW;
            }
        }
        #[inline]
        pub fn toggle(&mut self, i: usize) {
            self.0[i / BW] ^= 1 << i % BW;
        }

        pub fn count_ones(&self) -> u32 {
            self.0.iter().map(|&m| m.count_ones()).sum()
        }
    }

    impl Neg for BitVec {
        type Output = Self;
        fn neg(mut self) -> Self::Output {
            for x in &mut self.0 {
                *x = !*x;
            }
            self
        }
    }
    impl BitAndAssign<&'_ Self> for BitVec {
        fn bitand_assign(&mut self, rhs: &'_ Self) {
            assert_eq!(self.0.len(), rhs.0.len());
            for (x, y) in self.0.iter_mut().zip(&rhs.0) {
                x.bitand_assign(y);
            }
        }
    }
    impl BitOrAssign<&'_ Self> for BitVec {
        fn bitor_assign(&mut self, rhs: &'_ Self) {
            assert_eq!(self.0.len(), rhs.0.len());
            for (x, y) in self.0.iter_mut().zip(&rhs.0) {
                x.bitor_assign(y);
            }
        }
    }
    impl BitXorAssign<&'_ Self> for BitVec {
        fn bitxor_assign(&mut self, rhs: &'_ Self) {
            assert_eq!(self.0.len(), rhs.0.len());
            for (x, y) in self.0.iter_mut().zip(&rhs.0) {
                x.bitxor_assign(y);
            }
        }
    }
    impl ShlAssign<usize> for BitVec {
        fn shl_assign(&mut self, shift: usize) {
            if shift == 0 {
                return;
            }

            let n = self.bitlen();
            if shift >= n {
                self.0.fill(0);
                return;
            }

            let q = self.0.len();
            let q_shift = shift / BW;
            let r_shift = shift % BW;

            if r_shift == 0 {
                for n in (q_shift..q).rev() {
                    self.0[n] = self.0[n - q_shift];
                }
            } else {
                let sub_shift = (BW - r_shift) as u32;
                for n in ((q_shift + 1)..q).rev() {
                    self.0[n] =
                        (self.0[n - q_shift] << r_shift) | (self.0[n - q_shift - 1] >> sub_shift);
                }
                self.0[q_shift] = self.0[0] << r_shift;
            }
            self.0[..q_shift].fill(0);
        }
    }
    impl ShrAssign<usize> for BitVec {
        fn shr_assign(&mut self, shift: usize) {
            if shift == 0 {
                return;
            }

            let nbits = self.bitlen();
            if shift >= nbits {
                self.0.fill(0);
                return;
            }

            let q = self.0.len();
            let q_shift = shift / BW;
            let r_shift = shift % BW;

            if r_shift == 0 {
                for n in 0..q - q_shift {
                    self.0[n] = self.0[n + q_shift];
                }
            } else {
                let sub_shift = (BW - r_shift) as u32;
                for n in 0..q - q_shift - 1 {
                    self.0[n] =
                        (self.0[n + q_shift] >> r_shift) | (self.0[n + q_shift + 1] << sub_shift);
                }
                self.0[q - q_shift - 1] = self.0[q - 1] >> r_shift;
            }
            self.0[q.saturating_sub(q_shift)..q].fill(0);
        }
    }

    impl std::fmt::Debug for BitVec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BitVec(")?;
            for i in 0..self.0.len() {
                for j in 0..BW {
                    write!(f, "{}", ((self.0[i] >> j) & 1) as u8)?;
                }
            }
            write!(f, ")")?;
            Ok(())
        }
    }
}
