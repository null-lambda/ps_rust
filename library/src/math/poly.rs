pub mod algebra {
    use std::ops::*;
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
    }

    // Non-commutative algebras are not my business (yet)
    // pub trait Ring: SemiRing + Neg<Output = Self> {}

    pub trait CommRing: SemiRing + Neg<Output = Self> {}

    pub trait PowBy<E> {
        fn pow(&self, exp: E) -> Self;
    }

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

    macro_rules! impl_powby {
        ($(($uexp:ty, $iexp:ty),)+) => {
            $(
                impl<R: CommRing> PowBy<$uexp> for R {
                    fn pow(&self, exp: $uexp) -> R {
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

                impl<R: Field> PowBy<$iexp> for R {
                    fn pow(&self, exp: $iexp) -> R {
                        if exp < 0 {
                            self.inv().pow((-exp) as $uexp)
                        } else {
                            self.pow(exp as $uexp)
                        }
                    }
                }
            )+
        };
    }
    impl_powby!(
        (u8, i8),
        (u16, i16),
        (u32, i32),
        (u64, i64),
        (u128, i128),
        (usize, isize),
    );
}

pub mod ntt {
    use super::algebra::*;

    fn bit_reversal_perm<T>(xs: &mut [T]) {
        let n = xs.len();
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        if n == 1 {
            return;
        }

        for i in 0..n as u32 {
            let rev = i.reverse_bits() >> (u32::BITS - n_log2);
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }
    }

    pub fn radix4<T: CommRing + PowBy<u32> + Clone>(proot: T, xs: &mut [T]) {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        bit_reversal_perm(xs);

        let base: Vec<_> = (0..n_log2)
            .scan(proot.clone(), |acc, _| {
                let prev = acc.clone();
                *acc *= acc.clone();
                Some(prev)
            })
            .collect();

        let mut proot_pow: Vec<T> = vec![T::zero(); n];
        proot_pow[0] = T::one();

        let quartic_root = proot.pow(n as u32 / 4);

        let update_proot_pow = |proot_pow: &mut [T], k: u32| {
            let step = 1 << k;
            let base = base[(n_log2 - k - 1) as usize].clone();
            for i in (0..step).rev() {
                proot_pow[i * 2 + 1] = proot_pow[i].clone() * base.clone();
                proot_pow[i * 2] = proot_pow[i].clone();
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
                    let b0 = a0.clone();
                    let b1 = a1.clone();
                    *a0 = b0.clone() + b1.clone();
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

                for ((((a0, a1), a2), a3), pow1) in
                    t0.into_iter().zip(t1).zip(t2).zip(t3).zip(&proot_pow)
                {
                    let pow2 = pow1.clone() * pow1;
                    let pow1_shift = pow1.clone() * &quartic_root;

                    let b0 = a0.clone();
                    let b1 = a1.clone() * &pow2;
                    let b2 = a2.clone();
                    let b3 = a3.clone() * &pow2;

                    let c0 = b0.clone() + &b1;
                    let c1 = b0.clone() - &b1;
                    let c2 = (b2.clone() + &b3) * pow1;
                    let c3 = (b2 - b3) * pow1_shift;

                    *a0 = c0.clone() + &c2;
                    *a1 = c1.clone() + &c3;
                    *a2 = c0.clone() - &c2;
                    *a3 = c1.clone() - &c3;
                }
            }
            k += 2;
        }
    }
}

pub mod poly {
    use super::algebra::*;
    use super::ntt;
    use std::ops::*;

    // pub trait NTTSpec: Field + From<u32> {
    pub trait NTTSpec: Field + From<u32> + std::fmt::Debug {
        // TODO
        // Primitive nth root of unity
        fn nth_proot(n: u32) -> Self;
    }

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    // pub struct Poly<T>(Box<[T]>);
    pub struct Poly<T>(pub Vec<T>);

    impl<T: SemiRing + PartialEq> Poly<T> {
        pub fn new(coeffs: Vec<T>) -> Self {
            Self(coeffs)
        }

        pub fn lagrange(ps: impl IntoIterator<Item = (T, T)>) -> Self {
            todo!()
        }

        pub fn pop_zeros(&mut self) {
            while self.0.last().filter(|&c| c == &T::zero()).is_some() {
                self.0.pop();
            }
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn degree(&mut self) -> usize {
            self.pop_zeros();
            self.0.len() - 1
        }

        pub fn leading_coeff(&self) -> T {
            self.0.last().cloned().unwrap_or(T::zero())
        }

        pub fn eval(&self, x: T) -> T {
            let mut res = T::zero();
            for c in self.0.iter().rev() {
                res *= x.clone();
                res += c.clone();
            }
            res
        }

        pub fn reverse(&mut self) {
            self.0.reverse()
        }

        pub fn mod_xk(&mut self, exp: usize) {
            if self.degree() >= exp {
                self.0.truncate(exp);
            }
        }

        pub fn mul_xk(&mut self, exp: usize) {
            self.0 = ((0..exp).map(|_| T::zero()))
                .chain(std::mem::take(&mut self.0))
                .collect();
        }
    }

    impl<T: NTTSpec + PartialEq> Poly<T> {
        pub fn inv_mod_xk(&mut self, exp: usize) -> Self {
            assert!(self.0[0] != T::zero());
            let mut res = Poly::from(self.0[0].inv());
            let mut k = 1;
            let two = Poly::from(T::from(2));
            while k < exp {
                let mut r = self.clone();
                r.mod_xk(k << 1);
                let p = two.clone() - &(res.clone() * r);
                res *= p;
                res.mod_xk(k << 1);

                k <<= 1;
            }

            res
        }
    }

    impl<T: SemiRing> From<T> for Poly<T> {
        fn from(c: T) -> Self {
            Self(vec![c])
        }
    }

    impl<'a, T: SemiRing + PartialEq> MulAssign<&'a T> for Poly<T> {
        fn mul_assign(&mut self, rhs: &T) {
            self.0.iter_mut().for_each(|c| c.mul_assign(rhs.clone()));
        }
    }

    impl<T: SemiRing + PartialEq> MulAssign<T> for Poly<T> {
        fn mul_assign(&mut self, rhs: T) {
            self.mul_assign(&rhs);
        }
    }

    impl<'a, T: SemiRing + PartialEq> AddAssign<&'a Self> for Poly<T> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0.resize_with(self.len().max(rhs.len()), T::zero);
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.add_assign(b));
        }
    }

    impl<'a, T: SemiRing + PartialEq> Add<&'a Self> for Poly<T> {
        type Output = Self;
        fn add(mut self, rhs: &Self) -> Self {
            self += rhs;
            self
        }
    }

    impl<'a, T: SemiRing + PartialEq> SubAssign<&'a Self> for Poly<T> {
        fn sub_assign(&mut self, rhs: &Self) {
            self.0.resize_with(self.len().max(rhs.len()), T::zero);
            self.0
                .iter_mut()
                .zip(&rhs.0)
                .for_each(|(a, b)| a.sub_assign(b));
        }
    }

    impl<'a, T: SemiRing + PartialEq> Sub<&'a Self> for Poly<T> {
        type Output = Self;
        fn sub(mut self, rhs: &Self) -> Self {
            self -= rhs;
            self
        }
    }

    impl<T: NTTSpec + PartialEq> MulAssign<Self> for Poly<T> {
        fn mul_assign(&mut self, mut rhs: Self) {
            self.pop_zeros();
            rhs.pop_zeros();
            if self.len() == 0 || rhs.len() == 0 {
                self.0.clear();
                return;
            }

            let n = self.len() + rhs.len() - 1;
            let n_padded = n.next_power_of_two();

            self.0.resize(n_padded, T::zero());
            rhs.0.resize(n_padded, T::zero());

            let proot = T::nth_proot(n_padded as u32);
            ntt::radix4(proot.clone(), &mut self.0);
            ntt::radix4(proot.clone(), &mut rhs.0);
            for (a, b) in self.0.iter_mut().zip(&rhs.0) {
                a.mul_assign(b);
            }
            ntt::radix4(proot.inv(), &mut self.0);
            let n_inv = T::from(n_padded as u32).inv();
            self.0.iter_mut().for_each(|c| c.mul_assign(&n_inv));

            self.0.truncate(n);
        }
    }

    impl<T: NTTSpec + PartialEq> Mul<Self> for Poly<T> {
        type Output = Self;
        fn mul(mut self, rhs: Self) -> Self {
            self *= rhs;
            self
        }
    }

    impl<T: NTTSpec + PartialEq> DivAssign<Self> for Poly<T> {
        fn div_assign(&mut self, mut rhs: Self) {
            assert!(rhs.degree() != 0);
            self.pop_zeros();
            rhs.pop_zeros();
            if self.degree() < rhs.degree() {
                self.0.clear();
                return;
            }

            let n = self.degree();
            let m = rhs.degree();
            let l = n - m + 1;
            // print!("self {:?}, rhs {:?} -> ", self, rhs);
            self.reverse();
            self.mod_xk(l);
            rhs.reverse();
            rhs.mod_xk(l);
            // print!("self {:?}, rhs {:?} -> ", self, rhs);
            *self *= rhs.inv_mod_xk(l);
            self.0.resize(l, T::zero());
            self.reverse();
            // println!("{:?}", self);
        }
    }

    impl<'a, T: NTTSpec + PartialEq> RemAssign<Self> for Poly<T> {
        fn rem_assign(&mut self, rhs: Self) {
            let mut q = self.clone();
            q /= rhs.clone();
            q *= rhs;
            *self -= &q;
        }
    }

    impl<'a, T: CommRing> Neg for &Poly<T> {
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
}

pub mod num_mod {
    use super::algebra::*;
    use std::ops::*;

    pub trait Unsigned:
        Copy
        + Default
        + SemiRing
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

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
    }

    pub trait ByPrime: ModSpec {}

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct ModInt<M: ModSpec>(M::U);

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
    impl_modspec!(
        ByU32 u32, ByU32Prime u32,
        ByU64 u64, ByU64Prime u64,
        ByU128 u128, ByU128Prime u128
    );

    macro_rules! impl_by_prime {
        ($($t:ident $u:ty),+) => {
            $(
                impl<const MOD: $u> ByPrime for $t<MOD> {}
            )+
        };
    }
    impl_by_prime!(ByU32Prime u32, ByU64Prime u64, ByU128Prime u128);

    impl<'a, M: ModSpec> AddAssign<&'a Self> for ModInt<M> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<'a, M: ModSpec> SubAssign<&'a Self> for ModInt<M> {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<'a, M: ModSpec> MulAssign<&'a Self> for ModInt<M> {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 *= rhs.0;
            self.0 %= M::MODULUS;
        }
    }

    macro_rules! forward_ref_binop {
        ($($OpAssign:ident $op_assign:ident),+) => {
            $(
                impl<M: ModSpec> $OpAssign for ModInt<M> {
                    fn $op_assign(&mut self, rhs: Self) {
                        self.$op_assign(&rhs);
                    }
                }
            )+
        };
    }
    forward_ref_binop!(AddAssign add_assign, MulAssign mul_assign, SubAssign sub_assign);

    macro_rules! impl_op_by_op_assign {
        ($($Op:ident $op:ident $op_assign:ident),+) => {
            $(
                impl<'a, M: ModSpec> $Op<&'a Self> for ModInt<M> {
                    type Output = Self;
                    fn $op(mut self, rhs: &Self) -> Self {
                        self.$op_assign(rhs);
                        self
                    }
                }

                impl< M: ModSpec> $Op for ModInt<M> {
                    type Output = ModInt<M>;
                    fn $op(self, rhs: Self) -> Self::Output {
                        self.clone().$op(&rhs)
                    }
                }
            )+
        };
    }
    impl_op_by_op_assign!(Add add add_assign, Mul mul mul_assign, Sub sub sub_assign);

    impl<'a, M: ModSpec> Neg for &'a ModInt<M> {
        type Output = ModInt<M>;
        fn neg(self) -> ModInt<M> {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            ModInt(res)
        }
    }

    impl<M: ModSpec> Neg for ModInt<M> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl<M: ModSpec> Default for ModInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> SemiRing for ModInt<M> {
        fn one() -> Self {
            Self(1.into())
        }
    }
    impl<M: ModSpec> CommRing for ModInt<M> {}

    impl<'a, M: ByPrime> DivAssign<&'a Self> for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    impl<M: ByPrime> DivAssign for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn div_assign(&mut self, rhs: Self) {
            self.div_assign(&rhs);
        }
    }

    impl<'a, M: ByPrime> Div<&'a Self> for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        type Output = Self;
        fn div(mut self, rhs: &Self) -> Self {
            self.div_assign(rhs);
            self
        }
    }

    impl<M: ByPrime> Div for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            self / &rhs
        }
    }

    impl<M: ByPrime> Field for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
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
        (
            $( $($lower:ident)* < $target:ident < $($upper:ident)* ),* $(,)?
        ) => {
            $(
                $(
                    impl CmpUType<$lower> for $target {
                        type MaxT = $target;
                        fn upcast(lhs: Self) -> Self::MaxT {
                            lhs as Self::MaxT
                        }
                        fn upcast_rhs(rhs: $lower) -> Self::MaxT {
                            rhs as Self::MaxT
                        }
                        fn downcast(max: Self::MaxT) -> Self {
                            max as Self
                        }
                    }
                )*
                impl CmpUType<$target> for $target {
                    type MaxT = $target;
                    fn upcast(lhs: Self) -> Self::MaxT {
                        lhs as Self::MaxT
                    }
                    fn upcast_rhs(rhs: $target) -> Self::MaxT {
                        rhs as Self::MaxT
                    }
                    fn downcast(max: Self::MaxT) -> Self {
                        max as Self
                    }
                }
                $(
                    impl CmpUType<$upper> for $target {
                        type MaxT = $upper;
                        fn upcast(lhs: Self) -> Self::MaxT {
                            lhs as Self::MaxT
                        }
                        fn upcast_rhs(rhs: $upper) -> Self::MaxT {
                            rhs as Self::MaxT
                        }
                        fn downcast(max: Self::MaxT) -> Self {
                            max as Self
                        }
                    }
                )*
            )*
        };
    }
    impl_cmp_utype!(
        < u8 < u16 u32 u64 u128,
        u8 < u16 < u32 u64 u128,
        u8 u16 < u32 < u64 u128,
        u8 u16 u32 < u64 < u128,
        u8 u16 u32 u64 < u128 <,
    );

    impl<U, S, M> From<S> for ModInt<M>
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
                impl<M: ModSpec<U = $u>> From<ModInt<M>> for $u {
                    fn from(n: ModInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl<U: std::fmt::Debug, M: ModSpec<U = U>> std::fmt::Debug for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }
}

pub mod linear_recurrence {
    use super::algebra::CommRing;
    use super::poly::{NTTSpec, Poly};

    pub fn berlekamp_massey<T: CommRing>(seq: &[T]) -> Vec<T> {
        unimplemented!()
    }

    pub fn next<T: CommRing + Copy>(recurrence: &[T], init: &[T]) -> T {
        let l = recurrence.len();
        let n = init.len();
        assert!(n >= l);
        let mut value = recurrence[0] * init[n - 1];
        for i in 1..l {
            value += recurrence[i] * init[n - 1 - i];
        }
        value
    }

    // Kitamasa method, O(L log L log N)
    pub fn nth_by_ntt<T: NTTSpec + PartialEq + Copy>(
        recurrence: &[T],
        init: &[T],
        mut n: u64,
    ) -> T {
        let l = recurrence.len();
        assert!(1 <= l && l == init.len());

        let modulus = Poly::new(
            recurrence
                .iter()
                .rev()
                .map(|&c| -c)
                .chain(Some(T::one()))
                .collect(),
        );
        let mut pow = Poly::new(vec![T::one()]);
        let mut base = Poly::new(vec![T::zero(), T::one()]);

        while n > 0 {
            if n & 1 == 1 {
                {
                    // print!("{:?} ", pow);
                    // print!("/ {:?} ", modulus);
                    // let mut temp = pow.clone();
                    // temp /= modulus.clone();
                    // println!(" = {:?}", temp);
                }
                pow *= base.clone();
                pow %= modulus.clone();
            }
            base *= base.clone();
            base %= modulus.clone();
            n >>= 1;
        }

        pow.0.resize(l, T::zero());
        (0..l)
            .map(|i| init[i] * pow.0[i])
            .fold(T::zero(), |acc, x| acc + x)
    }
}

// Check:
// https://oeis.org/A039687
// https://oeis.org/A050526
// https://oeis.org/A300407

pub mod p13631489 {
    use super::algebra::PowBy;
    use super::num_mod::*;
    use super::poly::NTTSpec;
    pub const P: u64 = 13631489;
    pub const GEN: u64 = 15;
    pub type ModP = ModInt<ByU64Prime<P>>;
    impl NTTSpec for ModP {
        fn nth_proot(n: u32) -> Self {
            assert!((P - 1) % n as u64 == 0);
            ModP::from(GEN).pow((P - 1) / n as u64).into()
        }
    }
}

pub mod p104857601 {
    use super::algebra::PowBy;
    use super::num_mod::*;
    use super::poly::NTTSpec;
    pub const P: u64 = 104857601;
    pub const GEN: u64 = 3;
    pub type ModP = ModInt<ByU64Prime<P>>;
    impl NTTSpec for ModP {
        fn nth_proot(n: u32) -> Self {
            assert!((P - 1) % n as u64 == 0);
            ModP::from(GEN).pow((P - 1) / n as u64).into()
        }
    }
}

pub mod p167772161 {
    use super::algebra::PowBy;
    use super::num_mod::*;
    use super::poly::NTTSpec;
    pub const P: u64 = 167772161;
    pub const GEN: u64 = 3;
    pub type ModP = ModInt<ByU64Prime<P>>;
    impl NTTSpec for ModP {
        fn nth_proot(n: u32) -> Self {
            assert!((P - 1) % n as u64 == 0);
            ModP::from(GEN).pow((P - 1) / n as u64).into()
        }
    }
}

pub mod p998244353 {
    use super::algebra::PowBy;
    use super::num_mod::*;
    use super::poly::NTTSpec;
    pub const P: u64 = 998244353;
    pub const GEN: u64 = 3;
    pub type ModP = ModInt<ByU64Prime<P>>;
    impl NTTSpec for ModP {
        fn nth_proot(n: u32) -> Self {
            assert!((P - 1) % n as u64 == 0);
            ModP::from(GEN).pow((P - 1) / n as u64).into()
        }
    }
}

pub mod p9223372036737335297 {
    use super::algebra::PowBy;
    use super::num_mod::*;
    use super::poly::NTTSpec;

    pub const P: u128 = 9223372036737335297;
    pub const GEN: u128 = 3;
    pub type ModP = ModInt<ByU128Prime<P>>;
    impl NTTSpec for ModP {
        fn nth_proot(n: u32) -> Self {
            assert!((P - 1) % n as u128 == 0);
            ModP::from(GEN).pow((P - 1) / n as u128).into()
        }
    }
}

pub mod p18446744069414584321 {
    use super::algebra::PowBy;
    use super::num_mod::*;
    use super::poly::NTTSpec;
    pub const P: u128 = 18446744069414584321;
    pub const GEN: u128 = 7;
    pub type ModP = ModInt<ByU128Prime<P>>;
    impl NTTSpec for ModP {
        fn nth_proot(n: u32) -> Self {
            assert!((P - 1) % n as u128 == 0);
            ModP::from(GEN).pow((P - 1) / n as u128).into()
        }
    }
}
