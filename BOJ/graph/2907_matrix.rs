use std::io::Write;

use num_mod::ModOp;
use std::{collections::HashMap, hash::Hash};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn parse_mask(s: &[u8]) -> usize {
    let b0 = s.contains(&b'B') as usize;
    let b1 = s.contains(&b'J') as usize;
    let b2 = s.contains(&b'M') as usize;
    let b3 = s.contains(&b'P') as usize;
    b0 | b1 << 1 | b2 << 2 | b3 << 3
}

pub mod num_mod {
    use std::ops::*;
    pub trait ModOp<T> {
        fn zero(&self) -> T;
        fn one(&self) -> T;
        fn modulus(&self) -> T;
        fn add(&self, lhs: T, rhs: T) -> T;
        fn sub(&self, lhs: T, rhs: T) -> T;
        fn mul(&self, lhs: T, rhs: T) -> T;
        fn transform(&self, n: T) -> T;
        fn reduce(&self, n: T) -> T;
    }

    pub trait PowBy<T, E> {
        fn pow(&self, base: T, exp: E) -> T;
    }

    pub trait InvOp<T> {
        fn inv(&self, n: T) -> T;
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u32> for M {
        fn pow(&self, mut base: T, mut exp: u32) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u64> for M {
        fn pow(&self, mut base: T, mut exp: u64) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u128> for M {
        fn pow(&self, mut base: T, mut exp: u128) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<M: ModOp<u32>> InvOp<u32> for M {
        fn inv(&self, n: u32) -> u32 {
            self.pow(n, self.modulus() - 2)
        }
    }

    impl<M: ModOp<u64>> InvOp<u64> for M {
        fn inv(&self, n: u64) -> u64 {
            self.pow(n, self.modulus() - 2)
        }
    }

    impl<M: ModOp<u128>> InvOp<u128> for M {
        fn inv(&self, n: u128) -> u128 {
            self.pow(n, self.modulus() - 2)
        }
    }

    pub struct NaiveModOp<T> {
        m: T,
    }

    impl<T> NaiveModOp<T> {
        pub fn new(m: T) -> Self {
            Self { m }
        }
    }

    pub trait One {
        fn one() -> Self;
    }

    impl One for u32 {
        fn one() -> Self {
            1
        }
    }

    impl One for u64 {
        fn one() -> Self {
            1
        }
    }

    impl One for u128 {
        fn one() -> Self {
            1
        }
    }

    impl<T> ModOp<T> for NaiveModOp<T>
    where
        T: Copy
            + Default
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Rem<Output = T>
            + PartialOrd,
    {
        fn zero(&self) -> T {
            T::default()
        }
        fn one(&self) -> T {
            T::one()
        }
        fn modulus(&self) -> T {
            self.m
        }
        fn add(&self, lhs: T, rhs: T) -> T {
            if lhs + rhs >= self.m {
                lhs + rhs - self.m
            } else {
                lhs + rhs
            }
        }
        fn sub(&self, lhs: T, rhs: T) -> T {
            if lhs >= rhs {
                lhs - rhs
            } else {
                lhs + self.m - rhs
            }
        }
        fn mul(&self, lhs: T, rhs: T) -> T {
            (lhs * rhs) % self.m
        }
        fn transform(&self, n: T) -> T {
            n % self.m
        }
        fn reduce(&self, n: T) -> T {
            n % self.m
        }
    }

    macro_rules! impl_montgomery {
        ($type:ident, S = $single:ty, D = $double:ty, EXP = $exp:expr, LOG2_EXP = $log2_exp:expr, ZERO = $zero:expr, ONE = $one:expr) => {
            #[derive(Debug, Clone)]
            pub struct $type {
                m: $single,
                m_inv: $single,
                r2: $single,
            }

            impl $type {
                pub fn new(m: $single) -> Self {
                    debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
                    let mut m_inv = $one;
                    let two = $one + $one;
                    for _ in 0..$log2_exp {
                        m_inv = m_inv.wrapping_mul(two.wrapping_sub(m.wrapping_mul(m_inv)));
                    }
                    let r = m.wrapping_neg() % m;
                    let r2 = ((r as $double * r as $double % m as $double) as $single);

                    Self { m, m_inv, r2 }
                }

                fn reduce_double(&self, x: $double) -> $single {
                    debug_assert!((x as $double) < (self.m as $double) * (self.m as $double));
                    let q = (x as $single).wrapping_mul(self.m_inv);
                    let a = ((q as $double * self.m as $double) >> $exp) as $single;
                    let mut res = (x >> $exp) as $single + self.m - a;
                    if res >= self.m {
                        res -= self.m;
                    }
                    res
                }
            }

            impl ModOp<$single> for $type {
                fn zero(&self) -> $single {
                    $zero
                }

                fn one(&self) -> $single {
                    self.transform($one)
                }

                fn modulus(&self) -> $single {
                    self.m
                }

                fn mul(&self, x: $single, y: $single) -> $single {
                    debug_assert!(x < self.m);
                    debug_assert!(y < self.m);
                    self.reduce_double(x as $double * y as $double)
                }

                fn add(&self, x: $single, y: $single) -> $single {
                    debug_assert!(x < self.m);
                    debug_assert!(y < self.m);
                    let sum = x + y;
                    if sum >= self.m {
                        sum - self.m
                    } else {
                        sum
                    }
                }

                fn sub(&self, x: $single, y: $single) -> $single {
                    debug_assert!(x < self.m);
                    debug_assert!(y < self.m);
                    if x >= y {
                        x - y
                    } else {
                        x + self.m - y
                    }
                }

                fn reduce(&self, x: $single) -> $single {
                    self.reduce_double(x as $double)
                }

                fn transform(&self, x: $single) -> $single {
                    debug_assert!(x < self.m);
                    self.mul(x, self.r2)
                }
            }
        };
    }

    impl_montgomery! {
        MontgomeryU32,
        S = u32,
        D = u64,
        EXP = 32,
        LOG2_EXP = 5,
        ZERO = 0u32,
        ONE = 1u32
    }

    impl_montgomery! {
        MontgomeryU64,
        S = u64,
        D = u128,
        EXP = 64,
        LOG2_EXP = 6,
        ZERO = 0u64,
        ONE = 1u64
    }
}

fn mul_mat(mod_op: &impl ModOp<u32>, x: &[Vec<u32>], y: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = x.len();
    let mut res = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                res[i][j] = mod_op.add(res[i][j], mod_op.mul(x[i][k], y[k][j]));
            }
        }
    }
    res
}

fn apply(mod_op: &impl ModOp<u32>, init: &[u32], mat: &[Vec<u32>]) -> Vec<u32> {
    let n = init.len();
    let mut res = vec![0; n];
    for i in 0..n {
        for j in 0..n {
            res[i] = mod_op.add(res[i], mod_op.mul(init[j], mat[j][i]));
        }
    }
    res
}

fn apply_pow(mod_op: &impl ModOp<u32>, init: &[u32], base: &[Vec<u32>], mut exp: u32) -> Vec<u32> {
    let mut res = init.to_vec();
    let mut base = base.to_vec();
    while exp > 0 {
        if exp % 2 == 1 {
            res = apply(mod_op, &res, &base);
        }
        base = mul_mat(mod_op, &base, &base);
        exp >>= 1;
    }
    res
}

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mod_op = num_mod::MontgomeryU32::new(5557);

    let n: usize = input.value();
    let r: usize = input.value();

    let n_states = 2 * n + 1;
    let state = |vert: usize, delay: usize| {
        debug_assert!(vert < n);
        debug_assert!(delay < 2);
        vert * 2 + delay
    };
    let acc_state = n_states - 1;
    let src = state(0, 0);
    let dest = state(0, 0);

    let mut mats = vec![vec![vec![mod_op.zero(); n_states]; n_states]; 16];
    for _ in 0..r {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let w = parse_mask(input.token().as_bytes());
        for banned in 0..16 {
            mats[banned][state(u, 0)][state(v, 0)] = mod_op.one();
            if w & banned == 0 {
                mats[banned][state(u, 0)][state(v, 1)] = mod_op.one();
                mats[banned][state(v, 1)][state(v, 0)] = mod_op.one();
            }
        }
    }
    for banned in 0..16 {
        for s in 0..n_states {
            mats[banned][s][acc_state] = mats[banned][s][dest];
        }
        mats[banned][acc_state][acc_state] = mod_op.one();
    }

    let k: u32 = input.value();

    let mut acc = 0;
    for banned in 0..16 {
        let mut init = vec![0; n_states];
        init[src] = mod_op.one();

        let fin = apply_pow(&mod_op, &init, &mats[banned], k);
        let res = mod_op.reduce(fin[acc_state]);
        if banned.count_ones() % 2 == 0 {
            acc = mod_op.add(acc, res);
        } else {
            acc = mod_op.sub(acc, res);
        }
    }

    writeln!(output, "{}", acc).unwrap();
}
