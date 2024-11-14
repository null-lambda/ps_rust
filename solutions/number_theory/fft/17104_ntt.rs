use std::io::Write;

use num::InvOp;
use num::ModOp;
use num::Montgometry;
use num::PowBy;

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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod num {
    pub trait ModOp<T> {
        fn zero(&self) -> T;
        fn one(&self) -> T;
        fn modulus(&self) -> T;
        fn add(&self, lhs: T, rhs: T) -> T;
        fn sub(&self, lhs: T, rhs: T) -> T;
        fn mul(&self, lhs: T, rhs: T) -> T;
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

    // Montgomery reduction
    #[derive(Debug, Clone)]
    pub struct Montgometry<T> {
        m: T,
        m_inv: T,
        r2: T,
    }

    impl Montgometry<u32> {
        pub fn new(m: u32) -> Self {
            debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
            let mut m_inv = 1u32;
            for _ in 0..5 {
                m_inv = m_inv.wrapping_mul(2u32.wrapping_sub(m.wrapping_mul(m_inv)));
            }
            let r = m.wrapping_neg() % m;
            let r2 = (r as u64 * r as u64 % m as u64) as u32;

            Self { m, m_inv, r2 }
        }

        fn reduce_double(&self, x: u64) -> u32 {
            debug_assert!((x as u64) < (self.m as u64) * (self.m as u64));
            let q = (x as u32).wrapping_mul(self.m_inv);
            let a = ((q as u64 * self.m as u64) >> 32) as u32;
            let mut res = (x >> 32) as u32 + self.m - a;
            if res >= self.m {
                res -= self.m;
            }
            res as u32
        }

        pub fn reduce(&self, x: u32) -> u32 {
            self.reduce_double(x as u64)
        }

        pub fn transform(&self, x: u32) -> u32 {
            debug_assert!(x < self.m);
            self.mul(x, self.r2)
        }
    }

    impl ModOp<u32> for Montgometry<u32> {
        fn zero(&self) -> u32 {
            0
        }
        fn one(&self) -> u32 {
            self.transform(1)
        }
        fn modulus(&self) -> u32 {
            self.m
        }
        fn mul(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            self.reduce_double(x as u64 * y as u64)
        }

        fn add(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            let sum = x + y;
            if sum >= self.m {
                sum - self.m
            } else {
                sum
            }
        }

        fn sub(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            if x >= y {
                x - y
            } else {
                x + self.m - y
            }
        }
    }

    impl Montgometry<u64> {
        pub fn new(m: u64) -> Self {
            debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
            let mut m_inv = 1u64;
            for _ in 0..6 {
                // More iterations may be needed for u64 precision
                m_inv = m_inv.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(m_inv)));
            }
            let r = m.wrapping_neg() % m;
            let r2 = (r as u128 * r as u128 % m as u128) as u64;

            Self { m, m_inv, r2 }
        }

        pub fn reduce_double(&self, x: u128) -> u64 {
            debug_assert!((x as u128) < (self.m as u128) * (self.m as u128));
            let q = (x as u64).wrapping_mul(self.m_inv);
            let a = ((q as u128 * self.m as u128) >> 64) as u64;
            let mut res = (x >> 64) as u64 + self.m - a;
            if res >= self.m {
                res -= self.m;
            }
            res as u64
        }
        pub fn reduce(&self, x: u64) -> u64 {
            self.reduce_double(x as u128)
        }

        pub fn transform(&self, x: u64) -> u64 {
            debug_assert!(x < self.m);
            self.mul(x, self.r2)
        }
    }

    impl ModOp<u64> for Montgometry<u64> {
        fn zero(&self) -> u64 {
            0
        }

        fn one(&self) -> u64 {
            self.transform(1)
        }

        fn modulus(&self) -> u64 {
            self.m
        }

        fn mul(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            self.reduce_double(x as u128 * y as u128)
        }

        fn add(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            let sum = x + y;
            if sum >= self.m {
                sum - self.m
            } else {
                sum
            }
        }

        fn sub(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            if x >= y {
                x - y
            } else {
                x + self.m - y
            }
        }
    }
}

pub mod ntt {
    use std::iter;

    use crate::num::{ModOp, PowBy};

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

    pub fn radix4<T, M>(op: &M, proot: T, xs: &mut [T])
    where
        T: Copy,
        M: ModOp<T> + PowBy<T, u32>,
    {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        bit_reversal_perm(xs);

        let base: Vec<_> = (0..n_log2)
            .scan(proot, |acc, _| {
                let prev = *acc;
                *acc = op.mul(*acc, *acc);
                Some(prev)
            })
            .collect();

        let mut proot_pow = vec![op.zero(); n]; // Cache-friendly twiddle factors
        proot_pow[0] = op.one();

        let quartic_root = op.pow(proot, n as u32 / 4);

        let update_proot_pow = |proot_pow: &mut [T], k: u32| {
            let step = 1 << k;
            let base = base[(n_log2 - k - 1) as usize];
            for i in (0..step).rev() {
                proot_pow[i * 2 + 1] = op.mul(proot_pow[i], base);
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
                    *a0 = op.add(b0, b1);
                    *a1 = op.sub(b0, b1);
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
                    let pow2 = op.mul(pow1, pow1);
                    let pow1_shift = op.mul(pow1, quartic_root);

                    let b0 = *a0;
                    let b1 = op.mul(*a1, pow2);
                    let b2 = *a2;
                    let b3 = op.mul(*a3, pow2);

                    let c0 = op.add(b0, b1);
                    let c1 = op.sub(b0, b1);
                    let c2 = op.mul(op.add(b2, b3), pow1);
                    let c3 = op.mul(op.sub(b2, b3), pow1_shift);

                    *a0 = op.add(c0, c2);
                    *a1 = op.add(c1, c3);
                    *a2 = op.sub(c0, c2);
                    *a3 = op.sub(c1, c3);
                }
            }
            k += 2;
        }
    }

    // naive O(n^2)
    pub fn naive<T, M>(op: &M, proot: T, xs: &mut [T])
    where
        T: Copy,
        M: ModOp<T> + PowBy<T, u32>,
    {
        let n = xs.len().next_power_of_two();
        let proot_pow: Vec<T> = iter::successors(Some(op.one()), |&acc| Some(op.mul(acc, proot)))
            .take(n)
            .collect();
        let res: Vec<_> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| op.mul(xs[j], proot_pow[(i * j) % n]))
                    .fold(op.zero(), |acc, x| op.add(acc, x))
            })
            .collect();
        xs.copy_from_slice(&res);
    }
}

fn linear_sieve(n_max: u32) -> (Vec<u32>, Vec<u32>) {
    let mut min_prime_factor = vec![0; n_max as usize + 1];
    let mut primes = Vec::new();

    for i in 2..=n_max {
        if min_prime_factor[i as usize] == 0 {
            primes.push(i);
        }
        for &p in primes.iter() {
            if i * p > n_max {
                break;
            }
            min_prime_factor[(i * p) as usize] = p;
            if i % p == 0 {
                break;
            }
        }
    }

    (min_prime_factor, primes)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n_max = 1_000_000;
    let (mpf, _) = linear_sieve(n_max);
    let mut x: Vec<u32> = mpf
        .iter()
        .enumerate()
        .map(|(x, &p)| (x >= 2 && p == 0) as u32)
        .collect();

    // Polynomial multiplication with NTT
    let nx = x.len();
    let n = nx.next_power_of_two() * 2;
    x.resize(n, 0);

    let p = 998_244_353;
    // let p = 9223372036737335297;
    let mont = Montgometry::<u32>::new(p);
    let gen = 3;
    let proot = mont.pow(mont.transform(gen), (p - 1) / n as u32);
    let proot_inv = mont.inv(proot);
    let n_inv = mont.inv(mont.transform(n as u32));

    for a in &mut x {
        *a = mont.transform(*a);
    }
    ntt::radix4(&mont, proot, &mut x);
    for a in &mut x {
        *a = mont.mul(*a, *a);
    }
    ntt::radix4(&mont, proot_inv, &mut x);
    for a in &mut x {
        *a = mont.mul(*a, n_inv);
        *a = mont.reduce(*a);
    }

    let x = &x[..=n_max as usize];
    for _ in 0..input.value() {
        let a: u32 = input.value();
        let count = x[a as usize].div_ceil(2);
        writeln!(output, "{}", count).unwrap();
    }
}
