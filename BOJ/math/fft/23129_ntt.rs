use std::{collections::HashMap, io::Write};

use num_mod::{InvOp, ModOp, Montgomery, PowBy};

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

pub mod num_mod {
    use std::ops::*;

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
            (lhs + rhs) % self.m
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

    // Montgomery reduction
    #[derive(Debug, Clone)]
    pub struct Montgomery<T> {
        m: T,
        m_inv: T,
        r2: T,
    }

    impl Montgomery<u32> {
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
    }

    impl ModOp<u32> for Montgomery<u32> {
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

        fn reduce(&self, x: u32) -> u32 {
            self.reduce_double(x as u64)
        }

        fn transform(&self, x: u32) -> u32 {
            debug_assert!(x < self.m);
            self.mul(x, self.r2)
        }
    }

    impl Montgomery<u64> {
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
    }

    impl ModOp<u64> for Montgomery<u64> {
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

        fn reduce(&self, x: u64) -> u64 {
            self.reduce_double(x as u128)
        }

        fn transform(&self, x: u64) -> u64 {
            debug_assert!(x < self.m);
            self.mul(x, self.r2)
        }
    }
}

pub mod ntt {
    use std::iter;

    use crate::num_mod::{ModOp, PowBy};

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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let edges: Vec<(u32, u32)> = (0..m)
        .map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1))
        .map(|(u, v)| if u <= v { (u, v) } else { (v, u) })
        .collect();
    let mut edges_mult: HashMap<(u32, u32), u32> = Default::default();

    let r = 2 * (m + 1).next_power_of_two();
    let mut degree = vec![0; n];
    for &(u, v) in &edges {
        if u == v {
            degree[u as usize] += 1;
        } else {
            degree[u as usize] += 1;
            degree[v as usize] += 1;
        }
        *edges_mult.entry((u, v)).or_default() += 1;
    }
    let mut degree_poly = vec![0; r];
    for &d in &degree {
        degree_poly[d] += 1;
    }

    let p = 9223372036737335297;
    let mod_op = Montgomery::<u64>::new(p);
    let gen = mod_op.transform(3);
    let proot = mod_op.pow(gen, (p - 1) / r as u64);
    let proot_inv = mod_op.inv(proot);

    // Square polynomial with NTT
    for a in &mut degree_poly {
        *a = mod_op.transform(*a);
    }
    ntt::radix4(&mod_op, proot, &mut degree_poly);
    for a in &mut degree_poly {
        *a = mod_op.mul(*a, *a);
    }
    ntt::radix4(&mod_op, proot_inv, &mut degree_poly);
    let r_inv = mod_op.inv(mod_op.transform(r as u64));
    for a in &mut degree_poly {
        *a = mod_op.mul(*a, r_inv);
        *a = mod_op.reduce(*a);
    }
    let degree_poly = &mut degree_poly[..r - 1];
    for (&(u, v), &mult) in &edges_mult {
        let du = degree[u as usize];
        let dv = degree[v as usize];
        if u == v {
            degree_poly[(2 * du) as usize] -= 1;
            degree_poly[(2 * du - mult as usize) as usize] += 1;
        } else {
            degree_poly[(du + dv) as usize] -= 2;
            degree_poly[(du + dv - mult as usize) as usize] += 2;
        }
    }
    for u in 0..n {
        let self_mult = edges_mult.get(&(u as u32, u as u32)).copied().unwrap_or(0);
        degree_poly[(2 * degree[u] - self_mult as usize) as usize] -= 1;
    }
    for d in 0..r - 1 {
        degree_poly[d] /= 2;
    }

    let mut postfix_sum = vec![0u64; r];
    for i in (0..r - 1).rev() {
        postfix_sum[i] = degree_poly[i] + postfix_sum[i + 1];
    }

    for _ in 0..input.value() {
        let lb = input.value::<usize>() + 1;
        let ans = postfix_sum.get(lb).unwrap_or(&0);
        writeln!(output, "{}", ans).unwrap();
    }
}
