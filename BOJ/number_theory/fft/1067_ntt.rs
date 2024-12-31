use std::io::Write;

use num::Montgometry;

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

        pub fn reduce(&self, x: u64) -> u32 {
            debug_assert!((x as u64) < (self.m as u64) * (self.m as u64));
            let q = (x as u32).wrapping_mul(self.m_inv);
            let a = ((q as u64 * self.m as u64) >> 32) as u32;
            let mut res = (x >> 32) as u32 + self.m - a;
            if res >= self.m {
                res -= self.m;
            }
            res as u32
        }

        pub fn multiply(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            self.reduce(x as u64 * y as u64)
        }

        pub fn transform(&self, x: u32) -> u32 {
            debug_assert!(x < self.m);
            self.multiply(x, self.r2)
        }

        pub fn add(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            let sum = x + y;
            if sum >= self.m {
                sum - self.m
            } else {
                sum
            }
        }

        pub fn sub(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            if x >= y {
                x - y
            } else {
                x + self.m - y
            }
        }

        pub fn pow(&self, mut base: u32, mut exp: u32) -> u32 {
            let mut res = self.transform(1);
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.multiply(res, base);
                }
                base = self.multiply(base, base);
                exp >>= 1;
            }
            res
        }

        pub fn inv(&self, n: u32) -> u32 {
            // m must be prime
            self.pow(n, self.m - 2)
        }
    }

    impl Montgometry<u64> {
        pub fn new(m: u64) -> Self {
            debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
            let mut m_inv = 1u64;
            for _ in 0..6 {
                m_inv = m_inv.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(m_inv)));
            }
            let r = m.wrapping_neg() % m;
            let r2 = (r as u128 * r as u128 % m as u128) as u64;

            Self { m, m_inv, r2 }
        }

        pub fn reduce(&self, x: u128) -> u64 {
            debug_assert!((x as u128) < (self.m as u128) * (self.m as u128));
            let q = (x as u64).wrapping_mul(self.m_inv);
            let a = ((q as u128 * self.m as u128) >> 64) as u64;
            let mut res = (x >> 64) as u64 + self.m - a;
            if res >= self.m {
                res -= self.m;
            }
            res as u64
        }

        pub fn multiply(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            self.reduce(x as u128 * y as u128)
        }

        pub fn transform(&self, x: u64) -> u64 {
            debug_assert!(x < self.m);
            self.multiply(x, self.r2)
        }

        pub fn add(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            let sum = x + y;
            if sum >= self.m {
                sum - self.m
            } else {
                sum
            }
        }

        pub fn sub(&self, x: u64, y: u64) -> u64 {
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

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % m;
        }
        base = base * base % m;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: u64, p: u64) -> u64 {
    mod_pow(n, p - 2, p)
}

pub mod ntt {
    use std::iter;

    use crate::num::Montgometry;

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

    pub fn radix4_u32(mont: &Montgometry<u32>, proot: u32, xs: &mut [u32]) {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        bit_reversal_perm(xs);

        let base: Vec<u32> = (0..n_log2)
            .scan(proot, |acc, _| {
                let prev = *acc;
                *acc = mont.multiply(*acc, *acc);
                Some(prev)
            })
            .collect();

        let mut proot_pow = vec![0; n]; // Cache-friendly twiddle factors
        proot_pow[0] = mont.transform(1);

        let quartic_root = mont.pow(proot, n as u32 / 4);

        let update_proot_pow = |proot_pow: &mut [u32], k: u32| {
            let step = 1 << k;
            let base = base[(n_log2 - k - 1) as usize];
            for i in (0..step).rev() {
                proot_pow[i * 2 + 1] = mont.multiply(proot_pow[i], base);
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
                    *a0 = mont.add(b0, b1);
                    *a1 = mont.sub(b0, b1);
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
                    let pow2 = mont.multiply(pow1, pow1);
                    let pow1_shift = mont.multiply(pow1, quartic_root);

                    let b0 = *a0;
                    let b1 = mont.multiply(*a1, pow2);
                    let b2 = *a2;
                    let b3 = mont.multiply(*a3, pow2);

                    let c0 = mont.add(b0, b1);
                    let c1 = mont.sub(b0, b1);
                    let c2 = mont.multiply(mont.add(b2, b3), pow1);
                    let c3 = mont.multiply(mont.sub(b2, b3), pow1_shift);

                    *a0 = mont.add(c0, c2);
                    *a1 = mont.add(c1, c3);
                    *a2 = mont.sub(c0, c2);
                    *a3 = mont.sub(c1, c3);
                }
            }
            k += 2;
        }
    }

    // naive O(n^2)
    pub fn naive_u64(mont: &Montgometry<u64>, proot: u64, xs: &mut [u64]) {
        let n = xs.len().next_power_of_two();
        let proot_pow: Vec<u64> = iter::successors(Some(mont.transform(1)), |&acc| {
            Some(mont.multiply(acc, proot))
        })
        .take(n)
        .collect();
        let res: Vec<_> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| mont.multiply(xs[j], proot_pow[(i * j) % n]))
                    .fold(0, |acc, x| mont.add(acc, x))
            })
            .collect();
        xs.copy_from_slice(&res);
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n_orig: usize = input.value();
    let mut xs: Vec<u32> = (0..n_orig).map(|_| input.value()).collect();
    let mut ys: Vec<u32> = (0..n_orig).map(|_| input.value()).collect();
    ys.reverse();

    let n = n_orig.next_power_of_two() * 2;
    xs.resize(n, 0);
    let (left, right) = xs[0..n_orig * 2].split_at_mut(n_orig);
    right.copy_from_slice(&left);

    ys.resize(n, 0);

    let p = 998_244_353;
    let mont = Montgometry::<u32>::new(p);
    let gen = 3;

    assert!((p - 1) % n as u32 == 0);
    let mut proot = mod_pow(gen as u64, (p - 1) as u64 / n as u64, p as u64) as u32;
    let mut proot_inv = mod_inv(proot as u64, p as u64) as u32;
    let mut n_inv = mod_inv(n as u64, p as u64) as u32;

    proot = mont.transform(proot);
    proot_inv = mont.transform(proot_inv);
    n_inv = mont.transform(n_inv);

    for x in &mut xs {
        *x = mont.transform(*x);
    }
    for y in &mut ys {
        *y = mont.transform(*y);
    }

    ntt::radix4_u32(&mont, proot, &mut xs);
    ntt::radix4_u32(&mont, proot, &mut ys);
    for (x, y) in xs.iter_mut().zip(&ys) {
        *x = mont.multiply(*x, *y);
    }
    ntt::radix4_u32(&mont, proot_inv, &mut xs);

    for x in &mut xs {
        *x = mont.multiply(*x, n_inv);
        *x = mont.reduce(*x as u64);
    }

    let result = xs[n_orig - 1..2 * n_orig - 1].iter().max().unwrap();
    writeln!(output, "{}", result).unwrap();
}
