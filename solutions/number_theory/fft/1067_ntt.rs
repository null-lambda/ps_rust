use std::io::Write;

use num::MontgometryU32;
use num::MontgometryU64;

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
    pub struct MontgometryU32 {
        pub m: u32,
        m_inv: u32,
        r2: u32,
    }

    impl MontgometryU32 {
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
            let mut res = 1;
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

    // Montgomery reduction
    pub struct MontgometryU64 {
        pub m: u64,
        m_inv: u64,
        r2: u64,
    }

    impl MontgometryU64 {
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

    use crate::num::{MontgometryU32, MontgometryU64};

    pub fn u32(mont: &MontgometryU32, proot: u32, xs: &mut [u32]) {
        assert!(xs.len().is_power_of_two());
        let n = xs.len();
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;

        let brp = bit_reversal_perm(n);
        for i in 0..n as u32 {
            let rev = brp[i as usize];
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }

        let proot_pow: Vec<u32> = (0..n_log2)
            .scan(proot, |acc, _| {
                let prev = *acc;
                *acc = mont.multiply(*acc, *acc);
                Some(prev)
            })
            .collect();

        for k in 0..n_log2 {
            let step = 1 << k;
            let base = proot_pow[(n_log2 - 1 - k) as usize];
            for i in (0..n).step_by(step * 2) {
                let mut pow = mont.transform(1);
                for j in 0..step {
                    let even = xs[i + j];
                    let odd = mont.multiply(xs[i + j + step], pow);
                    xs[i + j] = mont.add(even, odd);
                    xs[i + j + step] = mont.sub(even, odd);
                    pow = mont.multiply(pow, base);
                }
            }
        }
    }

    pub fn u64(mont: &MontgometryU64, proot: u64, xs: &mut [u64]) {
        assert!(xs.len().is_power_of_two());
        let n = xs.len();
        let n_log2 = u64::BITS - (n as u64).leading_zeros() - 1;

        for i in 0..n as u32 {
            let rev = reverse_bits(n_log2, i);
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }

        let proot_pow: Vec<u64> = (0..n_log2)
            .scan(proot, |acc, _| {
                let prev = *acc;
                *acc = mont.multiply(*acc, *acc);
                Some(prev)
            })
            .collect();

        for k in 0..n_log2 {
            let step = 1 << k;
            let base = proot_pow[(n_log2 - 1 - k) as usize];
            for i in (0..n).step_by(step * 2) {
                let mut pow = mont.transform(1);
                for j in 0..step {
                    let even = xs[i + j];
                    let odd = mont.multiply(xs[i + j + step], pow);
                    xs[i + j] = mont.add(even, odd);
                    xs[i + j + step] = mont.sub(even, odd);
                    pow = mont.multiply(pow, base);
                }
            }
        }
    }

    // naive O(n^2)
    pub fn naive_u64(mont: &MontgometryU64, proot: u64, xs: &mut [u64]) {
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
    let mont = MontgometryU32::new(p);
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

    ntt::u32(&mont, proot, &mut xs);
    ntt::u32(&mont, proot, &mut ys);
    for (x, y) in xs.iter_mut().zip(&ys) {
        *x = mont.multiply(*x, *y);
    }
    ntt::u32(&mont, proot_inv, &mut xs);

    for x in &mut xs {
        *x = mont.multiply(*x, n_inv);
        *x = mont.reduce(*x as u64);
    }

    let result = xs[n_orig - 1..2 * n_orig - 1].iter().max().unwrap();
    writeln!(output, "{}", result).unwrap();
}
