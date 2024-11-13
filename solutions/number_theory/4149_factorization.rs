use std::io::Write;
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

mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Self {
            let mut seed = 0;
            unsafe {
                if std::arch::x86_64::_rdrand64_step(&mut seed) == 1 {
                    Self(seed)
                } else {
                    panic!("Failed to get entropy");
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            debug_assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

pub mod miller_rabin {
    fn mod_pow(mut base: u128, mut exp: u128, m: u128) -> u128 {
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

    pub fn is_prime_u64(n: u64) -> bool {
        if n < 2 {
            return false;
        }

        // let base = [2, 7, 61];
        let base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

        let n = n as u128;
        'outer: for a in base {
            if n == a {
                return true;
            }
            if n % a == 0 {
                return false;
            }
            let r = (n - 1).trailing_zeros();
            let d = (n - 1) >> r;
            let mut c = mod_pow(a, d, n);
            if c == 1 || c == n - 1 {
                continue;
            }
            for _ in 0..(n - 1).trailing_zeros() {
                c = c * c % n;
                if c == 1 {
                    return false;
                }
                if c == n - 1 {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn pollard_rho(n: u64, rng: &mut rand::SplitMix64) -> u64 {
    if n <= 1 {
        return 1;
    }
    if n % 2 == 0 {
        return 2;
    }
    if miller_rabin::is_prime_u64(n) {
        return n;
    }

    loop {
        let c = rng.range_u64(1..21.min(n)) as u128;
        let f = |x: u128| (x * x % n as u128 + c) % n as u128;

        let mut x = rng.range_u64(2..n) as u128;
        let mut y = x;
        let mut d;
        loop {
            x = f(x);
            y = f(f(y));
            d = gcd(n, (x as i64 - y as i64).abs() as u64);
            if d != 1 {
                break;
            }
        }
        if d != n {
            return d;
        }
    }
}

fn factorize(n: u64, rng: &mut rand::SplitMix64) -> Vec<u64> {
    let mut factors = vec![];
    let mut stack = vec![n];
    while let Some(n) = stack.pop() {
        if n == 1 {
            continue;
        }
        let p = pollard_rho(n, rng);
        if p == 1 {
            continue;
        }
        if p == n {
            factors.push(p);
            continue;
        }

        stack.push(p);
        stack.push(n / p);
    }
    factors.sort_unstable();
    factors
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: u64 = input.value();
    for p in factorize(n, &mut rand::SplitMix64::from_entropy()) {
        writeln!(output, "{}", p).unwrap();
    }
}
