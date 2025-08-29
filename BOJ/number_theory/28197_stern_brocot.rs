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

mod stern_brocot {
    pub type T = u32;

    fn partition_point_unbounded(mut pred: impl FnMut(T) -> bool) -> T {
        // Exponential search
        let mut left = 0;
        let mut right = 0;
        while pred(right) {
            left = right;
            right = right * 2 + 1;
            if right == T::MAX {
                return right;
            }
        }

        // Binary search
        while left < right {
            let mid = left + (right - left) / 2;
            if pred(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }

    // Let Q_N = { p/q: 1 <= p, 1 <= q <= N },
    //     P = { r in Q_N : pred(r) }.
    //
    // Find max (S U { 0/1 }) and min ((Q_N \ S) U { 1/0 }).
    pub fn partition_point(max_denom: T, mut pred: impl FnMut((T, T)) -> bool) -> [(T, T); 2] {
        assert!(max_denom >= 1);
        let mut l = (0, 1);
        let mut r = (1, 0);
        let mut m = (l.0 + r.0, l.1 + r.1);

        loop {
            let k = partition_point_unbounded(|k| {
                m.1 + l.1 * k <= max_denom && !pred((m.0 + l.0 * k, m.1 + l.1 * k))
            });
            if k > 0 {
                r = (m.0 + l.0 * (k - 1), m.1 + l.1 * (k - 1));
                m = (l.0 + r.0, l.1 + r.1);
                if m.1 > max_denom {
                    break;
                }
            }

            let k = partition_point_unbounded(|k| {
                m.1 + r.1 * (k + 1) <= max_denom && pred((m.0 + r.0 * (k + 1), m.1 + r.1 * (k + 1)))
            });
            l = (m.0 + r.0 * k, m.1 + r.1 * k);
            m = (l.0 + r.0, l.1 + r.1);
            if m.1 > max_denom {
                break;
            }
        }

        [l, r]
    }

    pub fn find_path_rle(mut x: (T, T)) -> Vec<(bool, T)> {
        assert!(x.0 != 0 && x.1 != 0);

        let mut is_right = true;
        let mut path = vec![];
        while x.1 != 0 {
            let q = x.0 / x.1;
            if q != 0 {
                path.push((is_right, q));
            }
            x = (x.1, x.0 % x.1);
            is_right ^= true;
        }

        let tail = &mut path.last_mut().unwrap().1;
        *tail -= 1;
        if *tail == 0 {
            path.pop();
        }

        path
    }

    pub fn locate_by_rle(path: impl IntoIterator<Item = (bool, T)>) -> (T, T) {
        let mut l = (0, 1);
        let mut r = (1, 0);
        let mut m = (l.0 + r.0, l.1 + r.1);

        for (is_right, len) in path {
            if !is_right {
                r = (m.0 + l.0 * (len - 1), m.1 + l.1 * (len - 1));
            } else {
                l = (m.0 + r.0 * (len - 1), m.1 + r.1 * (len - 1));
            }
            m = (l.0 + r.0, l.1 + r.1);
        }
        m
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

fn gen_euler_phi(min_prime_factor: &[u32]) -> Vec<u32> {
    let n_bound = min_prime_factor.len();
    let mut phi = vec![0; n_bound];
    phi[1] = 1;
    for i in 2..n_bound {
        let p = min_prime_factor[i as usize];
        phi[i] = if p == 0 {
            i as u32 - 1
        } else {
            let m = i as u32 / p;
            phi[m as usize] * if m % p == 0 { p } else { p - 1 }
        };
    }
    phi
}

fn gen_mobius(min_prime_factor: &[u32]) -> Vec<i8> {
    let n_max = min_prime_factor.len() - 1;
    let mut mu = vec![0; n_max + 1];
    mu[1] = 1;

    for i in 2..=n_max {
        let p = min_prime_factor[i];
        if p == 0 {
            mu[i] = -1;
        } else {
            let m = i as u32 / p;
            mu[i] = if m % p == 0 { 0 } else { -mu[m as usize] };
        }
    }
    mu
}

fn divisor_with_mobius(factors_unique: &[u64]) -> Vec<(u64, i8)> {
    let mut stack = vec![(1, 1i8, 0u32)];
    let mut res = vec![];
    while let Some((d, mu, i)) = stack.pop() {
        if i as usize == factors_unique.len() {
            res.push((d, mu));
        } else {
            let p = factors_unique[i as usize];
            stack.push((d, mu, i + 1));
            stack.push((d * p, -mu, i + 1));
        }
    }

    res
}

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut rng = rand::SplitMix64::from_entropy();

    for _ in 0..input.value() {
        let n: u32 = input.value();
        let k = input.value::<u32>() - 1;

        let b = n + 2;

        let mut factors = factorize(b as u64, &mut rng);
        factors.dedup();
        let mut ds = divisor_with_mobius(&factors);
        ds.sort_unstable();

        let count_le = |a: u32| {
            let mut res = 0;
            for &(d, mu) in &ds {
                res += (a / d as u32) as i64 * mu as i64;
            }
            res as u32
        };
        let a = partition_point(0, b, |x| count_le(x) <= k);
        if a == b {
            writeln!(output, "-1").unwrap();
            continue;
        }

        let mut path = stern_brocot::find_path_rle((a, b));
        assert!(path.len() >= 1);
        path[0].1 -= 1;

        let b0;
        let mut l = path.len();
        if path[0].1 != 0 {
            b0 = path[0].0;
        } else {
            l -= 1;
            b0 = path[1].0;
        }

        writeln!(output, "{} {}", l, b0 as u8).unwrap();
        for (_, s) in path {
            if s != 0 {
                write!(output, "{} ", s).unwrap();
            }
        }
        writeln!(output).unwrap();
    }
}
