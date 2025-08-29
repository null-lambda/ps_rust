use std::io::Write;

use dirichlet_prefix_sum::XudyhSieve;

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

pub mod dirichlet_prefix_sum {
    use std::collections::HashMap;

    pub type K = i32;
    pub type T = i32;

    // Let f and g be multiplicative functions.
    // Given O(1) computation oracles for prefix(g * f) and prefix(g),
    // compute prefix(f)(N) in O(N^(3/4)).
    //
    // With a suitable precomputed vector `small` of size O(N^(2/3)),
    // the time complexity reduces to O(N^(2/3)).
    pub struct XudyhSieve<PGF, PG> {
        prefix_g: PG,
        prefix_g_conv_f: PGF,
        g1: T,

        small: Vec<T>,
        large: HashMap<K, T>,
    }

    impl<PG: FnMut(K) -> T, PGF: FnMut(K) -> T> XudyhSieve<PGF, PG> {
        pub fn new(mut prefix_g: PG, prefix_g_conv_f: PGF, small: Vec<T>) -> Self {
            let g1 = prefix_g(1) - prefix_g(0);
            Self {
                prefix_g,
                prefix_g_conv_f,
                g1,
                small,
                large: Default::default(),
            }
        }

        pub fn get(&mut self, n: K) -> T {
            if n < self.small.len() as K {
                return self.small[n as usize];
            }
            if let Some(&res) = self.large.get(&n) {
                return res;
            }

            let mut res = (self.prefix_g_conv_f)(n);
            let mut d = 2;
            while d <= n {
                let t = n / d;
                let d_end = n / t;

                res -= self.get(t) * ((self.prefix_g)(d_end) - (self.prefix_g)(d - 1));

                d = d_end + 1;
            }
            res /= self.g1;

            self.large.insert(n, res);
            res
        }
    }
}

fn isqrt(n: u64) -> u64 {
    if n <= 4_503_599_761_588_223u64 {
        return (n as f64).sqrt() as u64;
    }

    let mut x = n;
    loop {
        let next_x = (x + n / x) / 2;
        if next_x >= x {
            return x;
        }
        x = next_x;
    }
}

fn partition_point<P>(mut left: u64, mut right: u64, mut pred: P) -> u64
where
    P: FnMut(u64) -> bool,
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

    let (mpf, _) = linear_sieve(1e7 as u32);
    let mu = gen_mobius(&mpf);
    let mut mertens = mu.into_iter().map(i32::from).collect::<Vec<_>>();
    for i in 1..mertens.len() {
        mertens[i] += mertens[i - 1];
    }
    let mut mertens = XudyhSieve::new(|n| n, |n| if n == 0 { 0 } else { 1 }, mertens);

    let mut count_prefix = |n: i64| {
        let n_sqrt = isqrt(n as u64) as i64;

        // let d_brute_cutoff = -1;
        let d_brute_cutoff = (n as f64).powf(1.0 / 3.0 + 1e-9) as i64 + 10;

        let mut res = 0;
        let mut d = 1i64;
        while d <= n_sqrt {
            let t = n / (d * d);
            let d_end = if d <= d_brute_cutoff {
                d
            } else {
                isqrt((n / t) as u64) as i64
            };

            let dm = (mertens.get(d_end as i32) - mertens.get(d as i32 - 1)) as i64;
            res += dm * t;

            d = d_end + 1;
        }
        res
    };

    let i: i64 = input.value();

    use std::f64::consts as c;
    let ratio = 6.0 / (c::PI * c::PI);
    let approx = i as f64 / ratio;
    let error = (i as f64).sqrt() / (ratio * ratio) + 2.0;

    let lower = if i <= 1e15 as i64 {
        0
    } else {
        (approx - error).max(0.0) as u64
    };
    let upper = ((approx + error) as u64).max(100);

    let ans = partition_point(lower, upper, |n| count_prefix(n as i64) < i);
    writeln!(output, "{}", ans).unwrap();
}
