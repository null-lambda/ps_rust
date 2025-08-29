use std::{collections::HashMap, io::Write};

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

fn factorize(min_prime_factor: &[u32], n: u32) -> Vec<(u32, u8)> {
    let mut factors = Vec::new();
    let mut x = n;
    while x > 1 {
        let p = min_prime_factor[x as usize];
        if p == 0 {
            factors.push((x as u32, 1));
            break;
        }
        let mut exp = 0;
        while x % p == 0 {
            exp += 1;
            x /= p;
        }
        factors.push((p, exp));
    }

    factors
}

fn for_each_divisor(factors: &[(u32, u8)], mut visitor: impl FnMut(u32)) {
    let mut stack = vec![(1, 0u32)];
    while let Some((mut d, i)) = stack.pop() {
        if i as usize == factors.len() {
            visitor(d);
        } else {
            let (p, exp) = factors[i as usize];
            for _ in 0..=exp {
                stack.push((d, i + 1));
                d *= p;
            }
        }
    }
}

pub struct Mertens {
    small: Vec<i64>,
    large: HashMap<i64, i64>,
}

impl Mertens {
    pub fn new(min_prime_factor: &[u32]) -> Self {
        let mu = gen_mobius(&min_prime_factor);
        let mut small = mu.iter().map(|&x| x as i64).collect::<Vec<_>>();
        for i in 1..mu.len() {
            small[i] += small[i - 1];
        }

        Self {
            small,
            large: Default::default(),
        }
    }

    pub fn get(&mut self, n: i64) -> i64 {
        if n < self.small.len() as i64 {
            return self.small[n as usize];
        }
        if let Some(&res) = self.large.get(&n) {
            return res;
        }

        let mut res = 1;
        let mut d = 2;
        loop {
            let t = n / d;
            let d_end = n / t;

            res -= self.get(t) * (d_end - d + 1);

            if d_end == n {
                break;
            }
            d = d_end + 1;
        }

        self.large.insert(n, res);
        res
    }

    pub fn even(&mut self, mut n: i64) -> i64 {
        let mut res = 0;
        loop {
            n /= 2;
            if n == 0 {
                break;
            }
            res -= self.get(n);
        }
        res
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let l: i64 = input.value();

    let (mpf, _) = linear_sieve(2.5e6 as u32);
    let mut mertens = Mertens::new(&mpf);

    let mut ans = 0;
    let mut d = 1;
    loop {
        let t = l / d;
        let d_end = l / t;

        let dm = mertens.get(d_end) - mertens.get(d - 1);
        let dm_even = mertens.even(d_end) - mertens.even(d - 1);
        let dm_odd = dm - dm_even;

        ans += (dm_even * t + dm_odd * (t / 2)) * t;

        if d_end == l {
            break;
        }
        d = d_end + 1;
    }
    writeln!(output, "{}", ans).unwrap();
}
