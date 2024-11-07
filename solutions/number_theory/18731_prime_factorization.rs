use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
};

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

fn batch_factorization(primes: &Vec<u32>, xs: &Vec<u32>) -> HashMap<u32, Vec<u32>> {
    let n = xs.len();
    let mut xs_freq: BTreeMap<u32, u32> = BTreeMap::new();
    for &x in xs {
        if x > 1 {
            *xs_freq.entry(x).or_default() += 1;
        }
    }

    let mut res: HashMap<u32, Vec<u32>> = HashMap::new();

    'outer: while let Some((mut x, f)) = xs_freq.pop_last() {
        let p_bound = (x as f64).sqrt().ceil() as u32 + 1;
        for &p in primes {
            if p > p_bound {
                break;
            }

            if x % p == 0 {
                let mut exp = 0;

                while x % p == 0 {
                    x /= p;
                    exp += 1;
                }
                res.entry(p).or_insert_with(|| {
                    // let max_exp = (x_max as f64).log(p as f64).ceil() as usize;
                    let max_exp = 63;
                    vec![0; max_exp + 1]
                })[exp] += f;
                if x > 1 {
                    *xs_freq.entry(x).or_default() += f;
                }
                continue 'outer;
            }
        }

        // x is prime
        // res.entry(x).or_insert_with(|| vec![0, 0])[1] += f;
        res.entry(x).or_insert_with(|| {
            // let max_exp = (x_max as f64).log(x as f64).ceil() as usize;
            let max_exp = 63;
            vec![0; max_exp + 1]
        })[1] += f
    }

    for (_p, freq) in &mut res {
        let nonzero_count = freq[1..].iter().sum::<u32>();
        freq[0] = n as u32 - nonzero_count;
    }

    res
}

const P: u64 = 1_000_000_007;

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let x_max: u32 = 1_000_000_000;
    let (_, primes) = linear_sieve((x_max as f64).sqrt() as u32 + 1);

    let n: usize = input.value();
    let m: usize = input.value();
    let a_orig: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let b_orig: Vec<u32> = (0..m).map(|_| input.value()).collect();

    let a_exps = batch_factorization(&primes, &a_orig);
    let b_exps = batch_factorization(&primes, &b_orig);

    let mut primes = a_exps.keys().chain(b_exps.keys()).collect::<Vec<_>>();
    primes.sort_unstable();
    primes.dedup();

    let mut ans = 1u64;
    for &p in &primes {
        let a_min = a_exps
            .get(&p)
            .and_then(|freq| {
                freq.iter()
                    .enumerate()
                    .find(|&(_, &f)| f > 0)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        let b_min = b_exps
            .get(&p)
            .and_then(|freq| {
                freq.iter()
                    .enumerate()
                    .find(|&(_, &f)| f > 0)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        let lower_bound = a_min.max(b_min);
        if lower_bound == 0 {
            continue;
        }

        // count elements leq than lb
        let a_cnt = a_exps
            .get(&p)
            .map_or(n as u32, |freq| freq[..=lower_bound].iter().sum::<u32>());
        let b_cnt = b_exps
            .get(&p)
            .map_or(m as u32, |freq| freq[..=lower_bound].iter().sum::<u32>());

        let factor = pow(lower_bound as u64 + 1, a_cnt as u64)
            + pow(lower_bound as u64 + 1, b_cnt as u64)
            + P
            - 1;
        ans *= factor % P;
        ans %= P;
    }
    writeln!(output, "{}", ans).unwrap();
}
use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
};

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

fn batch_factorization(primes: &Vec<u32>, xs: &Vec<u32>) -> HashMap<u32, Vec<u32>> {
    let n = xs.len();
    let mut xs_freq: BTreeMap<u32, u32> = BTreeMap::new();
    for &x in xs {
        if x > 1 {
            *xs_freq.entry(x).or_default() += 1;
        }
    }

    let mut res: HashMap<u32, Vec<u32>> = HashMap::new();

    'outer: while let Some((mut x, f)) = xs_freq.pop_last() {
        let p_bound = (x as f64).sqrt().ceil() as u32 + 1;
        for &p in primes {
            if p > p_bound {
                break;
            }

            if x % p == 0 {
                let mut exp = 0;

                while x % p == 0 {
                    x /= p;
                    exp += 1;
                }
                res.entry(p).or_insert_with(|| {
                    // let max_exp = (x_max as f64).log(p as f64).ceil() as usize;
                    let max_exp = 63;
                    vec![0; max_exp + 1]
                })[exp] += f;
                if x > 1 {
                    *xs_freq.entry(x).or_default() += f;
                }
                continue 'outer;
            }
        }

        // x is prime
        // res.entry(x).or_insert_with(|| vec![0, 0])[1] += f;
        res.entry(x).or_insert_with(|| {
            // let max_exp = (x_max as f64).log(x as f64).ceil() as usize;
            let max_exp = 63;
            vec![0; max_exp + 1]
        })[1] += f
    }

    for (_p, freq) in &mut res {
        let nonzero_count = freq[1..].iter().sum::<u32>();
        freq[0] = n as u32 - nonzero_count;
    }

    res
}

const P: u64 = 1_000_000_007;

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let x_max: u32 = 1_000_000_000;
    let (_, primes) = linear_sieve((x_max as f64).sqrt() as u32 + 1);

    let n: usize = input.value();
    let m: usize = input.value();
    let a_orig: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let b_orig: Vec<u32> = (0..m).map(|_| input.value()).collect();

    let a_exps = batch_factorization(&primes, &a_orig);
    let b_exps = batch_factorization(&primes, &b_orig);

    let mut primes = a_exps.keys().chain(b_exps.keys()).collect::<Vec<_>>();
    primes.sort_unstable();
    primes.dedup();

    let mut ans = 1u64;
    for &p in &primes {
        let a_min = a_exps
            .get(&p)
            .and_then(|freq| {
                freq.iter()
                    .enumerate()
                    .find(|&(_, &f)| f > 0)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        let b_min = b_exps
            .get(&p)
            .and_then(|freq| {
                freq.iter()
                    .enumerate()
                    .find(|&(_, &f)| f > 0)
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        let lower_bound = a_min.max(b_min);
        if lower_bound == 0 {
            continue;
        }

        // count elements leq than lb
        let a_cnt = a_exps
            .get(&p)
            .map_or(n as u32, |freq| freq[..=lower_bound].iter().sum::<u32>());
        let b_cnt = b_exps
            .get(&p)
            .map_or(m as u32, |freq| freq[..=lower_bound].iter().sum::<u32>());

        let factor = pow(lower_bound as u64 + 1, a_cnt as u64)
            + pow(lower_bound as u64 + 1, b_cnt as u64)
            + P
            - 1;
        ans *= factor % P;
        ans %= P;
    }
    writeln!(output, "{}", ans).unwrap();
}
