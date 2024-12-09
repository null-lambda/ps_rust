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
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let (_, primes) = linear_sieve(10_000);
    let i = primes.iter().position(|&p| p > 2000).unwrap();
    let steps = &primes[i..][..30];

    let k: usize = input.value();
    let n: u32 = input.value();

    for step in &steps[..k] {
        for i in 0..n {
            write!(output, "{} ", 1 + i * step).unwrap();
        }
        writeln!(output).unwrap();
    }
}
