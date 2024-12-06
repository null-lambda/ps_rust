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

    let t_max = 30000;
    let (_, primes) = linear_sieve(t_max);

    let mut triangle_count = vec![0; t_max as usize + 1];

    let n = primes.len();
    for i in 0..n {
        for j in i..n {
            for k in j..n {
                let p = primes[i];
                let q = primes[j];
                let r = primes[k];
                if r >= p + q {
                    break;
                }
                let s = p + q + r;
                if s > t_max {
                    break;
                }
                triangle_count[s as usize] += 1;
            }
        }
    }

    loop {
        let t: usize = input.value();
        if t == 0 {
            break;
        }
        let ans = triangle_count[t];
        writeln!(output, "{}", ans).unwrap();
    }
}
