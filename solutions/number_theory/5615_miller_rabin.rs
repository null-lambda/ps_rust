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

pub mod miller_rabin {
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

    pub fn is_prime_u32(n: u32) -> bool {
        if n < 2 {
            return false;
        }

        let base = [2, 7, 61];
        let n = n as u64;
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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let mut ans = 0;
    for _ in 0..input.value() {
        let a: u32 = input.value();
        if miller_rabin::is_prime_u32(2 * a + 1) {
            ans += 1;
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
