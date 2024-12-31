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

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }
    result
}

// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: u64, b: u64) -> (u64, i64, i64) {
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - (q as i64) * x.1));
        y = (y.1, (y.0 - (q as i64) * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}
fn crt(a1: u64, m1: u64, a2: u64, m2: u64) -> Option<(u64, u64)> {
    let (d, x, _y) = egcd(m1, m2);
    debug_assert!(d == 1);
    let m = m1 * m2;
    let a2_m_a1 = ((a2 as i64 - a1 as i64) % m as i64 + m as i64) as u64 % m;
    if a2_m_a1 % d != 0 {
        return None;
    }
    let mut x = ((x % m as i64) + m as i64) as u64 % m;
    x = (a2_m_a1 / d % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}

fn solve_naive(a: u64, b: u64) -> Option<u64> {
    let bound = 10_000_000;
    assert!(a < bound);
    if a == 1 {
        return Some(1);
    }

    let step = |x: u64| -> Option<u64> {
        let mut res = 1;
        for _ in 0..x {
            res = res * a;
            if res >= bound {
                return None;
            }
        }
        Some(res)
    };

    match b {
        0 => unimplemented!(),
        1 => Some(a),
        2.. => {
            let mut y = a;
            for _ in 0..b - 1 {
                y = step(y)?;
            }
            (y < bound).then(|| y)
        }
    }
}

fn solve(a: u64, b: u64, p: u64, m: u64, phi_m: u64) -> u64 {
    if a % p == 0 {
        let bound = 8;
        let step = |x: u64| -> u64 {
            let mut res = 1;
            for _ in 0..x {
                res = res * a;
                if res >= bound {
                    return bound;
                }
            }
            res
        };
        let step_final = |x: u64| mod_pow(a, x, m);
        return match b {
            0 => unimplemented!(),
            1 => a % m,
            2.. => {
                let mut y = a;
                for _ in 0..b - 2 {
                    y = step(y);
                }
                y = step_final(y);
                y
            }
        };
    }

    let step = |x: u64| mod_pow(a, x, phi_m);
    let step_final = |x: u64| mod_pow(a, x, m);

    match b {
        0 => unimplemented!(),
        1 => a % m,
        2.. => {
            let mut y = a;
            for _ in 0..b - 2 {
                y = step(y);
            }
            y = step_final(y);
            y
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let a: u64 = input.value();
    let b: u64 = input.value();

    if let Some(x) = solve_naive(a, b) {
        writeln!(output, "{}", x).unwrap();
        return;
    }
    let x2 = solve(a, b, 2, 2u64.pow(8), 2u64.pow(7));
    let x5 = solve(a, b, 5, 5u64.pow(8), (5 - 1) * 5u64.pow(7));
    let x = crt(x2, 2u64.pow(8), x5, 5u64.pow(8)).unwrap().0;
    writeln!(output, "{:08}", x).unwrap();
}
