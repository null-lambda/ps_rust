use std::io::Write;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
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

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut c, mut x, mut y) = if a.abs() > b.abs() {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 != 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - q * x.1));
        y = (y.1, (y.0 - q * y.1));
        c = (c.1, c.0 - q * c.1);
    }

    if c.0 < 0 {
        (-c.0, -x.0, -y.0)
    } else {
        (c.0, x.0, y.0)
    }
}

fn crt(a1: u64, m1: u64, a2: u64, m2: u64) -> Option<(u64, u64)> {
    let (d, x, _y) = egcd(m1 as i64, m2 as i64);
    let m = m1 / d as u64 * m2;
    let da = ((a2 as i64 - a1 as i64) % m as i64 + m as i64) as u64 % m;
    if da % d as u64 != 0 {
        return None;
    }
    let mut x = ((x % m as i64) + m as i64) as u64 % m;
    x = (da / d as u64 % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}

fn p_adic_decomp(mut n: u32, p: u32) -> (u32, u32) {
    let mut exp = 0;
    while n % p == 0 {
        n /= p;
        exp += 1;
    }
    (n, exp)
}

fn mpow(mut base: u64, mut exp: u64, m: u64) -> u64 {
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

fn minv(n: u64, p: u64) -> u64 {
    mpow(n, p - 2, p)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let (_mpf, primes) = linear_sieve(5e5 as u32);

    for _ in 0..input.value() {
        let n: usize = input.value();
        let mut m: u32 = input.value();

        const UNSET: u32 = !0;
        let mut parent = vec![UNSET; n];
        let mut degree = vec![1u32; n];
        for u in 1..n as u32 {
            let p = input.value::<u32>() - 1;
            parent[u as usize] = p;
            degree[p as usize] += 1;
        }
        degree[0] += 1;

        let mut size = vec![1u32; n];
        for mut u in 0..n as u32 {
            while degree[u as usize] == 1 {
                let p = parent[u as usize];
                degree[u as usize] -= 1;
                degree[p as usize] -= 1;

                size[p as usize] += size[u as usize];

                u = p;
            }
        }

        let mut factors = vec![];
        {
            let sqrt = (m as f64).sqrt() as u32;
            for &p in &primes {
                if p >= sqrt + 2 {
                    break;
                }

                let (m_next, e) = p_adic_decomp(m, p);
                m = m_next;
                if e >= 1 {
                    factors.push((p, e));
                }
            }

            if m >= 2 {
                factors.push((m as u32, 1));
            }
        }

        let (ans, _) = factors
            .into_iter()
            .map(|(p, e)| {
                let m = p.pow(e);

                let mut numer = 1u64;
                let mut denom = 1u64;

                let mut e_acc = 0;
                for s in 1..=n as u32 {
                    let (s, es) = p_adic_decomp(s, p);
                    numer = numer * s as u64 % m as u64;
                    e_acc += es;
                }
                for &s in &size {
                    let (s, es) = p_adic_decomp(s, p);
                    denom = denom * s as u64 % m as u64;
                    e_acc -= es;
                }

                let phi = m / p * (p - 1);
                let mut ans = numer * mpow(denom, phi as u64 - 1, m as u64) % m as u64;
                ans = ans * mpow(p as u64, e_acc as u64, m as u64) % m as u64;

                (ans, m as u64)
            })
            .fold((0, 1), |(a, m), (b, l)| crt(a, m, b, l).unwrap());

        writeln!(output, "{}", ans).unwrap();
    }
}
