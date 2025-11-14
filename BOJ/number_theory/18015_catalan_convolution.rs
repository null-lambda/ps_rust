// Fix B.
// $$F_i(x) = x + x (F_{i-1}(x) + F_{i+1}(x))     ( i = 0, ..., B-1 )$$
// $$F_{-1} = F_B = 0$$
// $$G = \sum_{i = 0 ... B-1} F_i$$
//
// Driven part: $F_i(x) = c(x)  \Rightarrow c(x) = x/(1-2x)$
//
// Homogeneous part: Assume $F_i(x) = a^i$. Then
// $$1 = x (a + a ^{-1})$$
// $$a = (1 - sqrt(1 - 4x^2))/(2x) = x C(x^2) = x + O(x^3)$$ where $C$ = OGF of Catalan seq., and
// $$[x^N] a^k = [2|N-k] [x^{(N-k)/2}] C(x)^k
//           = [2|N-k] (k/N) {N \choose (N-k)/2}$$
// which is known as Catalan convolution [https://codeforces.com/blog/entry/87585],
// and can be derived by Lagrange inversion.
//
// Solution:
// $$F_i(x) = c (1 - (a^{i+1} + a^{B-i})/(1 + a^{B+1}))$$
// $$G(x) = c (B - 2(a-a^{B+1})/((1-a)(1 + a^{B+1})))$$
//
// PIE? Introduce generating variable for $B$, and calculate it separately for
// $G^{(B)}(x)$ (both ends, $(1-y)^{-2}$ and $F_0^{(B)}(x)$ (upper end, $(1-y)^{-1}$. Then we obtain
// $$H^{(B)}(x) = G^{(B)}(x) - 2 G^{(B-1)}(x) + G^{(B-2)}(x) - F_0^{(B)}(x) + F_0^{(B-1)}(x)$$
// $$ Ans = [x^N] H^{(B)}(x) \equiv [x^N] c(x) S(a(x))
//        = \sum_{j=0 ... N-1} 2^{N-1-j} \sum_{k=1 ... j} S_k [x^j] a(x)^k$$
//        = \sum_{k=1 ... N-1} S_k \sum_{j=k ... N-1} 2^{N-1-j} [x^j] a(x)^k$$
//        = \sum_{k=1 ... N-1} S_k k \sum_{j=k ... N-1} 2^{N-1-j}
//            [2|j-k] (j-1)! / (((j+k)/2)!((j-k)/2)!)
//        = \sum_{k=1 ... N-1} S_k k \sum_{l=0 ... floor((N-1-k)/2)} 2^{N-1-k-2l}
//            (k+2l-1)! / ((k+l)!l!)
//        = ?
//        $$
//
// This is O(N^2). Something's wrong...
//
//
// Do a Lagrange inversion of composite polynomials. We already have a closed form
// $$S(t) = \sum_B c_B g^{B}(t) + d_B f^{B}(t)$$
// where
// $$g^{B}(t) = -2(t-t^{B+1})/((1-t)(1 + t^{B+1}))$$
// $$f^{B}(t) = -(t + t^B)/(1 + t^{B+1}))$$
//
// $$[x^j] S(a(x)) = j^{-1} [x^{j-1}] (1+x^2)^j S'(x)$$
// $$Ans = \sum_{j=0 ... N-1} 2^{N-1-j} [x^j] S(a(x))
//       = \sum_{j=1 ... N-1} 2^{N-1-j} [x^j] S(a(x))
//       = \sum_{j=1 ... N-1} (2^{N-1-j}/j) [x^{j-1}] (1+x^2)^j S'(x)
//       = \sum_{j=1 ... N-1} (2^{N-1-j}/j) [x^{j-1}] (1+x^2)^j S'(x)
//       = [x^{N-2}] S'(x) \sum_{j=0 ... N-2} (2x)^{N-2-j} (1+x^2)^{j+1}/(j+1)
//       = ?
//       $$
//
// Pitfalls: we need to compute ${N \choose i}$ modulo composite numbers, which is the real
// bottleneck.

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

fn p_adic_decomp(mut n: u64, p: u64) -> (u64, u32) {
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

fn quadratic_log() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let (_mpf, primes) = linear_sieve(5e5 as u32);

    let n: usize = input.value();
    let b: usize = input.value();
    // let m: u64 = input.value();
    let m = 1_000_000_000u64;

    let mut s = vec![0u64; n];
    for l in 0..=2.min(b) {
        let b_sub = b - l;
        let coeff = if l == 0 || l == 2 { 1 } else { m - 2 };

        let d = coeff * (m - 2) % m;
        for i in 1..n {
            let (q, r) = (i / (b_sub + 1), i % (b_sub + 1));

            if r != 0 {
                if q % 2 == 0 {
                    s[i] = (s[i] + d) % m;
                } else {
                    s[i] = (s[i] + m - d) % m;
                }
            }
        }
    }
    for l in 0..=1.min(b) {
        let b_sub = b - l;
        let coeff = if l == 0 { m - 1 } else { 1 };

        let d = coeff * (m - 1) % m;
        for i in 1..n {
            let (q, r) = (i / (b_sub + 1), i % (b_sub + 1));

            if r == 1 {
                if q % 2 == 0 {
                    s[i] = (s[i] + d) % m;
                } else {
                    s[i] = (s[i] + m - d) % m;
                }
            }
            if r == b_sub {
                if q % 2 == 0 {
                    s[i] = (s[i] + d) % m;
                } else {
                    s[i] = (s[i] + m - d) % m;
                }
            }
        }
    }

    let mut factors = vec![];
    {
        let mut m = m;
        let sqrt = (m as f64).sqrt() as u64;
        for &p in &primes {
            let p = p as u64;
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
            factors.push((m, 1));
        }
    }

    let (ans, _) = factors
        .into_iter()
        .map(|(p, e)| {
            let q = p.pow(e);
            let phi_q = q - q / p;

            let mut fc = vec![1u64; n + 1];
            let mut ifc = vec![1u64; n + 1];
            let mut fc_e = vec![0u32; n + 1];

            for i in 1..=n {
                let (f, e) = p_adic_decomp(i as u64, p);

                fc[i] = fc[i - 1] * f % q;
                fc_e[i] = fc_e[i - 1] + e;
            }

            ifc[n] = mpow(fc[n], phi_q - 1, q);
            for i in (1..=n).rev() {
                let (f, _e) = p_adic_decomp(i as u64, p);

                ifc[i - 1] = ifc[i] * f % q;
            }

            let mut ans = 0u64;
            for k in 1..n {
                for l in 0..=(n - 1 - k) / 2 {
                    let mut factor = 1;
                    let mut factor_e = 0;

                    factor = factor * fc[k] % q * ifc[k - 1] % q;
                    factor_e += fc_e[k] - fc_e[k - 1];

                    factor = factor * fc[k + 2 * l - 1] % q * ifc[k + l] % q * ifc[l] % q;
                    factor_e += fc_e[k + 2 * l - 1] - fc_e[k + l] - fc_e[l];

                    factor = factor * mpow(p, factor_e as u64, q) % q;

                    factor = factor * mpow(2, (n - 1 - k - 2 * l) as u64, q) % q;
                    factor = factor * s[k] % q;

                    ans = (ans + factor) % q;
                }
            }

            (ans, q as u64)
        })
        .fold((0, 1), |(a, m), (b, l)| crt(a, m, b, l).unwrap());

    writeln!(output, "{}", ans).unwrap();
}

fn linear_log() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let (_mpf, primes) = linear_sieve(5e5 as u32);

    let n: usize = input.value();
    let b: usize = input.value();
    let m: u64 = input.value();

    todo!()
    // let mut ans = 0u64;
    // writeln!(output, "{}", ans).unwrap();
}

fn main() {
    quadratic_log();
    // linear_log();
}
