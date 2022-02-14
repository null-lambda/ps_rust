mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};
use std::ops::{Add, Mul, Neg, Sub};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Complex64 {
    re: f64,
    im: f64,
}

impl Complex64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Complex64 {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Complex64 {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self {
        Complex64 {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Complex64 {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl From<f64> for Complex64 {
    fn from(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
}

// simple O(n^2) dfs
fn fft_simple(xs: &[Complex64]) -> Vec<Complex64> {
    let n = xs.len().next_power_of_two();
    let primitive_root_pow: Vec<Complex64> = (0..n)
        .map(|i| {
            use std::f64::consts::PI;
            let theta = -2.0 * PI * (i as f64) / (n as f64);
            Complex64::new(theta.cos(), theta.sin())
        })
        .collect();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| xs[j] * primitive_root_pow[(i * j) % n])
                .fold(0.0.into(), |acc, x| acc + x)
        })
        .collect()
}

fn fft(xs: &mut [Complex64]) {
    assert!(xs.len().is_power_of_two());
    let n = xs.len().next_power_of_two();

    let mut n_log2 = 0;
    while 1 << n_log2 < n {
        n_log2 += 1;
    }

    for i in 0usize..n {
        let rev = i.reverse_bits() >> (0usize.leading_zeros() - n_log2);
        if i < rev {
            xs.swap(i, rev);
        }
    }

    for step in (0..n_log2).map(|s| 1 << s) {
        use std::f64::consts::PI;
        let theta = -PI / (step as f64);
        let proot = Complex64::new(theta.cos(), theta.sin());
        for i in (0..n).step_by(step * 2) {
            let mut proot_pow = 1.0.into();
            for j in 0..step {
                let (even, odd) = (xs[i + j], xs[i + j + step]);
                xs[i + j] = even + odd * proot_pow;
                xs[i + j + step] = even - odd * proot_pow;
                proot_pow = proot_pow * proot;
            }
        }
    }
}

fn inv_fft(xs: &mut [Complex64]) {
    let n = xs.len();
    for x in xs.iter_mut() {
        *x = x.conj();
    }
    fft(xs);
    for x in xs.iter_mut() {
        *x = x.conj() * (1.0 / (n as f64)).into();
    }
}

fn gen_primes(upper_bound: u64) -> Vec<u64> {
    let mut sieve = vec![true; upper_bound as usize + 1];
    sieve[0] = false;
    sieve[1] = false;

    let mut primes = vec![2];
    for p in (3..=upper_bound as usize).step_by(2) {
        if sieve[p] {
            primes.push(p as u64);
            for j in (p * 2..=upper_bound as usize).step_by(p) {
                sieve[j] = false;
            }
        }
    }
    primes
}

#[test]
fn test() {
    let p = gen_primes(1000000);
    assert!(false);
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let upper_bound: usize = 1_000_000;

    let primes = gen_primes(upper_bound as u64);
    let xs = primes[1..].iter().map(|p| (p - 1) / 2);
    let ys = primes
        .iter()
        .map(|&p| p)
        .take_while(|&x| x <= (upper_bound as u64 / 2));

    // let deg_bound = xs.clone().max().unwrap().max(ys.clone().max().unwrap());
    // println!("{}", deg_bound);
    let deg_bound = 499991;

    fn build_poly(deg_bound: usize, iter: impl Iterator<Item = u64>) -> Vec<Complex64> {
        let n = (2 * deg_bound + 1).next_power_of_two();
        let mut f = vec![0.0.into(); n];
        for x in iter {
            f[x as usize] = 1.0.into();
        }
        f
    }

    let mut fx = build_poly(deg_bound, xs);
    let mut fy = build_poly(deg_bound, ys);

    let prod_fx_fy = {
        fft(&mut fx);
        fft(&mut fy);

        assert_eq!(fx.len(), fy.len());
        for i in 0..fx.len() {
            fx[i] = fx[i] * fy[i];
        }
        drop(fy);

        inv_fft(&mut fx);
        // fft(&mut fx);
        fx
    };

    let t = input.value();
    for _ in 0..t {
        let n: usize = input.value();
        let result = (prod_fx_fy[(n - 1) / 2].re).round() as u64;
        writeln!(output_buf, "{}", result).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
