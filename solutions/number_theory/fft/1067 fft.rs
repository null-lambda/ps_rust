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
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
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

    fn reverse_bits(n_log2: usize, mut x: usize) -> usize {
        let mut rev = 0;
        for _ in 0..n_log2 {
            rev <<= 1;
            if x & 1 != 0 {
                rev |= 1;
            }
            x >>= 1;
        }
        rev
    }

    for i in 0..n {
        let rev = reverse_bits(n_log2, i);
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

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let mut xs: Vec<Complex64> = (0..n).map(|_| input.value::<f64>().into()).collect();
    let mut ys: Vec<Complex64> = (0..n).map(|_| input.value::<f64>().into()).collect();
    ys.reverse();

    let n_extended = n.next_power_of_two() * 2;
    xs.resize(n_extended, 0.0.into());
    {
        let (left, right) = xs[0..n * 2].split_at_mut(n);
        right.copy_from_slice(&left);
    }
    ys.resize(n_extended, 0.0.into());

    // let c_fmt = |z: Complex64| format!("{:.1}+{:.1}I", z.re, z.im);
    // let vec_c_fmt = |zx: Vec<_>| zx.iter().map(|&z| c_fmt(z)).collect::<Vec<_>>().join(" ");
    // println!("{:?}", vec_c_fmt(xs.clone()));
    // println!("{:?}", vec_c_fmt(fft_simple(&xs).clone()));
    // let mut rs = xs.clone();
    // fft(&mut rs);
    // println!("{:?}", vec_c_fmt(rs.clone()));

    fft(&mut xs);
    fft(&mut ys);
    let mut prod: Vec<_> = (0..n_extended).map(|i| (xs[i] * ys[i]).conj()).collect();
    // inv_fft(&mut prod);
    fft(&mut prod);

    let result = prod[n - 1..2 * n - 1]
        .iter()
        .map(|z| (z.re / n_extended as f64).round() as i32)
        .max()
        .unwrap();
    writeln!(output_buf, "{}", result).unwrap();

    // println!("{:?}", vec_c_fmt(prod[n - 1..2 * n - 1].to_vec()));

    std::io::stdout().write(&output_buf[..]).unwrap();
}
