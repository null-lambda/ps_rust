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

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    const COORD_MIN: isize = -30000;
    const COORD_MAX: isize = 30000;
    const COORD_RANGE: usize = (COORD_MAX - COORD_MIN) as usize;

    let build_poly = |input: &mut &[u8]| -> (usize, Vec<Complex64>) {
        let n = input.value();
        let mut f = vec![0.0.into(); (COORD_RANGE * 2 + 1).next_power_of_two()];
        for _ in 0..n {
            let d = input.value::<isize>() - COORD_MIN;
            f[d as usize] = 1.0.into();
        }
        (n, f)
    };

    let (_, mut fx) = build_poly(&mut input);
    let ny = input.value();
    let ys: Vec<usize> = (0..ny)
        .map(|_| (input.value::<isize>() - COORD_MIN) as usize)
        .collect();
    let (_, mut fz) = build_poly(&mut input);

    let prod_fx_fz = {
        fft(&mut fx);
        fft(&mut fz);

        assert_eq!(fx.len(), fz.len());
        for i in 0..fx.len() {
            fx[i] = fx[i] * fz[i];
        }
        drop(fz);

        inv_fft(&mut fx);
        fx
    };

    let result: usize = ys
        .iter()
        .map(|&y| prod_fx_fz[2 * y].re.round() as usize)
        .sum();
    writeln!(output_buf, "{}", result).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
