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

// a0 + a1 x in Z[x]/(1+x+x2)
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Z3([i64; 2]);

impl Z3 {
    fn rot(&self) -> Self {
        let [a, b] = self.0;
        Self([-b, a - b])
    }

    fn irot(&self) -> Self {
        let [a, b] = self.0;
        Self([b - a, -a])
    }
}

impl std::ops::Add for Z3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl std::ops::Sub for Z3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl std::ops::Mul for Z3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let [a, b] = self.0;
        let [c, d] = rhs.0;
        Self([a * c - b * d, a * d + b * c - b * d])
    }
}

type T = Z3;

fn propagate_subset_sum(xs: &mut [T], modifier: impl Fn([&mut T; 3])) {
    let mut w = 1;
    while w < xs.len() {
        for t in xs.chunks_exact_mut(w * 3) {
            let (t0, rest) = t.split_at_mut(w);
            let (t1, t2) = rest.split_at_mut(w);
            for ((x0, x1), x2) in t0.iter_mut().zip(t1).zip(t2) {
                modifier([x0, x1, x2])
            }
        }

        w *= 3;
    }
}

fn fwht3(xs: &mut [T]) {
    propagate_subset_sum(xs, |[x0, x1, x2]| {
        let y0 = *x0 + *x1 + *x2;
        let y1 = *x0 + x1.rot() + x2.irot();
        let y2 = *x0 + x1.irot() + x2.rot();
        *x0 = y0;
        *x1 = y1;
        *x2 = y2;
    });
}

fn fwht3_inv(xs: &mut [T]) {
    propagate_subset_sum(xs, |[x0, x1, x2]| {
        let y0 = *x0 + *x1 + *x2;
        let y1 = *x0 + x1.irot() + x2.rot();
        let y2 = *x0 + x1.rot() + x2.irot();
        *x0 = y0;
        *x1 = y1;
        *x2 = y2;
    });
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let pow3k = 3usize.pow(k as u32);

    let mut freq = vec![Z3([0, 0]); pow3k];
    for _ in 0..n {
        let s = input.token().as_bytes();

        let mut acc = 0;
        for &b in s {
            acc *= 3;
            acc += (b - b'1') as usize;
        }
        freq[acc].0[0] += 1;
    }
    let s2 = freq.iter().map(|x| x.0[0] * x.0[0]).sum::<i64>();

    fwht3(&mut freq);
    for i in 0..pow3k {
        freq[i] = freq[i] * freq[i] * freq[i];
    }
    fwht3_inv(&mut freq);

    let mut ans = freq[0].0[0];
    ans /= pow3k as i64;

    ans = ans - 3 * s2 + 2 * n as i64;
    ans /= 6;
    writeln!(output, "{}", ans).unwrap();
}
