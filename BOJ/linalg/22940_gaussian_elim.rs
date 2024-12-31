use num::Frac64;
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

pub mod num {
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Debug, Clone, Copy)]
    pub struct Frac64(i64, i64);

    impl Frac64 {
        pub fn new(a: i64, b: i64) -> Self {
            assert!(b > 0);
            Self(a, b).normalize()
        }

        pub fn numer(&self) -> i64 {
            self.0
        }

        pub fn denom(&self) -> i64 {
            self.1
        }

        pub fn inner(&self) -> (i64, i64) {
            (self.0, self.1)
        }

        pub fn normalize(self) -> Self {
            let Self(a, b) = self;
            let d = gcd(a.abs() as u64, b.abs() as u64) as i64;
            Self(a / d, b / d)
        }

        pub fn zero() -> Self {
            Self(0, 1)
        }

        pub fn one() -> Self {
            Self(1, 1)
        }

        pub fn abs(self) -> Self {
            Self(self.0.abs(), self.1)
        }
    }

    impl Add for Frac64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 + rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Sub for Frac64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 - rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Mul for Frac64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.0, self.1 * rhs.1)
        }
    }

    impl Div for Frac64 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let s = rhs.0.signum();
            Self::new(self.0 * rhs.1 * s, self.1 * rhs.0 * s)
        }
    }

    impl From<i64> for Frac64 {
        fn from(a: i64) -> Self {
            Self::new(a, 1)
        }
    }

    impl PartialEq for Frac64 {
        fn eq(&self, other: &Self) -> bool {
            self.0 * other.1 == other.0 * self.1
        }
    }

    impl Eq for Frac64 {}

    impl PartialOrd for Frac64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some((self.0 * other.1).cmp(&(other.0 * self.1)))
        }
    }

    impl Ord for Frac64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Matrix<T> {
    n_rows: usize,
    n_cols: usize,
    data: Vec<T>,
}

// gaussian elimination
fn rref(mat: &Matrix<Frac64>) -> (usize, Frac64, Matrix<Frac64>) {
    let Matrix {
        n_rows,
        n_cols,
        ref data,
    } = *mat;

    let mut jagged: Vec<Vec<_>> = data.chunks_exact(n_cols).map(|t| t.to_vec()).collect();
    let mut det = Frac64::one();

    let mut rank = 0;
    for c in 0..n_cols {
        // let Some(pivot) = (rank..n_rows).find(|&j| jagged[j][c] != Frac64::zero()) else {
        //     continue;
        // };
        let Some(pivot) = (rank..n_rows)
            .filter(|&j| jagged[j][c] != Frac64::zero())
            .max_by_key(|&j| jagged[j][c].abs())
        else {
            continue;
        };
        jagged.swap(rank, pivot);

        det = det * jagged[rank][c];
        let inv_x = Frac64::one() / jagged[rank][c];
        for j in 0..n_cols {
            jagged[rank][j] = jagged[rank][j] * inv_x;
        }

        for i in 0..n_rows {
            if i == rank {
                continue;
            }
            let coeff = jagged[i][c];
            for j in 0..n_cols {
                jagged[i][j] = jagged[i][j] - jagged[rank][j] * coeff;
            }
        }
        rank += 1;
    }

    let mat = Matrix {
        n_rows,
        n_cols,
        data: jagged.into_iter().flatten().collect(),
    };
    (rank, det, mat)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = n + 1;
    let mat: Vec<_> = (0..n * m)
        .map(|_| {
            let x: i64 = input.value();
            Frac64::from(x)
        })
        .collect();
    let mat = Matrix {
        n_rows: n,
        n_cols: m,
        data: mat,
    };
    let (_, _, mat) = rref(&mat);
    for i in 0..n {
        let (p, q) = mat.data[i * m + m - 1].inner();
        assert!(q == 1);
        write!(output, "{} ", p).unwrap();
    }
}
