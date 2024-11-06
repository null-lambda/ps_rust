use std::{io::Write, iter};

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct Matrix<T> {
    n_rows: usize,
    n_cols: usize,
    data: Vec<T>,
}

const P: u64 = 998244353;

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: u64) -> u64 {
    pow(n, P - 2)
}

fn char_poly(a: &Matrix<u64>) -> Vec<u64> {
    let n = a.n_rows;
    assert_eq!(n, a.n_cols);

    let mut a_rows: Vec<Vec<_>> = (0..n).map(|i| a.data[i * n..][..n].to_vec()).collect();
    let mut col_idx = (0..n).collect::<Vec<_>>();
    for j in 0..n - 2 {
        for i in j + 1..n {
            if a_rows[i][col_idx[j]] != 0 {
                a_rows.swap(i, j + 1);
                col_idx.swap(i, j + 1);
                break;
            }
        }

        if a_rows[j + 1][col_idx[j]] != 0 {
            for i in j + 2..n {
                let x = a_rows[i][col_idx[j]] * mod_inv(a_rows[j + 1][col_idx[j]]) % P;
                let x_neg = (P - x) % P;
                for k in 0..n {
                    a_rows[i][col_idx[k]] =
                        (a_rows[i][col_idx[k]] + a_rows[j + 1][col_idx[k]] * x_neg) % P;
                }
                for k in 0..n {
                    a_rows[k][col_idx[j + 1]] =
                        (a_rows[k][col_idx[j + 1]] + a_rows[k][col_idx[i]] * x) % P;
                }
            }
        }
    }

    let a_prev = a_rows;
    let mut a = vec![0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = a_prev[i][col_idx[j]];
        }
    }

    let mut dp = vec![vec![]; n + 1];
    dp[0] = vec![1];
    for i in 1..=n {
        dp[i] = iter::once(0).chain(dp[i - 1].iter().copied()).collect();
        for j in 0..i {
            dp[i][j] = (dp[i][j] + P - a[(i - 1) * n + i - 1] * dp[i - 1][j] % P) % P;
        }
        let mut prod = 1;
        for j in (0..i - 1).rev() {
            prod = prod * a[(j + 1) * n + j] % P;
            for k in 0..=j {
                dp[i][k] = (dp[i][k] + P - a[j * n + i - 1] * prod % P * dp[j][k] % P) % P;
            }
        }
    }

    dp.pop().unwrap()
}

// gaussian elimination
fn rref(mat: &Matrix<u64>) -> (usize, u64, Matrix<u64>) {
    let Matrix {
        n_rows,
        n_cols,
        ref data,
    } = *mat;

    let mut jagged: Vec<Vec<_>> = data.chunks_exact(n_cols).map(|t| t.to_vec()).collect();
    let mut det = 1;

    let mut rank = 0;
    for c in 0..n_cols {
        let Some(pivot) = (rank..n_rows).find(|&j| jagged[j][c] != 0) else {
            continue;
        };
        // let Some(pivot) = (rank..n_rows)
        //     .filter(|&j| jagged[j][c] != Frac64::zero())
        //     .max_by_key(|&j| jagged[j][c].abs())
        // else {
        //     continue;
        // };
        if pivot != rank {
            jagged.swap(rank, pivot);
            det = (P - det) % P;
        }

        det = det * jagged[rank][c] % P;
        let inv_x = mod_inv(jagged[rank][c]);
        for j in 0..n_cols {
            jagged[rank][j] = jagged[rank][j] * inv_x % P;
        }

        for i in 0..n_rows {
            if i == rank {
                continue;
            }
            let coeff = jagged[i][c];
            for j in 0..n_cols {
                jagged[i][j] = (jagged[i][j] + P - jagged[rank][j] * coeff % P) % P;
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
    let q: usize = input.value();
    let mat: Vec<u64> = (0..n * n).map(|_| input.value()).collect();
    let mat = Matrix {
        n_rows: n,
        n_cols: n,
        data: mat,
    };

    let get_det = |x: u64| {
        let mut sub = mat.clone();
        for i in 0..n {
            sub.data[i * n + i] = (sub.data[i * n + i] + P - x) % P;
        }
        let (_, det, _) = rref(&sub);
        det
    };

    let coeffs = char_poly(&mat);
    // let mut system = vec![0; (n + 1) * (n + 2)];
    // for i in 0..n + 1 {
    //     println!("i = {}", i);
    //     let x = i as u64;
    //     let mut pow_x = 1;
    //     for j in 0..n + 1 {
    //         system[i * (n + 2) + j] = pow_x;
    //         pow_x = pow_x * x % P;
    //     }
    //     system[i * (n + 2) + n + 1] = get_det(x);
    // }
    // let system = Matrix {
    //     n_rows: n + 1,
    //     n_cols: n + 2,
    //     data: system,
    // };
    // let (_, _, coeffs) = rref(&system);
    // let coeffs: Vec<u64> = (0..n + 1)
    //     .map(|i| coeffs.data[i * (n + 2) + n + 1])
    //     .collect();

    let eval_char_poly = |x: u64| {
        let sign = if n % 2 == 0 { 1 } else { P - 1 };
        let mut res = 0;
        let mut pow_x = 1;
        for c in &coeffs {
            res = (res + c * pow_x) % P;
            pow_x = pow_x * x % P;
        }
        res * sign % P
    };

    // for x in 0..1000 {
    //     debug_assert_eq!(get_det(x), eval_char_poly(x));
    // }

    for _ in 0..q {
        let x: u64 = input.value::<u64>() % P;
        writeln!(output, "{}", eval_char_poly(x)).unwrap();
    }
}
