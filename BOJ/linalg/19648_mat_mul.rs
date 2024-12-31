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

fn floyd_warshall(dist: &mut Vec<Vec<u64>>) {
    let n = dist.len();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = dist[i][j].min(dist[i][k] + dist[k][j]);
            }
        }
    }
}

const P: u64 = 1_000_000_007;

fn mul_mat(a: &Vec<Vec<u64>>, b: &Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    let n = a.len();
    let mut res = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % P;
            }
        }
    }
    res
}

fn apply_mat(b: &Vec<u64>, c: &Vec<Vec<u64>>) -> Vec<u64> {
    let n = c.len();
    let mut res = vec![0; n];
    for i in 0..n {
        for j in 0..n {
            res[i] = (res[i] + b[i] * c[i][j]) % P;
        }
    }
    res
}

fn pow_mat(base: &Vec<Vec<u64>>, mut exp: u64) -> Vec<Vec<u64>> {
    let n = base.len();
    let mut base = base.clone();
    let mut res = vec![vec![0; n]; n];
    for i in 0..n {
        res[i][i] = 1;
    }
    while exp > 0 {
        if exp & 1 == 1 {
            res = mul_mat(&res, &base);
        }
        base = mul_mat(&base, &base);
        exp >>= 1;
    }
    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = 14;

    let edges = vec![
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 4),
        (3, 0),
        (3, 1),
        (3, 4),
        (5, 3),
        (6, 3),
        (4, 6),
        (4, 7),
        (6, 7),
        (8, 5),
        (6, 8),
        (6, 9),
        (6, 10),
        (7, 10),
        (10, 9),
        (9, 8),
        (11, 8),
        (12, 8),
        (9, 12),
        (10, 13),
        (13, 12),
        (12, 11),
    ];

    let inf: u64 = 1 << 30;
    let mut dist = vec![vec![inf; n]; n];
    for &(u, v) in &edges {
        dist[u][v] = 1;
    }

    let a0 = 3;
    let b0 = 9;

    let mut min_dist = dist.clone();
    floyd_warshall(&mut min_dist);
    let invalid_state = |a: usize, b: usize| min_dist[a][b] < 3 || min_dist[b][a] < 3 || a == b;

    let m = n * n;
    let mut initial_state = vec![0; m];
    initial_state[a0 * n + b0] = 1;

    let mut trans = vec![vec![0; m]; m];
    for s in 0..m {
        let (u1, u2) = (s / n, s % n);
        if invalid_state(u1, u2) {
            continue;
        }
        for t in 0..m {
            let (v1, v2) = (t / n, t % n);
            if invalid_state(v1, v2) {
                continue;
            }
            if dist[u1][v1] == 1 && dist[u2][v2] == 1 {
                trans[s][t] = 1;
            }
        }
    }

    let k: usize = input.value();
    let ans = apply_mat(&initial_state, &pow_mat(&trans, k as u64))
        .iter()
        .sum::<u64>()
        % P;
    writeln!(output, "{}", ans).unwrap();
}
