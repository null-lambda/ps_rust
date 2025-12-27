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

fn xor_traversal(
    mut degree: Vec<u32>,
    mut xor_neighbors: Vec<u32>,
    root: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = degree.len();
    degree[root as usize] += 2;

    let mut toposort = vec![];

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            xor_neighbors[p as usize] ^= u;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            toposort.push(u);

            u = p;
        }
    }
    toposort.push(root);

    let mut parent = xor_neighbors;
    parent[root as usize] = root;
    (toposort, parent)
}

fn reconstruct_concave_rec(
    l: i64,
    r: i64,
    yl: i64,
    yr: i64,
    f: &impl Fn(i64) -> i64,
    yield_f: &mut impl FnMut(i64, i64),
) {
    if r - l <= 1 {
        return;
    }

    let m = l + r >> 1;
    let ym = f(m);

    let dx0 = m - l;
    let dy0 = ym - yl;
    let dx1 = r - l;
    let dy1 = yr - yl;
    if dx0 * dy1 == dx1 * dy0 {
        for i in l + 1..r {
            yield_f(i, yl + dy0 * (i - l) / dx0);
        }
        return;
    }

    yield_f(m, ym);
    reconstruct_concave_rec(l, m, yl, ym, f, yield_f);
    reconstruct_concave_rec(m, r, ym, yr, f, yield_f);
}

fn reconstruct_concave(l: i64, r: i64, f: impl Fn(i64) -> i64, mut yield_f: impl FnMut(i64, i64)) {
    if r - l <= 1 {
        for x in l..=r {
            yield_f(x, f(x));
        }
        return;
    }

    let yl = f(l);
    let yr = f(r);
    yield_f(l, yl);
    yield_f(r, yr);
    reconstruct_concave_rec(l, r, yl, yr, &f, &mut yield_f)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let s = input.token().as_bytes();
    let mut xor = vec![0u32; n];
    let mut deg = vec![0u32; n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        xor[u as usize] ^= v;
        xor[v as usize] ^= u;
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let (mut toposort, parent) = xor_traversal(deg, xor, 0);

    toposort.reverse();
    let mut inv = vec![!0; n];
    for u in 0..n {
        inv[toposort[u] as usize] = u as u32;
    }

    let parent: Vec<_> = (0..n)
        .map(|u| inv[parent[toposort[u] as usize] as usize])
        .collect();
    let s: Vec<_> = (0..n).map(|u| s[toposort[u] as usize]).collect();

    let work = std::cell::Cell::new(0);
    let solve = |k: i64| {
        work.set(work.get() + 1);

        let inf = 1 << 58;
        let mut dp = vec![[0; 2]; n];
        for u in (0..n).rev() {
            if s[u as usize] == b'1' {
                dp[u as usize][0] = inf;
            }
            dp[u as usize][1] += 1 + k;

            if u == 0 {
                break;
            }
            let p = parent[u as usize];

            dp[p as usize][0] += dp[u as usize][0].min(dp[u as usize][1]);
            dp[p as usize][1] += dp[u as usize][0].min(dp[u as usize][1] - k);
        }

        dp[0][0].min(dp[0][1])
    };

    let mut ans = vec![0; n];
    reconstruct_concave(1, n as i64, solve, |k, a| ans[k as usize - 1] = a);
    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }

    eprintln!("work {:?}", work.get());
}
