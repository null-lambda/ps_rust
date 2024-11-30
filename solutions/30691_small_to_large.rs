use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

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

#[derive(Debug)]
struct SubTreeWeights {
    base: i64, // lazy
    deltas: BinaryHeap<(i64, u32)>,
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: i64 = input.value();

    let mut neighbors = vec![vec![]; n];
    for _ in 1..n {
        let u: usize = input.value();
        let v: usize = input.value();
        let c: i64 = input.value();
        neighbors[u - 1].push((v - 1, c));
        neighbors[v - 1].push((u - 1, c));
    }

    let root = 0;
    fn dfs_dp(
        neighbors: &[Vec<(usize, i64)>],
        k: i64,
        u: usize,
        p: usize,
        ans: &mut u32,
    ) -> SubTreeWeights {
        if neighbors[u].iter().filter(|&&(v, _)| v != p).count() == 0 {
            return SubTreeWeights {
                base: 0,
                deltas: [(0, u as u32)].into(),
            };
        }

        let mut sub_results = vec![];
        for &(v, c) in &neighbors[u] {
            if v == p {
                continue;
            }
            let mut sub_res = dfs_dp(neighbors, k, v, u, ans);
            sub_res.base += c;
            sub_results.push(sub_res);
        }
        sub_results.select_nth_unstable_by_key(0, |t| Reverse(t.deltas.len()));

        let mut sub_results = sub_results.into_iter();
        let mut large = sub_results.next().unwrap();
        while let Some(x) = large.deltas.peek() {
            if x.0 + large.base <= k {
                break;
            }
            large.deltas.pop();
        }
        large.deltas.extend(sub_results.flat_map(|small| {
            small.deltas.into_iter().filter_map(move |(w, i)| {
                (w + small.base <= k).then(|| (w + small.base - large.base, i))
            })
        }));
        large.deltas.push((-large.base, u as u32));
        *ans = (*ans).max(large.deltas.len() as u32);

        large
    }

    let mut ans = 1;
    dfs_dp(&neighbors, k, root, root, &mut ans);
    writeln!(output, "{}", ans).unwrap();
}
