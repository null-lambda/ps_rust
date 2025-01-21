use std::{collections::BTreeMap, io::Write};

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

#[derive(Clone, Default)]
struct NodeAgg {
    day: u32,
    score: u64,
    lif: BTreeMap<u32, u64>, // Largest monotonically increasing subforest
}

impl NodeAgg {
    fn vacant() -> Self {
        Self {
            day: 0,
            score: 0,
            lif: Default::default(),
        }
    }

    fn singleton(day: u32, score: u64) -> Self {
        Self {
            day,
            score,
            lif: Default::default(),
        }
    }

    fn pull_from(&mut self, mut other: Self) {
        if self.lif.len() < other.lif.len() {
            std::mem::swap(&mut self.lif, &mut other.lif);
        }
        for (d, s) in other.lif {
            self.lif.entry(d).and_modify(|x| *x += s).or_insert(s);
        }
    }

    fn finalize(&mut self) {
        self.lif
            .entry(self.day)
            .and_modify(|x| *x += self.score)
            .or_insert(self.score);

        let mut to_remove = vec![];
        for (d, s) in self.lif.range_mut(self.day + 1..) {
            if self.score == 0 {
                break;
            }
            let delta = self.score.min(*s);
            self.score -= delta;
            *s -= delta;

            if *s == 0 {
                to_remove.push(*d);
            }
        }

        for d in to_remove {
            self.lif.remove(&d);
        }
    }

    fn collapse(mut self) -> u64 {
        let mut max_prefix = 0;
        for (_d, s) in self.lif {
            self.score += s;
            max_prefix = max_prefix.max(self.score);
        }

        max_prefix
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let _k: usize = input.value();
    let mut degree = vec![1u32; n];
    let mut parent = vec![0; n];
    for u in 1..n {
        let p = input.value::<u32>() - 1;
        parent[u] = p;
        degree[p as usize] += 1;
    }

    let mut dp = vec![NodeAgg::vacant(); n];
    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let d = input.value();
        let w = input.value();
        dp[u] = NodeAgg::singleton(d, w);
    }

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = parent[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            let mut dp_u = std::mem::take(&mut dp[u as usize]);
            dp_u.finalize();
            dp[p as usize].pull_from(dp_u);

            u = p;
        }
    }
    dp[0].finalize();
    let ans = std::mem::take(&mut dp[0]).collapse();
    writeln!(output, "{}", ans).unwrap();
}
