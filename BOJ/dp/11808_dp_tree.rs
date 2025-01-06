use std::io::Write;

mod simple_io {
    use std::string::*;

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

const NEG_INF: i64 = -(1 << 56);

#[derive(Default)]
struct NodeData {
    score: Vec<i64>,
}

impl NodeData {
    fn new() -> Self {
        Self { score: vec![0, 0] }
    }

    fn root() -> Self {
        Self {
            score: vec![0, NEG_INF],
        }
    }

    fn size(&self) -> usize {
        self.score.len() - 1
    }

    fn pull_from(&mut self, other: Self, weight: i64, k: usize) {
        let mut conv = vec![NEG_INF; k.min(self.size() + other.size()) + 1];
        for i in 0..=self.size() {
            for j in 0..=other.size() {
                if i + j > k {
                    continue;
                }
                conv[i + j] = conv[i + j]
                    .max(self.score[i] + other.score[j] + 2 * weight * j.min(k + 1 - j) as i64);
            }
        }
        self.score = conv;
    }

    fn collapse(&self, k: usize) -> i64 {
        self.score[k]
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for i_tc in 1..=input.value() {
        let n: usize = input.value();
        let k: usize = input.value();
        let mut degree = vec![1; n];
        let mut parent = vec![(0, 0); n];
        for u in 1..n {
            let p = input.value::<u32>() - 1;
            let w = input.value::<i64>();
            parent[u] = (p, w);
            degree[p as usize] += 1;
        }
        degree[0] += 2;

        let mut dp: Vec<_> = (0..n).map(|_| NodeData::new()).collect();
        dp[0] = NodeData::root();

        for mut u in 0..n {
            while degree[u] == 1 {
                let (p, w) = parent[u];
                degree[u] -= 1;
                degree[p as usize] -= 1;

                let dp_u = std::mem::take(&mut dp[u]);
                dp[p as usize].pull_from(dp_u, w, k);

                u = p as usize;
            }
        }
        let ans = dp[0].collapse(k);
        writeln!(output, "Case {i_tc}: {}", ans).unwrap();
    }
}
