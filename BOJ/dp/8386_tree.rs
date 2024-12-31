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

const INF: u32 = u32::MAX / 4;
#[derive(Clone, Debug)]
struct MinCost {
    exclusive: Vec<u32>,
    inclusive: Vec<u32>,
}

impl MinCost {
    fn leaf() -> Self {
        Self {
            exclusive: vec![INF, INF],
            inclusive: vec![INF, 0],
        }
    }

    fn move_up(&mut self) {
        for i in 0..self.exclusive.len() {
            self.exclusive[i] = self.exclusive[i].min(self.inclusive[i] + 1);
        }
        for i in self.exclusive.len()..self.inclusive.len() {
            self.exclusive.push(self.inclusive[i] + 1);
        }
        self.inclusive.insert(1, 1);
    }

    fn pull_from(&mut self, child: Self) {
        let mut conv = vec![INF; self.inclusive.len() + child.inclusive.len() - 1];
        for i in 0..self.inclusive.len() {
            conv[i] = conv[i].min(self.inclusive[i] + 1);
        }
        for i in 0..self.inclusive.len() {
            for j in 0..child.inclusive.len() {
                conv[i + j] = conv[i + j].min(self.inclusive[i] + child.inclusive[j])
            }
        }
        self.inclusive = conv;

        self.exclusive.resize(
            (self.exclusive.len())
                .max(self.inclusive.len())
                .max(child.exclusive.len()),
            INF,
        );
        for i in 0..child.inclusive.len() {
            self.exclusive[i] = self.exclusive[i].min(child.inclusive[i] + 1);
        }
        for i in 0..child.exclusive.len() {
            self.exclusive[i] = self.exclusive[i].min(child.exclusive[i]);
        }
    }
}

fn postorder(neighbors: &[Vec<usize>], u: usize, p: usize, visitor: &mut impl FnMut(usize, usize)) {
    for &v in &neighbors[u] {
        if v == p {
            continue;
        }
        postorder(neighbors, v, u, visitor);
    }
    visitor(u, p);
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let a = input.value::<usize>() - 1;
        let b = input.value::<usize>() - 1;
        neighbors[a].push(b);
        neighbors[b].push(a);
    }

    let mut dp: Vec<Option<MinCost>> = vec![None; n];
    postorder(&neighbors, 0, 0, &mut |u, p| {
        let vs: Vec<usize> = neighbors[u].iter().copied().filter(|&v| v != p).collect();
        if vs.is_empty() {
            dp[u] = Some(MinCost::leaf());
        } else {
            let mut acc = dp[vs[0]].clone().unwrap();
            acc.move_up();
            for &v in &vs[1..] {
                acc.pull_from(dp[v].clone().unwrap());
            }
            dp[u] = Some(acc);
        };
    });

    let dp = dp[0].as_ref().unwrap();
    for _ in 0..input.value() {
        let k = input.value::<usize>();
        let mut ans = (dp.exclusive.get(k).copied().unwrap_or(INF))
            .min(dp.inclusive.get(k).copied().unwrap_or(INF)) as i32;
        if ans >= INF as i32 {
            ans = -1;
        }
        writeln!(output, "{}", ans).unwrap();
    }
}
