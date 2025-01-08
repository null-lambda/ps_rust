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

pub mod cht {
    use core::{num::NonZeroU32, ops::RangeInclusive};

    // max Li-Chao tree of lines
    // TODO: add segment insertion
    type V = i64;
    type K = i64;
    const NEG_INF: V = i64::MIN;

    #[derive(Clone)]
    pub struct Line {
        slope: V,
        intercept: V,
    }

    impl Line {
        pub fn new(slope: V, intercept: V) -> Self {
            Self { slope, intercept }
        }

        fn eval(&self, x: V) -> K {
            self.slope * x + self.intercept
        }

        fn bottom() -> Self {
            Self {
                slope: 0,
                intercept: NEG_INF,
            }
        }
    }

    #[derive(Clone)]
    struct NodeRef(NonZeroU32);

    #[derive(Clone)]
    struct Node {
        children: [Option<NodeRef>; 2],
        line: Line,
    }

    impl Node {
        fn new() -> Self {
            Self {
                children: [None, None],
                line: Line::bottom(),
            }
        }
    }

    #[derive(Clone)]
    pub struct LiChaoTree {
        pool: Vec<Node>,
        interval: RangeInclusive<V>,
    }

    impl LiChaoTree {
        pub fn new(interval: RangeInclusive<V>) -> Self {
            Self {
                pool: vec![Node::new()],
                interval,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let index = self.pool.len();
            self.pool.push(node);
            NodeRef(NonZeroU32::new(index as u32).unwrap())
        }

        // pub fn insert_segment(&mut self, interval: (V, V), mut line: Line) {
        //     unimplemented!()
        // }

        pub fn insert(&mut self, mut line: Line) {
            let mut u = 0;
            let (mut x_left, mut x_right) = self.interval.clone().into_inner();
            loop {
                let x_mid = (x_left + x_right) / 2;
                let top = &mut self.pool[u].line;
                if top.eval(x_mid) < line.eval(x_mid) {
                    std::mem::swap(top, &mut line);
                }
                u = if top.eval(x_left) < line.eval(x_left) {
                    x_right = x_mid;
                    match self.pool[u].children[0] {
                        Some(ref c) => c.0.get() as usize,
                        None => {
                            let c = self.alloc(Node::new());
                            self.pool[u].children[0] = Some(c.clone());
                            c.0.get() as usize
                        }
                    }
                } else if top.eval(x_right) < line.eval(x_right) {
                    x_left = x_mid + 1;
                    match self.pool[u].children[1] {
                        Some(ref c) => c.0.get() as usize,
                        None => {
                            let c = self.alloc(Node::new());
                            self.pool[u].children[1] = Some(c.clone());
                            c.0.get() as usize
                        }
                    }
                } else {
                    return;
                };
            }
        }

        pub fn eval(&self, x: V) -> K {
            debug_assert!(self.interval.contains(&x));
            let mut u = 0;
            let mut result = self.pool[u].line.eval(x);
            let (mut x_left, mut x_right) = self.interval.clone().into_inner();
            loop {
                let x_mid = (x_left + x_right) / 2;
                let branch = if x <= x_mid {
                    x_right = x_mid;
                    0
                } else {
                    x_left = x_mid + 1;
                    1
                };
                if let Some(c) = &self.pool[u].children[branch] {
                    u = c.0.get() as usize;
                } else {
                    return result;
                }
                result = result.max(self.pool[u].line.eval(x));
            }
        }
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let t_bound: usize = input.value();
    let mut ps: Vec<_> = (0..n)
        .map(|_| {
            let t: usize = input.value();
            let p: i64 = input.value();
            let f: i64 = input.value();
            (t, p, f)
        })
        .collect();
    ps.sort_unstable_by_key(|&(_, _, f)| f);

    // dp[ti][i] <- pi
    // dp[t+ti][i] <- max { dp[t][j] + pi - (fi-fj)^2 : i < j }
    //              = max { dp[t][j] - fj^2 + 2 fj fi } + pi - fi^2
    //
    // Define g[t][i] = dp[t][i] - fi^2. Then
    // g[ti][i] <- pi - fi^2
    // g[t+ti][i] = max { g[t][j] + 2 fj fi } + pi - 2 fi^2
    //
    // For each t, assign a dynamic upper convex hull of the family of lines
    // F_t(i) = { (x |-> g[t][j] + 2 fj x) : j < i } and increment i
    let f_bound = 10_000;
    let mut hulls = vec![cht::LiChaoTree::new(1..=f_bound); t_bound + 1];

    let mut ans = 0;
    for i in 0..n {
        let (ti, pi, fi) = ps[i];
        if ti > t_bound {
            continue;
        }

        for t in (0..=t_bound - ti).rev() {
            let mut g = hulls[t].eval(fi);
            if g >= 0 {
                g += pi - 2 * fi * fi;
                hulls[t + ti].insert(cht::Line::new(2 * fi, g));
                ans = ans.max(g + fi * fi);
            }
        }

        hulls[ti].insert(cht::Line::new(2 * fi, pi - fi * fi));
        ans = ans.max(pi);
    }

    writeln!(output, "{}", ans).unwrap();
}
