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

const MAX_SUM: i64 = 10_000_000_000_000;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut max_hull = cht::LiChaoTree::new(-MAX_SUM..=MAX_SUM);
    max_hull.insert(cht::Line::new(0, 0));

    let mut ans = 0;
    let mut sum = 0;
    let mut sum2 = 0;
    for i in 1..=n {
        let x: i64 = input.value();
        sum += x;

        ans = ans.max(sum * i as i64 - sum2 + max_hull.eval(sum));
        max_hull.insert(cht::Line::new(-(i as i64), sum2));

        sum2 += sum;
    }
    writeln!(output, "{}", ans).unwrap();
}
