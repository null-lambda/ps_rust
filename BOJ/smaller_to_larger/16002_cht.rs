use std::{io::Write, mem::MaybeUninit};

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

    // Max Li-Chao tree of lines, or a line-like family.

    // A family of functions, that have similar ordering properties to lines.
    pub trait LineOrd {
        type X: Clone + Ord;
        type Y: Clone + Ord;
        const BOTTOM: Self;

        fn bisect(lhs: &Self::X, rhs: &Self::X) -> (Self::X, Self::X);
        fn eval(&self, x: &Self::X) -> Self::Y;
    }

    #[derive(Clone)]
    pub struct Line<V> {
        pub slope: V,
        pub intercept: V,
    }

    impl<V> Line<V> {
        pub fn new(slope: V, intercept: V) -> Self {
            Self { slope, intercept }
        }
    }

    impl LineOrd for Line<i64> {
        type X = i64;
        type Y = i64;

        const BOTTOM: Self = Self {
            slope: 0,
            intercept: std::i64::MIN,
        };

        fn bisect(lhs: &Self::X, rhs: &Self::X) -> (Self::X, Self::X) {
            let mid = lhs + rhs >> 1;
            (mid, mid + 1)
        }

        fn eval(&self, x: &Self::X) -> Self::Y {
            self.slope * x + self.intercept
        }
    }

    #[derive(Clone)]
    pub struct NodeRef(NonZeroU32);

    impl NodeRef {
        fn new(index: usize) -> Self {
            Self(unsafe { NonZeroU32::new(index as u32).unwrap_unchecked() })
        }

        fn get(&self) -> usize {
            self.0.get() as usize
        }
    }

    struct Node<L> {
        children: [Option<NodeRef>; 2],
        line: L,
    }

    impl<L: LineOrd> Node<L> {
        fn new() -> Self {
            Self {
                children: [None, None],
                line: L::BOTTOM,
            }
        }
    }

    pub struct LiChaoTree<L: LineOrd> {
        pool: Vec<Node<L>>,
        x_range: RangeInclusive<L::X>,
    }

    impl<L: LineOrd> LiChaoTree<L> {
        pub fn new(x_range: RangeInclusive<L::X>) -> Self {
            Self {
                pool: vec![Node::new()],
                x_range,
            }
        }

        fn alloc(&mut self, node: Node<L>) -> NodeRef {
            let index = self.pool.len();
            self.pool.push(node);
            NodeRef::new(index)
        }

        // pub fn insert_segment(&mut self, interval: (V, V), mut line: Line) {
        //     unimplemented!()
        // }

        pub fn insert(&mut self, mut line: L) {
            let mut u = 0;
            let (mut x_left, mut x_right) = self.x_range.clone().into_inner();
            loop {
                let (x_mid, x_mid_next) = L::bisect(&x_left, &x_right);
                let top = &mut self.pool[u].line;
                if top.eval(&x_mid) < line.eval(&x_mid) {
                    std::mem::swap(top, &mut line);
                }

                let branch = if top.eval(&x_left) < line.eval(&x_left) {
                    x_right = x_mid;
                    0
                } else if top.eval(&x_right) < line.eval(&x_right) {
                    x_left = x_mid_next;
                    1
                } else {
                    return;
                };

                if self.pool[u].children[branch].is_none() {
                    self.pool[u].children[branch] = Some(self.alloc(Node::new()));
                }
                u = unsafe {
                    self.pool[u].children[branch]
                        .as_ref()
                        .unwrap_unchecked()
                        .get()
                };
            }
        }

        pub fn eval(&self, x: &L::X) -> L::Y {
            debug_assert!(self.x_range.contains(&x));
            let mut u = 0;
            let mut result = self.pool[u].line.eval(x);
            let (mut x_left, mut x_right) = self.x_range.clone().into_inner();
            loop {
                let (x_mid, x_mid_next) = L::bisect(&x_left, &x_right);
                let branch = if x <= &x_mid {
                    x_right = x_mid;
                    0
                } else {
                    x_left = x_mid_next;
                    1
                };

                if let Some(c) = &self.pool[u].children[branch] {
                    u = c.get();
                } else {
                    return result;
                }
                result = result.max(self.pool[u].line.eval(x));
            }
        }
    }
}

struct NodeAgg {
    lines: Vec<cht::Line<i64>>,
    neg_hull: cht::LiChaoTree<cht::Line<i64>>,
    delta: i64,

    min_cost: i64,
}

impl NodeAgg {
    fn empty() -> Self {
        Self {
            lines: Vec::new(),
            neg_hull: cht::LiChaoTree::new(0..=1_000_001),
            delta: 0,
            min_cost: 0,
        }
    }

    fn finalize(&mut self, weight: i32) {
        self.delta += self.min_cost;
        if self.lines.is_empty() {
            let l = cht::Line::new(-weight as i64, 0);
            self.lines.push(l.clone());
            self.neg_hull.insert(l);
        }
    }

    fn pull_from(&mut self, weight: i32, mut child: Self) {
        let child_arm = child.delta - child.neg_hull.eval(&(weight as i64));
        self.min_cost += child_arm;
        child.delta -= child_arm;

        if self.lines.len() < child.lines.len() {
            std::mem::swap(&mut self.lines, &mut child.lines);
            std::mem::swap(&mut self.neg_hull, &mut child.neg_hull);
            std::mem::swap(&mut self.delta, &mut child.delta);
        }

        let delta = child.delta - self.delta;
        for mut l in child.lines {
            l.intercept -= delta;
            self.lines.push(l.clone());
            self.neg_hull.insert(l);
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut degree = vec![1u32; n];
    let mut parents = vec![0];
    for _ in 1..n {
        let p = input.value::<u32>() - 1;
        parents.push(p);
        degree[p as usize] += 1;
    }
    degree[0] += 2;
    let weights: Vec<i32> = (0..n).map(|_| input.value()).collect();

    let mut dp: Vec<_> = (0..n).map(|_| MaybeUninit::new(NodeAgg::empty())).collect();
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = parents[u as usize];
            degree[p as usize] -= 1;
            degree[u as usize] -= 1;

            unsafe {
                let mut dp_u =
                    std::mem::replace(&mut dp[u as usize], MaybeUninit::uninit()).assume_init();
                dp_u.finalize(weights[u as usize]);

                {
                    // print!("u = {}, min_cost: {} hull: ", u, dp_u.min_cost);
                    // for x in 0..=10 {
                    //     print!("{} ", dp_u.delta - dp_u.neg_hull.eval(&x));
                    // }
                    // println!();
                }

                dp[p as usize]
                    .assume_init_mut()
                    .pull_from(weights[p as usize], dp_u);
            }

            u = p;
        }
    }
    let dp_root = unsafe { dp[0].assume_init_mut() };
    dp_root.finalize(weights[0]);
    let ans = dp_root.min_cost;
    writeln!(output, "{}", ans).unwrap();

    // {
    //     let u = 0;
    //     let dp_u = dp_root;
    //     print!("u = {}, hull: ", u);
    //     for x in 0..=10 {
    //         print!("{} ", dp_u.delta - dp_u.neg_hull.eval(&x));
    //     }
    //     println!();
    // }
}
