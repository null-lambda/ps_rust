use std::{cell::OnceCell, io::Write, vec};

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

thread_local! {
    static PREFIX_SUM: OnceCell<Vec<i64>> = OnceCell::new();
}

struct Func {
    i: u32,
    delta: i64,
    n_components: u32,
}

impl cht::LineOrd for Func {
    type X = u32;
    type Y = (i64, u32);
    const BOTTOM: Self = Self {
        i: 0,
        delta: 1 << 60,
        n_components: 0,
    };

    fn bisect(lhs: &Self::X, rhs: &Self::X) -> (Self::X, Self::X) {
        let mid = lhs + rhs >> 1;
        (mid, mid + 1)
    }

    fn eval(&self, &i: &Self::X) -> Self::Y {
        PREFIX_SUM.with(|prefix| {
            let prefix = unsafe { prefix.get().unwrap_unchecked() };
            let dp_next = self.delta
                + (i as i64 - self.i as i64) * (prefix[i as usize] - prefix[self.i as usize]);
            (-dp_next, self.n_components + 1)
        })
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut k: usize = input.value();
    k = k.min(n);

    PREFIX_SUM.with(|prefix| {
        let mut xs = vec![0; n + 1];
        for i in 1..=n {
            let x: i64 = input.value();
            xs[i] = xs[i - 1] + x;
        }
        prefix.set(xs).unwrap();
    });

    let unconstrained = |slope: i64| {
        let mut neg_hull = cht::LiChaoTree::new(0..=n as u32);
        neg_hull.insert(Func {
            i: 0,
            delta: -slope,
            n_components: 0,
        });
        let mut dp_neg = (0, 0);
        for i in 1..=n {
            dp_neg = neg_hull.eval(&(i as u32));
            neg_hull.insert(Func {
                i: i as u32,
                delta: -dp_neg.0 - slope,
                n_components: dp_neg.1,
            });
        }
        let dp = -dp_neg.0;
        let k = dp_neg.1;
        (dp + k as i64 * slope, k)
    };

    let mut left = -1.3e14 as i64;
    let mut right = 1;
    while left < right {
        let mid = left + right >> 1;
        if unconstrained(mid).1 < k as u32 {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    let opt_slope = left;

    let (dp_base, k_upper) = unconstrained(opt_slope);
    let ans = dp_base + opt_slope * (k as i64 - k_upper as i64);
    writeln!(output, "{}", ans).unwrap();
}
