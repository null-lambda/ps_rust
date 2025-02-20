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
