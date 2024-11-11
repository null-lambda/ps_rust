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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod segtree {
    pub trait Monoid {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    pub trait Group: Monoid {
        fn sub(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    pub mod persistent {
        use super::*;

        use std::ops::Range;

        const UNSET: u32 = std::u32::MAX;
        pub type NodeRef = u32;
        type Link = NodeRef;

        struct Node<M: Monoid> {
            value: M::X,
            children: [Link; 2],
        }

        pub struct NodePool<M: Monoid> {
            n: usize,
            nodes: Vec<Node<M>>,
            pub monoid: M,
        }

        impl<M: Monoid> NodePool<M> {
            fn add_node(&mut self, value: M::X) -> NodeRef {
                let node = Node {
                    value,
                    children: [UNSET; 2],
                };
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                idx
            }

            pub fn with_size(n: usize, monoid: M) -> (Self, NodeRef) {
                let mut this = Self {
                    n,
                    nodes: vec![],
                    monoid,
                };
                this.with_size_rec(0..n);
                (this, 0)
            }

            fn with_size_rec(&mut self, range: Range<usize>) -> Link {
                debug_assert!(range.start <= range.end);
                let Range { start, end } = range;
                if end - start == 0 {
                    return UNSET;
                }

                let mid = (start + end) / 2;
                let u = self.add_node(self.monoid.id());
                if end - start > 1 {
                    self.nodes[u as usize].children =
                        [self.with_size_rec(start..mid), self.with_size_rec(mid..end)];
                }

                u
            }

            pub fn set(&mut self, root: NodeRef, idx: usize, value: M::X) -> NodeRef {
                let mut path = vec![];
                let mut node = root;
                let (mut start, mut end) = (0, self.n);
                loop {
                    if end - start == 1 {
                        break;
                    }

                    let mid = (start + end) / 2;
                    if idx < mid {
                        path.push((node, 0u8));
                        end = mid;
                        node = self.nodes[node as usize].children[0];
                    } else {
                        path.push((node, 1u8));
                        start = mid;
                        node = self.nodes[node as usize].children[1];
                    }
                }

                let mut root = self.add_node(value);
                for (node, branch) in path.into_iter().rev() {
                    let (left, right) = match branch {
                        0 => (root, self.nodes[node as usize].children[1]),
                        1 => (self.nodes[node as usize].children[0], root),
                        _ => unreachable!(),
                    };
                    root = self.add_node(self.monoid.combine(
                        &self.nodes[left as usize].value,
                        &self.nodes[right as usize].value,
                    ));
                    self.nodes[root as usize].children = [left, right];
                }
                root
            }

            pub fn query_range(&self, node: NodeRef, range: Range<usize>) -> M::X {
                self.query_range_rec(node, 0..self.n, range)
            }

            pub fn query_range_rec(
                &self,
                node: NodeRef,
                node_range: Range<usize>,
                query_range: Range<usize>,
            ) -> M::X {
                let Range { start, end } = node_range;
                let Range {
                    start: query_start,
                    end: query_end,
                } = query_range;
                if query_end <= start || end <= query_start {
                    return self.monoid.id();
                }
                if query_start <= start && end <= query_end {
                    return self.nodes[node as usize].value.clone();
                }
                let mid = (start + end) / 2;
                let c = self.nodes[node as usize].children;
                self.monoid.combine(
                    &self.query_range_rec(c[0], start..mid, query_range.clone()),
                    &self.query_range_rec(c[1], mid..end, query_range),
                )
            }
        }

        impl<M: Group> NodePool<M>
        where
            M::X: Ord,
        {
            pub fn nth(&self, node_left: NodeRef, node_right: NodeRef, bound: M::X) -> usize {
                self.nth_rec(node_left, node_right, bound, 0..self.n)
            }

            fn nth_rec(
                &self,
                node_left: NodeRef,
                node_right: NodeRef,
                bound: M::X,
                range: Range<usize>,
            ) -> usize {
                let Range { start, end } = range;
                if end - start == 1 {
                    return start;
                }
                let mid = (start + end) / 2;
                let acc_left = self.monoid.sub(
                    &self.nodes[self.nodes[node_right as usize].children[0] as usize].value,
                    &self.nodes[self.nodes[node_left as usize].children[0] as usize].value,
                );
                if bound < acc_left {
                    self.nth_rec(
                        self.nodes[node_left as usize].children[0],
                        self.nodes[node_right as usize].children[0],
                        bound,
                        start..mid,
                    )
                } else {
                    self.nth_rec(
                        self.nodes[node_left as usize].children[1],
                        self.nodes[node_right as usize].children[1],
                        self.monoid.sub(&bound, &acc_left),
                        mid..end,
                    )
                }
            }
        }
    }
}

use std::{collections::HashMap, hash::Hash};

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

use segtree::{persistent::*, Group, Monoid};
impl Monoid for u32 {
    type X = u32;
    fn id(&self) -> Self::X {
        0
    }
    fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X {
        a + b
    }
}

impl Group for u32 {
    fn sub(&self, a: &Self::X, b: &Self::X) -> Self::X {
        a - b
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let xs: Vec<i32> = (0..n).map(|_| input.value()).collect();
    let (xs_map, xs_map_inv) = compress_coord(xs.iter().cloned());
    let xs = xs.into_iter().map(|x| xs_map_inv[&x]).collect::<Vec<_>>();
    let x_max = xs_map.len() as u32;

    let (mut pool, base_root) = NodePool::with_size(x_max as usize, 0);
    let mut freq = vec![base_root];
    for i in 0..n {
        freq.push(pool.set(freq[i], xs[i] as usize, 1));
    }

    for _ in 0..q {
        let l: usize = input.value();
        let r: usize = input.value();
        let k: u32 = input.value();
        let ans = pool.nth(freq[l - 1], freq[r], k - 1);

        writeln!(output, "{}", xs_map[ans]).unwrap();
    }
}
