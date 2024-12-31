use std::{cmp::Ordering, io::Write};

use segtree::persistent::*;

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

pub mod segtree {
    pub mod persistent {
        use std::{num::NonZeroU32, ops::Range};

        pub trait Monoid {
            type X: Clone;
            fn id(&self) -> Self::X;
            fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X;
        }

        #[derive(Clone, Copy)]
        pub struct NodeRef(NonZeroU32);

        struct Node<M: Monoid> {
            value: M::X,
            children: Option<[NodeRef; 2]>,
        }

        impl<M: Monoid> Clone for Node<M> {
            fn clone(&self) -> Self {
                Self {
                    value: self.value.clone(),
                    children: self.children,
                }
            }
        }

        pub struct SegTree<M: Monoid> {
            n: usize,
            nodes: Vec<Node<M>>,
            monoid: M,
        }

        impl<M: Monoid> SegTree<M> {
            fn alloc(&mut self, node: Node<M>) -> NodeRef {
                let idx = self.nodes.len();
                self.nodes.push(node);
                NodeRef(NonZeroU32::new(idx as u32).unwrap())
            }

            fn get_node(&self, idx: NodeRef) -> &Node<M> {
                &self.nodes[idx.0.get() as usize]
            }

            fn clone(&mut self, idx: NodeRef) -> NodeRef {
                let node = self.nodes[idx.0.get() as usize].clone();
                self.alloc(node)
            }

            fn get_node_mut(&mut self, idx: NodeRef) -> &mut Node<M> {
                &mut self.nodes[idx.0.get() as usize]
            }

            fn get_node_mut_multiple<'a, const N: usize>(
                &'a mut self,
                indices: [NodeRef; N],
            ) -> [&'a mut Node<M>; N] {
                let ptr = self.nodes.as_mut_ptr();
                unsafe { indices.map(|i| &mut *ptr.add(i.0.get() as usize)) }
            }

            pub fn with_size(n: usize, monoid: M) -> (Self, NodeRef) {
                debug_assert!(n > 0);
                let dummy = Node {
                    value: monoid.id(),
                    children: None,
                };
                let mut this = Self {
                    n,
                    nodes: vec![dummy],
                    monoid,
                };
                let root = this.with_size_rec(0..n);
                (this, root)
            }

            fn with_size_rec(&mut self, range: Range<usize>) -> NodeRef {
                debug_assert!(range.start < range.end);
                let Range { start, end } = range;
                let node = self.alloc(Node {
                    value: self.monoid.id(),
                    children: None,
                });
                if end - start > 1 {
                    let mid = (start + end) >> 1;
                    self.get_node_mut(node).children =
                        Some([self.with_size_rec(start..mid), self.with_size_rec(mid..end)]);
                }
                node
            }

            pub fn set(&mut self, root: NodeRef, idx: usize, value: M::X) -> NodeRef {
                debug_assert!(idx < self.n);

                let mut path = vec![];
                let mut node = root;
                let (mut start, mut end) = (0, self.n);
                loop {
                    if end - start == 1 {
                        break;
                    }

                    let mid = (start + end) >> 1;
                    if idx < mid {
                        path.push((node, 0u8));
                        end = mid;
                        node = unsafe { self.get_node(node).children.unwrap_unchecked()[0] };
                    } else {
                        path.push((node, 1u8));
                        start = mid;
                        node = unsafe { self.get_node(node).children.unwrap_unchecked()[1] };
                    }
                }

                let mut curr = self.alloc(Node {
                    value,
                    children: None,
                });
                while let Some((node, branch)) = path.pop() {
                    let [left, right] = unsafe {
                        match branch {
                            0 => [curr, self.get_node(node).children.unwrap_unchecked()[1]],
                            1 => [self.get_node(node).children.unwrap_unchecked()[0], curr],
                            _ => std::hint::unreachable_unchecked(),
                        }
                    };
                    curr = self.alloc(Node {
                        value: self
                            .monoid
                            .combine(&self.get_node(left).value, &self.get_node(right).value),
                        children: Some([left, right]),
                    });
                }

                curr
            }

            pub fn get(&self, root: NodeRef, idx: usize) -> M::X {
                debug_assert!(idx < self.n);
                let mut node = root;
                let (mut start, mut end) = (0, self.n);
                while end - start > 1 {
                    let mid = (start + end) >> 1;
                    if idx < mid {
                        node = unsafe { self.get_node(node).children.unwrap_unchecked()[0] };
                        end = mid;
                    } else {
                        node = unsafe { self.get_node(node).children.unwrap_unchecked()[1] };
                        start = mid;
                    }
                }
                self.get_node(node).value.clone()
            }

            pub fn query_range(&self, root: NodeRef, range: Range<usize>) -> M::X {
                self.query_range_rec(root, 0..self.n, range)
            }

            fn query_range_rec(
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
                    return self.get_node(node).value.clone();
                }
                let mid = (start + end) / 2;
                let c = unsafe { self.get_node(node).children.unwrap_unchecked() };
                self.monoid.combine(
                    &self.query_range_rec(c[0], start..mid, query_range.clone()),
                    &self.query_range_rec(c[1], mid..end, query_range),
                )
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

struct Additive;

impl Monoid for Additive {
    type X = u32;
    fn id(&self) -> Self::X {
        0
    }
    fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X {
        a + b
    }
}

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs: Vec<u32> = (0..n).map(|_| input.value()).collect();

    let (x_map, x_map_inv) = compress_coord(xs.iter().cloned());
    let x_max = x_map.len() - 1;

    let (mut count, mut root) = SegTree::with_size(x_max + 1, Additive);
    let mut roots = vec![root];

    for x in xs {
        root = count.set(root, x_map_inv[&x] as usize, 1);
        roots.push(root);
    }

    let q: usize = input.value();
    let mut ans = 0i32;
    for _ in 0..q {
        let a: u64 = input.value();
        let b: u64 = input.value();
        let c: u64 = input.value();
        let d: u64 = input.value();
        let k: usize = input.value();

        let mut l = ((a * ans.max(0) as u64 + b) % n as u64) as usize;
        let mut r = ((c * ans.max(0) as u64 + d) % n as u64) as usize;
        if l > r {
            std::mem::swap(&mut l, &mut r);
        }
        println!("{}..={}", l + 1, r + 1);

        let count_lt = |x: u32| {
            count.query_range(roots[r + 1], 0..x as usize)
                - count.query_range(roots[l], 0..x as usize)
        };

        for x in 0..=x_max + 1 {
            print!("{} ", count_lt(x as u32));
        }
        println!();

        let i = partition_point(0, x_max as u32 + 1, |x| count_lt(x + 1) < k as u32);
        println!("{:?}", (i, x_max));
        ans = if i == x_max as u32 + 1 {
            -1
        } else {
            x_map[i as usize] as i32
        };

        writeln!(output, "{}", ans).unwrap();
    }
}