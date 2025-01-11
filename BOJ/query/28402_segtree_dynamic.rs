use std::{io::Write, ops::Range};

use segtree_dynamic::{Monoid, SegTree};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

pub mod segtree_dynamic {
    use std::{num::NonZeroU32, ops::Range};

    pub trait Monoid {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Clone, Copy)]
    struct NodeRef(NonZeroU32);

    struct Node<M: Monoid> {
        sum: M::X,
        children: [Option<NodeRef>; 2],
    }

    #[derive(Default)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        pool: Vec<Node<M>>,
        monoid: M,
    }

    impl<M: Monoid> SegTree<M> {
        pub fn with_size(n: usize, monoid: M) -> Self {
            Self {
                n,
                pool: vec![Node {
                    sum: monoid.id(),
                    children: [None, None],
                }],
                monoid,
            }
        }

        pub fn modify(&mut self, idx: usize, f: impl FnOnce(&mut M::X)) {
            debug_assert!(idx < self.n);
            self.modify_rec(idx, f, 0, 0..self.n);
        }

        fn pull_up(&mut self, u: usize) {
            let lhs = self.pool[u].children[0].map_or(self.monoid.id(), |c| {
                self.pool[c.0.get() as usize].sum.clone()
            });
            let rhs = self.pool[u].children[1].map_or(self.monoid.id(), |c| {
                self.pool[c.0.get() as usize].sum.clone()
            });
            self.pool[u].sum = self.monoid.op(&lhs, &rhs);
        }

        fn modify_rec(
            &mut self,
            idx: usize,
            f: impl FnOnce(&mut M::X),
            u: usize,
            node: Range<usize>,
        ) {
            if node.start + 1 == node.end {
                f(&mut self.pool[u].sum);
                return;
            }

            let mid = node.start + node.end >> 1;
            let branch = !(idx < mid) as usize;

            let c = match self.pool[u].children[branch] {
                Some(c) => c,
                None => {
                    let c = NodeRef(unsafe { NonZeroU32::new_unchecked(self.pool.len() as u32) });
                    self.pool.push(Node {
                        sum: self.monoid.id(),
                        children: [None, None],
                    });
                    self.pool[u].children[branch] = Some(c);
                    c
                }
            };

            if branch == 0 {
                self.modify_rec(idx, f, c.0.get() as usize, node.start..mid);
            } else {
                self.modify_rec(idx, f, c.0.get() as usize, mid..node.end);
            }
            self.pull_up(u);
        }

        pub fn sum_range(&self, range: Range<usize>) -> M::X {
            self.sum_range_rec(range, 0, 0..self.n)
        }

        fn sum_range_rec(&self, query: Range<usize>, u: usize, node: Range<usize>) -> M::X {
            if node.end <= query.start || query.end <= node.start {
                return self.monoid.id();
            }
            if query.start <= node.start && node.end <= query.end {
                return self.pool[u].sum.clone();
            }

            let mid = node.start + node.end >> 1;
            let lhs = self.pool[u].children[0].map_or(self.monoid.id(), |c| {
                self.sum_range_rec(query.clone(), c.0.get() as usize, node.start..mid)
            });
            let rhs = self.pool[u].children[1].map_or(self.monoid.id(), |c| {
                self.sum_range_rec(query, c.0.get() as usize, mid..node.end)
            });
            self.monoid.op(&lhs, &rhs)
        }
    }
}

#[derive(Debug, Default)]
struct MaxOp;

impl Monoid for MaxOp {
    type X = i32;
    fn id(&self) -> i32 {
        std::i32::MIN
    }
    fn op(&self, a: &i32, b: &i32) -> i32 {
        (*a).max(*b)
    }
}

#[derive(Default)]
struct T2Agg {
    weights: Vec<(u32, i32)>,
    tree: SegTree<MaxOp>,
}

impl T2Agg {
    fn singleton(n: usize, t1_euler_idx: usize, weight: i32) -> Self {
        let mut tree = SegTree::with_size(n, MaxOp);
        tree.modify(t1_euler_idx, |x| *x = weight);
        Self {
            weights: vec![(t1_euler_idx as u32, weight)],
            tree,
        }
    }

    fn pull_from(&mut self, mut child: Self) {
        if self.weights.len() < child.weights.len() {
            std::mem::swap(self, &mut child);
        }
        for (t1_euler_idx, weight) in child.weights {
            self.weights.push((t1_euler_idx, weight));
            self.tree.modify(t1_euler_idx as usize, |x| *x = weight);
        }
    }

    fn finalize(&mut self, t1_subtree_range: Range<usize>) -> i32 {
        self.tree.sum_range(t1_subtree_range)
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    // Outline:
    // For each node i in T2, maintain a max segment tree on an Euler tour of T1,
    // consisting of subtree_T2(i) nodes and their corresponding a_i's.
    // (Use a dynamic segment tree with smaller-to-larger propagation.)
    // Then, perform a range query representing subtree_T2(i).

    let n: usize = input.value();
    let xs: Vec<i32> = (0..n).map(|_| input.i32()).collect();

    let root = 0;
    let mut read_tree = || {
        let mut degree = vec![0; n];
        let mut xor_neighbors = vec![0; n];
        degree[root] += 2;
        for _ in 0..n - 1 {
            let u = input.u32() - 1;
            let v = input.u32() - 1;
            degree[u as usize] += 1;
            degree[v as usize] += 1;
            xor_neighbors[u as usize] ^= v;
            xor_neighbors[v as usize] ^= u;
        }
        (degree, xor_neighbors)
    };

    let (t1_euler_in, t1_size) = {
        let (mut degree, mut xor_neighbors) = read_tree();
        let mut size = vec![1; n];
        let mut topological_order = vec![];
        for mut u in 0..n as u32 {
            while degree[u as usize] == 1 {
                let p = xor_neighbors[u as usize];
                degree[u as usize] -= 1;
                degree[p as usize] -= 1;
                xor_neighbors[p as usize] ^= u;
                topological_order.push((u, p));

                size[p as usize] += size[u as usize];

                u = p;
            }
        }

        let mut euler_in = size.clone(); // 1-based
        for (u, p) in topological_order.into_iter().rev() {
            let last_idx = euler_in[p as usize];
            euler_in[p as usize] -= euler_in[u as usize];
            euler_in[u as usize] = last_idx;
        }

        (euler_in, size)
    };
    let t1_euler_in = |u: usize| t1_euler_in[u] as usize - 1; // 0-based
    let t1_subtree_range = |u: usize| t1_euler_in(u)..t1_euler_in(u) + t1_size[u];

    let mut t2_dp: Vec<_> = (0..n)
        .map(|u| T2Agg::singleton(n, t1_euler_in(u), xs[u]))
        .collect();

    let mut ans = vec![0; n];
    let (mut t2_degree, mut t2_xor_neighbors) = read_tree();
    for mut u in 0..n as u32 {
        while t2_degree[u as usize] == 1 {
            let p = t2_xor_neighbors[u as usize];
            t2_degree[u as usize] -= 1;
            t2_degree[p as usize] -= 1;
            t2_xor_neighbors[p as usize] ^= u;

            let mut t2_dp_u = std::mem::take(&mut t2_dp[u as usize]);
            ans[u as usize] = t2_dp_u.finalize(t1_subtree_range(u as usize));
            t2_dp[p as usize].pull_from(t2_dp_u);

            u = p;
        }
    }
    ans[root] = t2_dp[root as usize].finalize(t1_subtree_range(root));

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
