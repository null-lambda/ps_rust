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

use std::{num::NonZeroU32, ops::Range};

type X = u32;

#[derive(Clone, Copy)]
pub struct NodeRef(NonZeroU32);

struct Node {
    value: X,
    children: Option<[NodeRef; 2]>,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            children: self.children,
        }
    }
}

pub struct SegTree {
    n: usize,
    max_height: u32,
    nodes: Vec<Node>,
}

impl SegTree {
    fn alloc(&mut self, node: Node) -> NodeRef {
        let idx = self.nodes.len();
        self.nodes.push(node);
        NodeRef(NonZeroU32::new(idx as u32).unwrap())
    }

    fn get_node(&self, idx: NodeRef) -> &Node {
        &self.nodes[idx.0.get() as usize]
    }

    fn clone(&mut self, idx: NodeRef) -> NodeRef {
        let node = self.nodes[idx.0.get() as usize].clone();
        self.alloc(node)
    }

    fn get_node_mut(&mut self, idx: NodeRef) -> &mut Node {
        &mut self.nodes[idx.0.get() as usize]
    }

    pub fn with_size(n: usize) -> (Self, NodeRef) {
        debug_assert!(n > 0);
        let n = n.next_power_of_two();
        let max_height = usize::BITS - n.leading_zeros();
        let dummy = Node {
            value: 0,
            children: None,
        };
        let mut this = Self {
            n,
            max_height,
            nodes: vec![dummy],
        };
        let root = this.with_size_rec(0..n);
        (this, root)
    }

    fn with_size_rec(&mut self, range: Range<usize>) -> NodeRef {
        debug_assert!(range.start < range.end);
        let Range { start, end } = range;
        let node = self.alloc(Node {
            value: 0,
            children: None,
        });
        if end - start > 1 {
            let mid = (start + end) >> 1;
            self.get_node_mut(node).children =
                Some([self.with_size_rec(start..mid), self.with_size_rec(mid..end)]);
        }
        node
    }

    pub fn modify(&mut self, root: NodeRef, idx: usize, f: impl FnOnce(X) -> X) -> NodeRef {
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

        let old = self.get_node(node).value.clone();
        let mut curr = self.alloc(Node {
            value: f(old),
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
                value: &self.get_node(left).value + &self.get_node(right).value,
                children: Some([left, right]),
            });
        }

        curr
    }

    pub fn get(&self, root: NodeRef, idx: usize) -> X {
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

    pub fn query_range(&self, root: NodeRef, range: Range<usize>) -> X {
        self.query_range_rec(range, 0..self.n as usize, root)
    }

    fn query_range_rec(
        &self,
        query_range: Range<usize>,
        node_range: Range<usize>,
        node: NodeRef,
    ) -> X {
        let Range { start, end } = node_range;
        let Range {
            start: query_start,
            end: query_end,
        } = query_range;
        if query_end <= start || end <= query_start {
            return 0;
        }
        if query_start <= start && end <= query_end {
            return self.get_node(node).value.clone();
        }
        let mid = (start + end) / 2;
        let c = unsafe { self.get_node(node).children.unwrap_unchecked() };
        self.query_range_rec(query_range.clone(), start..mid, c[0])
            + self.query_range_rec(query_range.clone(), mid..end, c[1])
    }

    pub fn query_xor(&self, left_root: NodeRef, right_root: NodeRef, x: X) -> X {
        self.query_xor_rec(x, left_root, right_root, self.max_height - 1, 1)
    }

    fn query_xor_rec(
        &self,
        x: X,
        left_node: NodeRef,
        right_node: NodeRef,
        height: u32,
        acc: X,
    ) -> X {
        if acc >= self.n as u32 {
            return acc - self.n as u32;
        }
        let primary = (x >> (height - 1)) & 1 == 0;
        let primary_size = self
            .get_node(self.get_node(right_node).children.unwrap()[primary as usize])
            .value
            - self
                .get_node(self.get_node(left_node).children.unwrap()[primary as usize])
                .value;

        let branch = if primary_size > 0 { primary } else { !primary };

        self.query_xor_rec(
            x,
            self.get_node(left_node).children.unwrap()[branch as usize],
            self.get_node(right_node).children.unwrap()[branch as usize],
            height - 1,
            acc << 1 | (branch as X),
        )
    }

    pub fn nth(&self, k: u32, left_root: NodeRef, right_root: NodeRef) -> X {
        self.query_kth_rec(k, left_root, right_root, self.max_height - 1, 0)
    }

    fn query_kth_rec(
        &self,
        k: u32,
        left_root: NodeRef,
        right_root: NodeRef,
        height: u32,
        acc: X,
    ) -> X {
        if height == 0 {
            return acc;
        }
        let left_size = self
            .get_node(self.get_node(right_root).children.unwrap()[0])
            .value
            - self
                .get_node(self.get_node(left_root).children.unwrap()[0])
                .value;
        let (k, branch) = if left_size >= k {
            (k, 0)
        } else {
            (k - left_size, 1)
        };

        self.query_kth_rec(
            k,
            self.get_node(left_root).children.unwrap()[branch],
            self.get_node(right_root).children.unwrap()[branch],
            height - 1,
            acc << 1 | (branch as X),
        )
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let x_max = 500_000usize.next_power_of_two();

    let (mut count, mut root) = SegTree::with_size(x_max + 1);
    let mut roots = vec![root];

    let m: usize = input.value();

    for _ in 0..m {
        match input.token() {
            "1" => {
                let x = input.value::<usize>();
                root = count.modify(root, x, |x| x + 1);
                roots.push(root);
            }
            "2" => {
                let l = input.value::<usize>();
                let r = input.value::<usize>();
                let x = input.value::<usize>();
                let ans = count.query_xor(roots[l - 1], roots[r], x as u32);
                writeln!(output, "{}", ans).unwrap();
            }
            "3" => {
                let k = input.value::<usize>();
                root = roots[roots.len() - k - 1];
                roots.truncate(roots.len() - k);
            }
            "4" => {
                let l = input.value::<usize>();
                let r = input.value::<usize>();
                let x = input.value::<usize>();
                let ans = count.query_range(roots[r], 0..x + 1)
                    - count.query_range(roots[l - 1], 0..x + 1);
                writeln!(output, "{}", ans).unwrap();
            }
            "5" => {
                let l = input.value::<usize>();
                let r = input.value::<usize>();
                let k: u32 = input.value();
                let ans = count.nth(k, roots[l - 1], roots[r]);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
