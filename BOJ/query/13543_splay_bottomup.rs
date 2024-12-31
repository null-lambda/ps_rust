use std::io::Write;

use splay::SplayForest;

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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

const P: u64 = 1 << 32;

static mut COMB: [[u64; 11]; 11] = [[0; 11]; 11];

fn init_comb() {
    let n_max = 10;
    for i in 0..=n_max {
        unsafe {
            COMB[i][0] = 1;
            for j in 1..=i {
                COMB[i][j] = (COMB[i - 1][j - 1] + COMB[i - 1][j]) % P;
            }
        }
    }
}

fn comb(n: usize, k: usize) -> u64 {
    unsafe {
        debug_assert_ne!(COMB[0][0], 0, "Should call init_comb() first");
        COMB[n][k]
    }
}

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut res = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            res = res * base % P;
        }

        base = base * base % P;
        exp >>= 1;
    }
    res
}

#[allow(unused)]
pub mod splay {
    use super::*;

    use std::{
        cmp::Ordering,
        fmt::{self, Debug},
        iter,
        marker::PhantomData,
        mem::{self, MaybeUninit},
        num::NonZeroU32,
        ops::{Deref, DerefMut, Index, IndexMut, Range},
        ptr::{self, NonNull},
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        Left = 0,
        Right = 1,
    }

    impl Branch {
        pub fn inv(&self) -> Self {
            match self {
                Branch::Left => Branch::Right,
                Branch::Right => Branch::Left,
            }
        }
    }

    type Link<'pool> = Option<NodeRef<'pool>>;

    #[derive(Debug)]
    pub struct Node<'pool> {
        pub value: u64,
        pub sum: [u64; 11],
        pub size: u32,
        children: [Link<'pool>; 2],
        parent: Link<'pool>,
        _marker: PhantomData<&'pool ()>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct NodeRef<'pool> {
        idx: NonZeroU32,
        _marker: PhantomData<&'pool mut Node<'pool>>,
    }

    impl Debug for NodeRef<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct SplayForest<'pool> {
        pool: Vec<Node<'pool>>,
    }

    impl<'pool> SplayForest<'pool> {
        pub fn new() -> Self {
            let dummy = Node {
                value: 0,
                sum: Default::default(),
                size: 0,
                children: [None, None],
                parent: None,
                _marker: PhantomData,
            };
            Self { pool: vec![dummy] }
        }

        pub fn new_node(&mut self, value: u64, parent: Link<'pool>) -> NodeRef<'pool> {
            let idx = self.pool.len();
            self.pool.push(Node {
                value,
                sum: [value; 11],
                size: 1,
                children: [None, None],
                parent,
                _marker: PhantomData,
            });
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
                _marker: PhantomData,
            }
        }

        pub fn add_root(&mut self, value: u64) -> NodeRef<'pool> {
            let root = self.new_node(value, None);
            root
        }

        pub fn get_node<'a>(&'a self, ptr: NodeRef<'pool>) -> &'a Node<'pool> {
            &self.pool[ptr.idx.get() as usize]
        }

        pub fn get_node_mut<'a>(&'a mut self, ptr: NodeRef<'pool>) -> &'a mut Node<'pool> {
            &mut self.pool[ptr.idx.get() as usize]
        }

        pub fn get_node_with_children<'a>(
            &'a mut self,
            ptr: NodeRef<'pool>,
        ) -> (&'a mut Node<'pool>, [Option<&'a mut Node<'pool>>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = &mut *pool_ptr.add(ptr.idx.get() as usize);
                let children = node
                    .children
                    .map(|child| child.map(|child| &mut *pool_ptr.add(child.idx.get() as usize)));
                (node, children)
            }
        }

        fn push_down(&mut self, node: NodeRef<'pool>) {
            unimplemented!();
        }

        fn pull_up(&mut self, node: NodeRef<'pool>) {
            let (node, children) = self.get_node_with_children(node);
            node.size = 1;

            for child in children.iter().flatten() {
                node.size += child.size;
            }

            node.sum = [0; 11];
            let mut size_left = 1;
            if let Some(child) = children[Branch::Left as usize].as_ref() {
                size_left = child.size + 1;
                for i in 0..=10 {
                    node.sum[i] += child.sum[i];
                    node.sum[i] %= P;
                }
            }
            for i in 0..=10 {
                node.sum[i] = (node.sum[i] + node.value * pow(size_left as u64, i as u64)) % P;
            }
            if let Some(child) = children[Branch::Right as usize].as_ref() {
                for i in 0..=10 {
                    for k in 0..=i {
                        node.sum[i] += child.sum[k] * comb(i as usize, k as usize) % P
                            * pow(size_left as u64, (i - k) as u64);
                        node.sum[i] %= P;
                    }
                }
            }
        }

        pub fn is_root(&self, node: NodeRef<'pool>) -> bool {
            self.get_node(node).parent.is_none()
        }

        pub fn get_branch(&self, node: NodeRef<'pool>) -> Option<Branch> {
            let parent = self.get_node(node).parent?;
            Some(
                if self.get_node(parent).children[Branch::Left as usize] == Some(node) {
                    Branch::Left
                } else {
                    Branch::Right
                },
            )
        }

        fn attach(&mut self, node: NodeRef<'pool>, mut child: NodeRef<'pool>, branch: Branch) {
            unsafe {
                debug_assert_ne!(node, child);
                self.get_node_mut(node).children[branch as usize] = Some(child);
                self.get_node_mut(child).parent = Some(node);
            }
        }

        fn detach(&mut self, node: NodeRef<'pool>, branch: Branch) -> Option<NodeRef<'pool>> {
            let child = self.get_node_mut(node).children[branch as usize].take()?;
            self.get_node_mut(child).parent = None;
            Some(child)
        }

        fn rotate(&mut self, node: NodeRef<'pool>) {
            debug_assert!(!self.is_root(node));
            let branch = self.get_branch(node).unwrap();
            let mut child = self.detach(node, branch.inv());
            let parent = self.get_node(node).parent.unwrap();
            if let Some(child) = child {
                self.attach(parent, child, branch);
            } else {
                self.detach(parent, branch);
            }
            if let Some(mut grandparent) = self.get_node(parent).parent {
                self.attach(grandparent, node, self.get_branch(parent).unwrap());
            } else {
                self.get_node_mut(node).parent = None;
            }
            self.attach(node, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(node);
        }

        pub fn splay(&mut self, mut node: NodeRef<'pool>) {
            // eprintln!("splay {:?}", node);
            while let Some(parent) = self.get_node(node).parent {
                if let Some(grandparent) = self.get_node(parent).parent {
                    if self.get_branch(node) == self.get_branch(parent) {
                        self.rotate(parent);
                    } else {
                        self.rotate(node);
                    }
                }
                self.rotate(node);
                // self.debug_print_topo();
            }
        }

        fn first(&mut self, mut node: NodeRef<'pool>) -> NodeRef<'pool> {
            while let Some(left) = self.get_node(node).children[Branch::Left as usize] {
                node = left;
            }
            node
        }

        fn last(&mut self, mut node: NodeRef<'pool>) -> NodeRef<'pool> {
            while let Some(right) = self.get_node(node).children[Branch::Right as usize] {
                node = right;
            }
            node
        }

        #[must_use]
        pub fn splay_nth(&mut self, mut node: NodeRef<'pool>, mut n: usize) -> NodeRef<'pool> {
            debug_assert!(n < self.get_node(node).size as usize, "Out of bounds");
            loop {
                let node_ref = self.get_node(node);
                let mut left_size = node_ref.children[Branch::Left as usize]
                    .map_or(0, |x| self.get_node(x).size as usize);
                match n.cmp(&left_size) {
                    Ordering::Equal => break,
                    Ordering::Less => {
                        node = node_ref.children[Branch::Left as usize].unwrap();
                    }
                    Ordering::Greater => {
                        n -= left_size + 1;
                        node = node_ref.children[Branch::Right as usize].unwrap();
                    }
                }
            }
            self.splay(node);
            node
        }

        #[must_use]
        pub fn merge(&mut self, mut left: NodeRef<'pool>, right: NodeRef<'pool>) -> NodeRef<'pool> {
            let r = self.last(left);
            self.splay(r);
            self.attach(r, right, Branch::Right);
            self.pull_up(r);
            r
        }

        #[must_use]
        pub fn insert(&mut self, node: NodeRef<'pool>, pos: usize, value: u64) -> NodeRef<'pool> {
            debug_assert!(pos <= self.get_node(node).size as usize);
            let mut new_node = self.new_node(value, None);
            if pos == self.get_node(node).size as usize {
                let last = self.last(node);
                self.splay(last);
                self.attach(last, new_node, Branch::Right);
                self.pull_up(last);
                last
            } else {
                let mut right = self.splay_nth(node, pos);
                if let Some(left) = self.get_node(right).children[Branch::Left as usize] {
                    let last = self.last(left);
                    self.attach(last, new_node, Branch::Right);
                    self.splay(last);
                    last
                } else {
                    self.attach(right, new_node, Branch::Left);
                    self.pull_up(right);
                    right
                }
            }
        }

        #[must_use]
        pub fn remove(&mut self, mut node: NodeRef<'pool>, pos: usize) -> Option<NodeRef<'pool>> {
            debug_assert!(pos < self.get_node(node).size as usize);
            node = self.splay_nth(node, pos);
            let mut left = self.detach(node, Branch::Left);
            let mut right = self.detach(node, Branch::Right);
            if let Some((left, right)) = left.zip(right) {
                Some(self.merge(left, right))
            } else {
                left.or_else(|| right)
            }
        }

        #[must_use]
        pub fn set(&mut self, mut node: NodeRef<'pool>, pos: usize, value: u64) -> NodeRef<'pool> {
            node = self.splay_nth(node, pos);
            self.get_node_mut(node).value = value;
            node
        }

        #[must_use]
        pub fn extract_range(
            &mut self,
            node: NodeRef<'pool>,
            range: Range<usize>,
        ) -> (NodeRef<'pool>, NodeRef<'pool>) {
            let Range { start: l, end: r } = range;
            let n = self.get_node(node).size as usize;
            debug_assert!(l < r && r <= n);
            if r == self.get_node(node).size as usize {
                if l == 0 {
                    (node, node)
                } else {
                    //   l
                    //  / \
                    // .   m
                    let left = self.splay_nth(node, l - 1);
                    let mid = self.get_node(left).children[Branch::Right as usize].unwrap();
                    (left, mid)
                }
            } else {
                let right = self.splay_nth(node, r);
                if l == 0 {
                    //   r
                    //  / \
                    // m   .
                    let mid = self.get_node(right).children[Branch::Left as usize].unwrap();
                    (right, mid)
                } else {
                    //     r
                    //    / \
                    //   l   .
                    //  / \
                    // .   m
                    let left_mid = self.detach(right, Branch::Left).unwrap();
                    let mut left = self.splay_nth(left_mid, l - 1);
                    self.attach(right, left, Branch::Left);
                    self.pull_up(right);
                    let mid = self.get_node(left).children[Branch::Right as usize].unwrap();
                    (right, mid)
                }
            }
        }

        pub fn from_iter(iter: impl IntoIterator<Item = u64>) -> (Self, Option<NodeRef<'pool>>) {
            let mut tree = SplayForest::new();
            let mut root = None;
            for value in iter {
                let node = tree.new_node(value, None);
                root = Some(root.map_or(node, |root| {
                    tree.attach(node, root, Branch::Left);
                    tree.pull_up(node);
                    node
                }));
            }
            (tree, root)
        }

        pub fn preorder(&self, node: NodeRef<'pool>, mut visitor: impl FnMut(NodeRef<'pool>)) {
            fn rec<'pool>(
                tree: &SplayForest<'pool>,
                node: NodeRef<'pool>,
                visitor: &mut impl FnMut(NodeRef<'pool>),
            ) {
                visitor(node);
                let children = tree.get_node(node).children;
                for child in children.into_iter().flatten() {
                    rec(tree, child, visitor);
                }
            }
            rec(self, node, &mut visitor);
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();

    init_comb();

    let (mut tree, mut root) = SplayForest::from_iter((0..n).map(|_| input.value()));

    for _i_query in 0..input.value() {
        let cmd: u8 = input.value();
        match cmd {
            1 => {
                let p: usize = input.value();
                let x: u64 = input.value();
                root = Some(if let Some(root) = root {
                    tree.insert(root, p, x)
                } else {
                    tree.add_root(x)
                });
            }
            2 => {
                let p: usize = input.value();
                root = tree.remove(root.unwrap(), p);
            }
            3 => {
                let p: usize = input.value();
                let x: u64 = input.value();
                root = Some(tree.set(root.unwrap(), p, x));
                continue;
            }
            4 => {
                let l: usize = input.value();
                let r: usize = input.value();
                let k: usize = input.value();
                let (new_root, sub_node) = tree.extract_range(root.unwrap(), l..r + 1);
                root = Some(new_root);
                writeln!(output, "{}", tree.get_node(sub_node).sum[k]).unwrap();
            }
            _ => panic!(),
        }
    }
}
