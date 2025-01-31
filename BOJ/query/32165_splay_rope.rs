use std::io::Write;

use splay::{IntrusiveNode, Link, NodeSpec, SizedNode, SplayForest};

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

pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        cmp::Ordering,
        fmt::{self, Debug},
        num::NonZeroU32,
        ops::Range,
    };

    // Adjoin an identity element to a binary operation.
    fn lift_binary<A>(
        combine: impl FnOnce(A, A) -> A,
    ) -> impl FnOnce(Option<A>, Option<A>) -> Option<A> {
        |lhs, rhs| match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => Some(combine(lhs, rhs)),
            (None, rhs) => rhs,
            (lhs, None) => lhs,
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        Left = 0,
        Right = 1,
    }

    impl Branch {
        pub fn usize(self) -> usize {
            self as usize
        }

        pub fn inv(&self) -> Self {
            match self {
                Branch::Left => Branch::Right,
                Branch::Right => Branch::Left,
            }
        }
    }

    // Intrusive node link, invertible.
    #[derive(Default, Debug)]
    pub struct Link {
        pub inv: bool,
        pub children: [Option<NodeRef>; 2],
        pub parent: Option<NodeRef>,
    }

    pub trait IntrusiveNode {
        fn link(&self) -> &Link;
        fn link_mut(&mut self) -> &mut Link;
    }

    pub trait NodeSpec: IntrusiveNode {
        fn push_down(&mut self, _children: [Option<&mut Self>; 2]) {}
        fn pull_up(&mut self, _children: [Option<&mut Self>; 2]) {}

        // type Cx;
        // fn push_down(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
        // fn pull_up(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
    }

    pub trait SizedNode: NodeSpec {
        fn size(&self) -> usize;
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeRef {
        pub idx: NonZeroU32,
    }

    impl NodeRef {
        fn get(&self) -> usize {
            self.idx.get() as usize
        }
    }

    impl Debug for NodeRef {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct SplayForest<V> {
        pub pool: Vec<V>,
    }

    impl<V: NodeSpec> SplayForest<V> {
        pub fn new() -> Self {
            let dummy = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            Self { pool: vec![dummy] }
        }

        pub fn add_root(&mut self, node: V) -> NodeRef {
            let idx = self.pool.len();
            self.pool.push(node);
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub fn get<'a>(&'a self, u: NodeRef) -> &'a V {
            &self.pool[u.get()]
        }

        pub unsafe fn get_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut V {
            &mut self.pool[u.get()]
        }

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut V, [Option<&'a mut V>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = &mut *pool_ptr.add(u.get());
                let children = node
                    .link()
                    .children
                    .map(|child| child.map(|child| &mut *pool_ptr.add(child.get())));
                (node, children)
            }
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let node = self.get_mut(u);
                let link = node.link_mut();
                if link.inv {
                    link.inv = false;
                    link.children.swap(0, 1);
                    for child in link.children.into_iter().flatten() {
                        self.get_mut(child).link_mut().inv ^= true;
                    }
                }

                let (node, children) = self.get_with_children(u);
                node.push_down(children);
            }
        }

        fn pull_up(&mut self, node: NodeRef) {
            unsafe {
                let (node, children) = self.get_with_children(node);
                node.pull_up(children);
            }
        }

        pub fn get_parent(&self, u: NodeRef) -> Option<(NodeRef, Branch)> {
            let p = self.get(u).link().parent?;
            if self.get(p).link().children[Branch::Left.usize()] == Some(u) {
                Some((p, Branch::Left))
            } else if self.get(p).link().children[Branch::Right.usize()] == Some(u) {
                Some((p, Branch::Right))
            } else {
                None
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.get_parent(u).is_none()
        }

        fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_mut(child).link_mut().parent = Some(u);
            }
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            unsafe {
                let child = self.get_mut(u).link_mut().children[branch as usize].take()?;
                self.get_mut(child).link_mut().parent = None;
                Some(child)
            }
        }

        fn rotate(&mut self, u: NodeRef) {
            let (parent, branch) = self.get_parent(u).expect("Root shouldn't be rotated");
            let child = self.detach(u, branch.inv());
            if let Some(child) = child {
                self.attach(parent, child, branch);
            } else {
                self.detach(parent, branch);
            }

            match self.get_parent(parent) {
                Some((grandparent, grandbranch)) => {
                    self.attach(grandparent, u, grandbranch);
                }
                None => unsafe {
                    self.get_mut(u).link_mut().parent = None;
                },
            }
            self.attach(u, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            while let Some((parent, branch)) = self.get_parent(u) {
                if let Some((grandparent, grandbranch)) = self.get_parent(parent) {
                    self.push_down(grandparent);
                    self.push_down(parent);
                    self.push_down(u);
                    if branch != grandbranch {
                        self.rotate(u);
                    } else {
                        self.rotate(parent);
                    }
                } else {
                    self.push_down(parent);
                    self.push_down(u);
                }
                self.rotate(u);
            }
            self.push_down(u);
        }

        pub fn collect_from(&mut self, iter: impl IntoIterator<Item = V>) -> Option<NodeRef> {
            let mut iter = iter.into_iter();
            let mut root = self.add_root(iter.next()?);
            for node in iter {
                let u = self.add_root(node);
                self.attach(u, root, Branch::Left);
                self.pull_up(u);
                root = u;
            }
            Some(root)
        }

        // Breaks amortized time complexity if not splayed afterwards.
        pub unsafe fn find_by(
            &self,
            mut u: NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) -> NodeRef {
            while let Some(next) =
                next(self, u).and_then(|branch| self.get(u).link().children[branch.usize()])
            {
                u = next;
            }
            u
        }

        // Caution: if u is not a root, then only the subtree nodes can be accessed.
        // Call splay(u) beforehand to walk on the full tree.
        pub fn splay_by(
            &mut self,
            u: &mut NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) {
            *u = unsafe { self.find_by(*u, &mut next) };
            self.splay(*u);
        }

        pub fn splay_first(&mut self, u: &mut NodeRef) {
            self.splay_by(u, |_, _| Some(Branch::Left))
        }

        pub fn splay_last(&mut self, u: &mut NodeRef) {
            self.splay_by(u, |_, _| Some(Branch::Right))
        }

        pub fn push(&mut self, root: &mut NodeRef, node: V) {
            self.splay_last(root);
            let new = self.add_root(node);
            self.attach(*root, new, Branch::Right);
            self.pull_up(*root);
        }

        pub fn inorder(&mut self, u: NodeRef, visitor: &mut impl FnMut(&mut Self, NodeRef)) {
            if let Some(left) = self.get(u).link().children[Branch::Left.usize()] {
                self.inorder(left, visitor);
            }
            visitor(self, u);
            if let Some(right) = self.get(u).link().children[Branch::Right.usize()] {
                self.inorder(right, visitor);
            }
        }

        pub fn split_left(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let left = self.detach(u, Branch::Left)?;
            self.pull_up(u);
            Some(left)
        }

        pub fn merge_nonnull(&mut self, mut lhs: NodeRef, mut rhs: NodeRef) -> NodeRef {
            self.splay(lhs);
            self.splay_last(&mut lhs);
            self.splay(rhs);
            self.splay_first(&mut rhs);
            assert!(self.is_root(lhs) && self.is_root(rhs) && lhs != rhs);
            self.attach(rhs, lhs, Branch::Left);
            self.pull_up(rhs);
            rhs
        }

        pub fn merge(&mut self, lhs: Option<NodeRef>, rhs: Option<NodeRef>) -> Option<NodeRef> {
            lift_binary(|lhs, rhs| self.merge_nonnull(lhs, rhs))(lhs, rhs)
        }

        pub fn remove(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let left = self.detach(u, Branch::Left);
            let right = self.detach(u, Branch::Right);
            self.pull_up(u);

            self.merge(left, right)
        }
    }

    impl<V: SizedNode> SplayForest<V> {
        pub fn splay_nth(&mut self, u: &mut NodeRef, mut idx: usize) {
            debug_assert!(idx < self.get(*u).size());
            self.splay_by(u, |forest, u| {
                let left_size = forest.get(u).link().children[Branch::Left.usize()]
                    .map_or(0, |left| forest.get(left).size());
                match idx.cmp(&left_size) {
                    Ordering::Equal => None,
                    Ordering::Less => Some(Branch::Left),
                    Ordering::Greater => {
                        idx -= left_size as usize + 1;
                        Some(Branch::Right)
                    }
                }
            });
        }

        pub fn position(&mut self, u: NodeRef) -> usize {
            self.splay(u);
            self.get(u).link().children[Branch::Left.usize()]
                .map_or(0, |left| self.get(left).size())
        }

        pub fn split_at(
            &mut self,
            mut u: NodeRef,
            idx: usize,
        ) -> (Option<NodeRef>, Option<NodeRef>) {
            debug_assert!(idx <= self.get(u).size());
            if idx == self.get(u).size() {
                return (Some(u), None);
            } else {
                self.splay_nth(&mut u, idx);
                let left = self.split_left(u);
                (left, Some(u))
            }
        }

        pub fn insert_at(&mut self, root: &mut NodeRef, idx: usize, u: NodeRef) {
            debug_assert!(idx <= self.get(*root).size());
            let (lhs, rhs) = self.split_at(*root, idx);
            let mid = self.merge(lhs, Some(u));
            *root = self.merge(mid, rhs).unwrap()
        }

        pub fn with_range(
            &mut self,
            root: &mut NodeRef,
            range: Range<usize>,
            mut f: impl FnMut(&V),
        ) {
            unimplemented!()
        }
    }
}

#[derive(Default, Debug)]
pub struct RopeNode {
    value: u32,
    size: u32,
    link: Link,
}

impl RopeNode {
    fn new(value: u32) -> Self {
        Self {
            value,
            size: 1,
            link: Default::default(),
        }
    }
}

impl IntrusiveNode for RopeNode {
    fn link(&self) -> &Link {
        &self.link
    }

    fn link_mut(&mut self) -> &mut Link {
        &mut self.link
    }
}

impl NodeSpec for RopeNode {
    fn push_down(&mut self, _children: [Option<&mut Self>; 2]) {}
    fn pull_up(&mut self, children: [Option<&mut Self>; 2]) {
        self.size = 1;
        for c in children.iter().flatten() {
            self.size += c.size;
        }
    }
}

impl SizedNode for RopeNode {
    fn size(&self) -> usize {
        self.size as usize
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut forest = SplayForest::new();

    let mut front = vec![];
    for i in 0..n {
        front.push(forest.add_root(RopeNode::new(i as u32)));
    }

    let mut transpositions: Vec<Option<_>> = vec![];
    for _ in 0..q {
        let mut history = None;
        match input.token() {
            "s" => {
                let i = input.value::<u32>() - 1;
                let j = input.value::<u32>() - 1;
                let ci = forest.add_root(RopeNode::new(i));
                let cj = forest.add_root(RopeNode::new(j));
                forest.merge_nonnull(front[j as usize], ci);
                forest.merge_nonnull(front[i as usize], cj);
                front[i as usize] = ci;
                front[j as usize] = cj;
                history = Some((ci, cj));
            }
            "f" | "r" => {
                let i = input.value::<usize>() - 1;
                let (ci, cj) = transpositions[i].unwrap();
                let prefix_i = unsafe { forest.split_left(ci).unwrap_unchecked() };
                let prefix_j = unsafe { forest.split_left(cj).unwrap_unchecked() };
                forest.merge_nonnull(prefix_j, ci);
                forest.merge_nonnull(prefix_i, cj);
            }
            "q" => {
                let i = input.value::<usize>() - 1;
                let mut h = front[i];
                forest.splay(h);
                forest.splay_first(&mut h);
                let ans = forest.get(h).value + 1;
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }

        transpositions.push(history);
    }
}
