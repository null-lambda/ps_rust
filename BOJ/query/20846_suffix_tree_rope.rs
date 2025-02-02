use std::io::Write;

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

pub mod debug {
    #[cfg(debug_assertions)]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(T);

    #[cfg(not(debug_assertions))]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(std::marker::PhantomData<T>);

    impl<T> Label<T> {
        #[inline]
        pub fn new(value: T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(value)
            }
            #[cfg(not(debug_assertions))]
            {
                Self(Default::default())
            }
        }

        #[inline]
        pub fn new_with(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(value())
            }
            #[cfg(not(debug_assertions))]
            {
                Self(Default::default())
            }
        }
    }

    impl<T: std::fmt::Debug> std::fmt::Debug for Label<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            #[cfg(debug_assertions)]
            {
                write!(f, "{:?}", self.0)
            }
            #[cfg(not(debug_assertions))]
            {
                write!(f, "()")
            }
        }
    }
}

pub mod suffix_trie {
    /// O(N) representation of a suffix trie using a suffix automaton.
    /// References: https://cp-algorithms.com/string/suffix-automaton.html

    pub type T = u32;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = 1 << 30;

    // HashMap-based transition table, for generic types
    #[derive(Clone, Debug, Default)]
    pub struct TransitionTable(std::collections::HashMap<T, NodeRef>);

    impl TransitionTable {
        pub fn get(&self, c: T) -> NodeRef {
            self.0.get(&c).copied().unwrap_or(UNSET)
        }

        pub fn set(&mut self, c: T, u: NodeRef) {
            self.0.insert(c, u);
        }
    }

    // array-based transition table, for small set of alphabets.
    pub const N_ALPHABET: usize = 26;

    // Since the number of transition is O(N), Most transitions tables are slim.
    pub const STACK_CAP: usize = 1;

    //     #[derive(Clone, Debug)]
    //     pub enum TransitionTable {
    //         Stack([(T, NodeRef); STACK_CAP]),
    //         HeapFixed(Box<[NodeRef; N_ALPHABET]>),
    //     }

    //     impl TransitionTable {
    //         pub fn get(&self, key: T) -> NodeRef {
    //             match self {
    //                 Self::Stack(items) => items
    //                     .iter()
    //                     .find_map(|&(c, u)| (c == key).then(|| u))
    //                     .unwrap_or(UNSET),
    //                 Self::HeapFixed(vec) => vec[key as usize],
    //             }
    //         }
    //         pub fn set(&mut self, key: T, u: NodeRef) {
    //             match self {
    //                 Self::Stack(arr) => {
    //                     for (c, v) in arr.iter_mut() {
    //                         if c == &(N_ALPHABET as u8) {
    //                             *c = key;
    //                         }
    //                         if c == &key {
    //                             *v = u;
    //                             return;
    //                         }
    //                     }

    //                     let mut vec = Box::new([UNSET; N_ALPHABET]);
    //                     for (c, v) in arr.iter() {
    //                         vec[*c as usize] = *v;
    //                     }
    //                     vec[key as usize] = u;
    //                     *self = Self::HeapFixed(vec);
    //                 }
    //                 Self::HeapFixed(vec) => vec[key as usize] = u,
    //             }
    //         }
    //     }

    //     impl Default for TransitionTable {
    //         fn default() -> Self {
    //             Self::Stack([(N_ALPHABET as u8, UNSET); STACK_CAP])
    //         }
    //     }

    #[derive(Debug, Default, Clone)]
    pub struct Node {
        /// DAWG of the string
        pub children: TransitionTable,

        /// Suffix tree of the reversed string (equivalent to the compressed trie of the reversed prefixes)
        pub rev_depth: u32,
        pub rev_parent: NodeRef,

        /// An auxilary tag for the substring reconstruction
        pub first_endpos: u32,
    }

    impl Node {
        pub fn is_rev_terminal(&self) -> bool {
            self.first_endpos + 1 == self.rev_depth
        }
    }

    pub struct SuffixAutomaton {
        pub nodes: Vec<Node>,
        pub tail: NodeRef,
    }

    impl SuffixAutomaton {
        pub fn new() -> Self {
            let root = Node {
                children: Default::default(),
                rev_parent: UNSET,
                rev_depth: 0,

                first_endpos: UNSET,
            };

            Self {
                nodes: vec![root],
                tail: 0,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let u = self.nodes.len() as u32;
            self.nodes.push(node);
            u
        }

        pub fn push(&mut self, x: T) -> NodeRef {
            let u = self.alloc(Node {
                children: Default::default(),
                rev_parent: 0,
                rev_depth: self.nodes[self.tail as usize].rev_depth + 1,

                first_endpos: self.nodes[self.tail as usize].rev_depth,
            });
            let mut p = self.tail;
            self.tail = u;

            while p != UNSET {
                let c = self.nodes[p as usize].children.get(x);
                if c != UNSET {
                    if self.nodes[p as usize].rev_depth + 1 == self.nodes[c as usize].rev_depth {
                        self.nodes[u as usize].rev_parent = c;
                    } else {
                        let c_cloned = self.alloc(Node {
                            rev_depth: self.nodes[p as usize].rev_depth + 1,
                            ..self.nodes[c as usize].clone()
                        });

                        self.nodes[u as usize].rev_parent = c_cloned;
                        self.nodes[c as usize].rev_parent = c_cloned;

                        while p != UNSET && self.nodes[p as usize].children.get(x) == c {
                            self.nodes[p as usize].children.set(x, c_cloned);
                            p = self.nodes[p as usize].rev_parent;
                        }
                    }

                    break;
                }

                self.nodes[p as usize].children.set(x, u);
                p = self.nodes[p as usize].rev_parent;
            }

            u
        }
    }
}

pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        cmp::Ordering,
        fmt::{self, Debug},
        mem::MaybeUninit,
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
        pub lazy_inv: bool,
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
        fn on_reverse(&mut self) {}

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
        pub pool: Vec<MaybeUninit<V>>,
    }

    impl<V: NodeSpec> SplayForest<V> {
        pub fn new() -> Self {
            let dummy = MaybeUninit::uninit();
            Self { pool: vec![dummy] }
        }

        pub fn add_root(&mut self, node: V) -> NodeRef {
            let idx = self.pool.len();
            self.pool.push(MaybeUninit::new(node));
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub fn get<'a>(&'a self, u: NodeRef) -> &'a V {
            unsafe { &self.pool[u.get()].assume_init_ref() }
        }

        pub unsafe fn get_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut V {
            self.pool[u.get()].assume_init_mut()
        }

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut V, [Option<&'a mut V>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = (&mut *pool_ptr.add(u.get())).assume_init_mut();
                let children = node.link().children.map(|child| {
                    child.map(|child| (&mut *pool_ptr.add(child.get())).assume_init_mut())
                });
                (node, children)
            }
        }

        pub fn reverse(&mut self, u: NodeRef) {
            unsafe {
                let node = self.get_mut(u);
                node.on_reverse();
                node.link_mut().lazy_inv ^= true;
            }
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let link = self.get_mut(u).link_mut();
                if link.lazy_inv {
                    link.lazy_inv = false;
                    link.children.swap(0, 1);
                    for child in link.children.into_iter().flatten() {
                        self.reverse(child);
                    }
                }

                let (node, children) = self.get_with_children(u);
                node.push_down(children);
            }
        }

        pub fn pull_up(&mut self, node: NodeRef) {
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

        pub fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_mut(child).link_mut().parent = Some(u);
            }
        }

        pub fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
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

        // Caution: breaks amortized time complexity if not splayed afterwards.
        pub unsafe fn find_by(
            &mut self,
            mut u: NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) -> NodeRef {
            loop {
                self.push_down(u);
                if let Some(next) =
                    next(self, u).and_then(|branch| self.get(u).link().children[branch.usize()])
                {
                    u = next;
                } else {
                    break;
                }
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

        pub fn predecessor(&mut self, mut u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            u = self.get(u).link().children[Branch::Left.usize()]?;
            self.splay_last(&mut u);
            Some(u)
        }

        pub fn successor(&mut self, mut u: NodeRef) -> NodeRef {
            self.splay(u);
            u = self.get(u).link().children[Branch::Right.usize()].unwrap();
            self.splay_first(&mut u);
            u
        }

        pub fn inorder(&mut self, u: NodeRef, visitor: &mut impl FnMut(&mut Self, NodeRef)) {
            self.push_down(u);
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

        pub fn split_right(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let right = self.detach(u, Branch::Right)?;
            self.pull_up(u);
            Some(right)
        }

        pub fn merge_nonnull(&mut self, mut lhs: NodeRef, mut rhs: NodeRef) -> NodeRef {
            self.splay(lhs);
            self.splay_last(&mut lhs);
            self.splay(rhs);
            self.splay_first(&mut rhs);
            debug_assert!(self.is_root(lhs) && self.is_root(rhs) && lhs != rhs);
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

    impl<V> Drop for SplayForest<V> {
        fn drop(&mut self) {
            for node in self.pool.iter_mut().skip(1) {
                unsafe {
                    node.assume_init_drop();
                }
            }
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

        pub fn split_nonnull_at(
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

        pub fn split_at(
            &mut self,
            u: Option<NodeRef>,
            idx: usize,
        ) -> (Option<NodeRef>, Option<NodeRef>) {
            if let Some(u) = u {
                self.split_nonnull_at(u, idx)
            } else {
                debug_assert_eq!(idx, 0);
                (None, None)
            }
        }

        pub fn insert_at(&mut self, root: &mut NodeRef, idx: usize, u: NodeRef) {
            debug_assert!(idx <= self.get(*root).size());
            let (lhs, rhs) = self.split_nonnull_at(*root, idx);
            let mid = self.merge(lhs, Some(u));
            *root = self.merge(mid, rhs).unwrap()
        }

        pub fn with_range(
            &mut self,
            root: &mut NodeRef,
            range: Range<usize>,
            f: impl FnOnce(&mut Self, NodeRef),
        ) {
            assert!(range.start < range.end && range.end <= self.get(*root).size());
            let (rest, rhs) = self.split_nonnull_at(*root, range.end);
            let (lhs, mid) = self.split_nonnull_at(rest.unwrap(), range.start);
            self.splay(unsafe { mid.unwrap_unchecked() });
            f(self, unsafe { mid.unwrap_unchecked() });
            self.merge(lhs, mid);
            *root = self.merge(rest, rhs).unwrap();
        }
    }
}

const UNSET: u32 = !0;

struct SuffixTreeNode {
    trie_node_idx: u32,
    terminal_idx: u32,
    terminal_count: u32,

    debug_tag: debug::Label<(u32, Box<[u32]>)>,

    node_size: u32,
    link: splay::Link,
}

impl SuffixTreeNode {
    fn new(
        debug_tag: debug::Label<(u32, Box<[u32]>)>,
        trie_node_idx: u32,
        terminal_idx: Option<u32>,
    ) -> Self {
        Self {
            trie_node_idx,
            terminal_idx: terminal_idx.unwrap_or(UNSET),
            terminal_count: terminal_idx.is_some() as u32,

            debug_tag,

            node_size: 1,
            link: Default::default(),
        }
    }

    fn terminal_count_le(&self, forest: &splay::SplayForest<Self>) -> u32 {
        self.link.children[0].map_or(0, |lhs| forest.get(lhs).terminal_count)
            + (self.terminal_idx != UNSET) as u32
    }
}

impl splay::IntrusiveNode for SuffixTreeNode {
    fn link(&self) -> &splay::Link {
        &self.link
    }
    fn link_mut(&mut self) -> &mut splay::Link {
        &mut self.link
    }
}

impl splay::NodeSpec for SuffixTreeNode {
    fn pull_up(&mut self, children: [Option<&mut Self>; 2]) {
        self.terminal_count = (self.terminal_idx != UNSET) as u32;
        self.node_size = 1;
        for child in children.iter().flatten() {
            self.terminal_count += child.terminal_count;
            self.node_size += child.node_size;
        }
    }
}

impl splay::SizedNode for SuffixTreeNode {
    fn size(&self) -> usize {
        self.node_size as usize
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: u32 = input.value();
    let xs: Vec<u32> = (0..n).map(|_| input.u32()).collect();

    let mut automaton = suffix_trie::SuffixAutomaton::new();
    let mut terminal_idx = vec![];
    for (i, x) in xs.iter().copied().enumerate().rev() {
        let u = automaton.push(x);
        terminal_idx.resize(automaton.nodes.len(), None);
        terminal_idx[u as usize] = Some(i as u32)
    }
    let n_nodes = automaton.nodes.len();

    let mut keys = vec![0; n_nodes];
    let mut size = vec![1u32; n_nodes];
    let mut euler_tour = vec![];
    let mut rev_children = vec![vec![]; n_nodes];
    {
        for u in 1..n_nodes {
            let p = automaton.nodes[u as usize].rev_parent;

            let node = &automaton.nodes[u as usize];
            let parent = &automaton.nodes[p as usize];
            // let substr_len = node.rev_depth - parent.rev_depth;
            // let substr = &xs
            //     [xs.len() - 1 - (node.first_endpos as usize - parent.rev_depth as usize)..]
            //     [..substr_len as usize];
            // let key = substr[0];
            let key = xs[xs.len() - 1 - (node.first_endpos as usize - parent.rev_depth as usize)];
            keys[u as usize] = key;

            rev_children[p as usize].push((key, u as u32));
        }

        let mut stack = vec![(0, 0)];
        while let Some((u, iv)) = stack.pop() {
            if iv == 0 {
                rev_children[u as usize].sort_unstable_by_key(|&(key, _)| key);
                euler_tour.push(u as i32);
            }
            if iv < rev_children[u as usize].len() {
                let (_, v) = rev_children[u as usize][iv as usize];
                stack.push((u, iv + 1));
                stack.push((v, 0));
            } else {
                euler_tour.push(-1 - u as i32);
            }
        }

        for &u in euler_tour.iter().rev() {
            if u < 0 {
                continue;
            }
            let p = automaton.nodes[u as usize].rev_parent;
            if p == suffix_trie::UNSET {
                continue;
            }
            size[p as usize] += size[u as usize];
        }
    }

    let mut forest = splay::SplayForest::new();
    let nodes: Vec<_> = (0..n_nodes)
        .flat_map(|u| {
            let p = automaton.nodes[u as usize].rev_parent;
            if p == suffix_trie::UNSET {
                return [
                    SuffixTreeNode::new(Default::default(), (u << 1) as u32, None),
                    SuffixTreeNode::new(
                        Default::default(),
                        (u << 1 | 1) as u32,
                        terminal_idx[u as usize],
                    ),
                ];
            }

            let node = &automaton.nodes[u as usize];
            let parent = &automaton.nodes[p as usize];
            let tag = || {
                debug::Label::new_with(|| {
                    let substr_len = node.rev_depth - parent.rev_depth;
                    let substr = &xs
                        [xs.len() - 1 - (node.first_endpos as usize - parent.rev_depth as usize)..]
                        [..substr_len as usize];
                    (keys[u], substr.iter().copied().collect())
                })
            };
            [
                SuffixTreeNode::new(tag(), (u << 1) as u32, terminal_idx[u as usize]),
                SuffixTreeNode::new(tag(), (u << 1 | 1) as u32, None),
            ]
        })
        .map(|node| forest.add_root(node))
        .collect();

    let mut rotation_events = vec![];
    for u in 1..n_nodes {
        let p = automaton.nodes[u as usize].rev_parent;
        if rev_children[p as usize].len() == 1 {
            continue;
        }
        rotation_events.push((m - keys[u], p));
    }
    rotation_events.sort_unstable_by_key(|&(key, ..)| key);

    let mut root = nodes[euler_tour[0] as usize];
    for &u in &euler_tour[1..] {
        let u = nodes[if u >= 0 {
            (u as usize) << 1
        } else {
            ((-1 - u) as usize) << 1 | 1
        }];
        forest.attach(u, root, splay::Branch::Left);
        forest.pull_up(u);
        root = u;
    }

    let q: usize = input.value();
    let mut queries: Vec<_> = (0..q)
        .map(|i| {
            let shift = input.u32();
            let k = input.u32() - 1;
            (shift % m, k, i as u32)
        })
        .collect();
    queries.sort_unstable_by_key(|&(shift, ..)| shift);

    let mut rotation_iter = rotation_events.into_iter().peekable();
    let mut ans = vec![0; q];
    for &(threshold, k, i) in &queries {
        while let Some((_, u)) = rotation_iter.next_if(|&(shift, ..)| shift <= threshold) {
            // rotate childrens of p
            let u_start = nodes[(u as usize) << 1];
            let u_end = nodes[((u as usize) << 1) | 1];
            let c_end = forest.predecessor(u_end).unwrap();
            let c_start = nodes[forest.get(c_end).trie_node_idx as usize ^ 1];

            let lhs = u_start;
            forest.split_right(lhs);
            let rhs = u_end;
            forest.split_left(rhs);
            let mid2 = c_start;
            let mid1 = forest.split_left(mid2);
            let lhs = forest.merge_nonnull(lhs, mid2);
            let rhs = forest.merge(mid1, Some(rhs));
            forest.merge(Some(lhs), rhs);
        }

        // find kth terminal with terminal_count
        let mut k_curr = k;
        forest.splay(root);
        forest.splay_by(&mut root, |forest, u| {
            let count_le = forest.get(u).terminal_count_le(forest);
            if count_le <= k_curr {
                k_curr -= count_le;
                Some(splay::Branch::Right)
            } else {
                Some(splay::Branch::Left)
            }
        });
        if forest.get(root).terminal_count_le(&forest) <= k {
            root = forest.successor(root);
        }

        ans[i as usize] = forest.get(root).terminal_idx as u32;
    }

    for a in ans {
        writeln!(output, "{}", a + 1).unwrap();
    }
}
