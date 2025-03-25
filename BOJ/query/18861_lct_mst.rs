use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

use cmp::Trivial;
use link_cut::NodeRef;

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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod link_cut {
    use std::{
        fmt::{self, Debug},
        num::NonZeroU32,
        ops::{Index, IndexMut},
    };

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

    // Intrusive node link
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
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeRef {
        pub idx: NonZeroU32,
    }

    impl NodeRef {
        fn usize(&self) -> usize {
            self.idx.get() as usize
        }

        pub unsafe fn dangling() -> Self {
            Self {
                idx: NonZeroU32::new(!0).unwrap(),
            }
        }
    }

    impl Debug for NodeRef {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct LinkCutForest<S> {
        pub nodes: Vec<S>,
    }

    impl<S> Index<NodeRef> for LinkCutForest<S> {
        type Output = S;
        fn index(&self, index: NodeRef) -> &Self::Output {
            &self.nodes[index.usize()]
        }
    }

    impl<S> IndexMut<NodeRef> for LinkCutForest<S> {
        fn index_mut(&mut self, index: NodeRef) -> &mut Self::Output {
            &mut self.nodes[index.usize()]
        }
    }

    impl<S: NodeSpec> LinkCutForest<S> {
        pub fn new() -> Self {
            let dummy = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            Self { nodes: vec![dummy] }
        }

        pub fn add_root(&mut self, node: S) -> NodeRef {
            let idx = self.nodes.len();
            self.nodes.push(node);
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut S, [Option<&'a mut S>; 2]) {
            unsafe {
                let pool_ptr = self.nodes.as_mut_ptr();
                let node = &mut *pool_ptr.add(u.usize());
                let children = node
                    .link()
                    .children
                    .map(|child| child.map(|child| &mut *pool_ptr.add(child.usize())));
                (node, children)
            }
        }

        fn reverse(&mut self, u: NodeRef) {
            let link = self[u].link_mut();
            link.inv ^= true;
            link.children.swap(0, 1);
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let link = self[u].link_mut();
                if link.inv {
                    link.inv = false;
                    for c in link.children.into_iter().flatten() {
                        self.reverse(c);
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

        pub fn internal_parent(&self, u: NodeRef) -> Result<(NodeRef, Branch), Option<NodeRef>> {
            match self[u].link().parent {
                Some(p) => {
                    if self[p].link().children[Branch::Left.usize()] == Some(u) {
                        Ok((p, Branch::Left)) // parent on a chain
                    } else if self[p].link().children[Branch::Right.usize()] == Some(u) {
                        Ok((p, Branch::Right)) // parent on a chain
                    } else {
                        Err(Some(p)) // path-parent
                    }
                }
                None => Err(None), // true root
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.internal_parent(u).is_err()
        }

        fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            self[u].link_mut().children[branch as usize] = Some(child);
            self[child].link_mut().parent = Some(u);
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            let child = self[u].link_mut().children[branch as usize].take()?;
            self[child].link_mut().parent = None;
            Some(child)
        }

        fn rotate(&mut self, u: NodeRef) {
            let (p, bp) = self.internal_parent(u).expect("Root shouldn't be rotated");
            let c = self[u].link_mut().children[bp.inv().usize()].replace(p);
            self[p].link_mut().children[bp.usize()] = c;
            if let Some(c) = c {
                self[c].link_mut().parent = Some(p);
            }

            if let Ok((g, bg)) = self.internal_parent(p) {
                self[g].link_mut().children[bg.usize()] = Some(u);
            }

            self[u].link_mut().parent = self[p].link().parent;
            self[p].link_mut().parent = Some(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            while let Ok((p, _)) = self.internal_parent(u) {
                if let Ok((g, _)) = self.internal_parent(p) {
                    self.push_down(g);
                    self.push_down(p);
                    self.push_down(u);

                    let (_, bp) = unsafe { self.internal_parent(u).unwrap_unchecked() };
                    let (_, bg) = unsafe { self.internal_parent(p).unwrap_unchecked() };
                    if bp == bg {
                        self.rotate(p); // zig-zig
                    } else {
                        self.rotate(u); // zig-zag
                    }
                    self.rotate(u);

                    self.pull_up(g);
                    self.pull_up(p);
                    self.pull_up(u);
                } else {
                    self.push_down(p);
                    self.push_down(u);

                    self.rotate(u); // zig

                    self.pull_up(p);
                    self.pull_up(u);
                }
            }
            self.push_down(u);
        }

        pub fn access(&mut self, u: NodeRef) {
            unsafe {
                self.splay(u);
                self[u].link_mut().children[Branch::Right.usize()] = None;
                while let Some(path_parent) = self.internal_parent(u).unwrap_err_unchecked() {
                    self.splay(path_parent);
                    self[path_parent].link_mut().children[Branch::Right.usize()] = Some(u);
                    self.splay(u);
                }
            }
        }

        pub fn reroot(&mut self, u: NodeRef) {
            self.access(u);
            self.reverse(u);
        }

        pub fn link(&mut self, parent: NodeRef, child: NodeRef) {
            self.reroot(child);
            self.link_root(parent, child);
        }

        pub fn link_root(&mut self, parent: NodeRef, child: NodeRef) {
            self.access(child);
            self.access(parent);
            self.attach(child, parent, Branch::Left);
            self.pull_up(child);
        }

        pub fn cut(&mut self, child: NodeRef) {
            self.access(child);
            if self[child].link().children[Branch::Left.usize()].is_some() {
                self.detach(child, Branch::Left);
                self.pull_up(child);
            }
        }

        pub fn find_root(&mut self, mut u: NodeRef) -> NodeRef {
            self.access(u);
            while let Some(left) = self[u].link().children[Branch::Left.usize()] {
                u = left;
                self.push_down(u);
            }
            self.splay(u);
            u
        }

        pub fn is_connected(&mut self, u: NodeRef, v: NodeRef) -> bool {
            self.find_root(u) == self.find_root(v)
        }

        pub fn get_parent(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.access(u);
            let mut left = self[u].link().children[Branch::Left.usize()]?;
            self.push_down(left);
            while let Some(right) = self[left].link().children[Branch::Right.usize()] {
                left = right;
                self.push_down(left);
            }
            self.splay(left);
            Some(left)
        }

        pub fn get_lca(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
            self.access(lhs);
            self.access(rhs);
            self.splay(lhs);
            self[lhs].link().parent.unwrap_or(lhs)
        }

        pub fn access_vertex_path(&mut self, path_top: NodeRef, path_bot: NodeRef) {
            self.reroot(path_top);
            self.access(path_bot);
            self.splay(path_top);
        }
    }
}

struct MinNode {
    value: u32,
    this_cyclic: link_cut::NodeRef,

    min: (u32, Trivial<link_cut::NodeRef>),
    ends: Option<[link_cut::NodeRef; 2]>,

    link: link_cut::Link,
}

impl MinNode {
    fn id() -> Self {
        Self::singleton(u32::MAX, None)
    }

    fn singleton(value: u32, ends: Option<[link_cut::NodeRef; 2]>) -> Self {
        Self {
            value,
            min: (value, Trivial(unsafe { NodeRef::dangling() })),
            this_cyclic: unsafe { NodeRef::dangling() },
            ends,

            link: Default::default(),
        }
    }
}

impl link_cut::IntrusiveNode for MinNode {
    fn link(&self) -> &link_cut::Link {
        &self.link
    }

    fn link_mut(&mut self) -> &mut link_cut::Link {
        &mut self.link
    }
}

impl link_cut::NodeSpec for MinNode {
    fn push_down(&mut self, _children: [Option<&mut Self>; 2]) {}
    fn pull_up(&mut self, children: [Option<&mut Self>; 2]) {
        self.min = (self.value, Trivial(self.this_cyclic));
        for c in children.into_iter().flatten() {
            self.min = self.min.min(c.min);
        }
    }
}

mod cmp {
    // The equalizer of all things
    use std::cmp::Ordering;

    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            // All values are equal, but Some(_)™ are more equal than others...
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut lct = link_cut::LinkCutForest::new();
    let verts: Vec<_> = (0..n).map(|_| lct.add_root(MinNode::id())).collect();

    let mut edges: BinaryHeap<_> = Default::default();
    let mut free = vec![];
    for _ in 0..q {
        match input.token() {
            "1" => {
                let u = verts[input.value::<usize>() - 1];
                let v = verts[input.value::<usize>() - 1];
                let d: u32 = input.value();

                if lct.is_connected(u, v) {
                    lct.reroot(v);
                    lct.access(u);
                    let mut cap = (lct[u].value, Trivial(lct[u].this_cyclic));
                    if let Some(c0) = lct[u].link.children[0] {
                        cap = cap.min(lct[c0].min);
                    }
                    if cap.0 >= d {
                        continue;
                    }

                    let e = cap.1 .0;
                    let [p, q] = lct[e].ends.unwrap();
                    lct.reroot(e);
                    lct.cut(p);
                    lct.cut(q);
                }

                let e = free.pop().unwrap_or_else(|| {
                    let e = lct.add_root(MinNode::singleton(d, Some([u, v])));
                    lct[e].this_cyclic = e;
                    e
                });
                lct.link(e, u);
                lct.link(e, v);
                edges.push((Reverse(d), Trivial((u, v, e))));
            }
            "2" => {
                let cutoff: u32 = input.value();
                while let Some(&(Reverse(d), ..)) = edges.peek() {
                    if d >= cutoff {
                        break;
                    }

                    let (_, Trivial((u, v, e))) = edges.pop().unwrap();
                    if lct.is_connected(e, u) && lct.is_connected(e, v) {
                        lct.reroot(e);
                        lct.cut(u);
                        lct.cut(v);
                    }
                }
            }
            _ => {
                let u = verts[input.value::<usize>() - 1];
                let v = verts[input.value::<usize>() - 1];

                let mut cap = 0;
                if lct.is_connected(u, v) {
                    lct.reroot(v);
                    lct.access(u);
                    cap = lct[u].value;
                    if let Some(c0) = lct[u].link.children[0] {
                        cap = cap.min(lct[c0].min.0);
                    }
                }
                writeln!(output, "{}", cap).unwrap();
            }
        }
    }
}
