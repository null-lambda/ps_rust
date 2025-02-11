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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }

    use std::{fmt::Debug, rc::Rc};

    #[cfg(debug_assertions)]
    #[derive(Clone)]
    pub struct Label(Rc<dyn Debug>);

    #[cfg(not(debug_assertions))]
    #[derive(Clone)]
    pub struct Label;

    impl Label {
        #[inline]
        pub fn new_with<T: Debug + 'static>(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(Rc::new(value()))
            }
            #[cfg(not(debug_assertions))]
            {
                Self
            }
        }
    }

    impl Debug for Label {
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

    impl Default for Label {
        fn default() -> Self {
            Self::new_with(|| ())
        }
    }
}

pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        fmt::{self, Debug},
        mem::MaybeUninit,
        num::NonZeroU32,
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
        children: [Option<NodeRef>; 2],
        parent: Option<NodeRef>,
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

        pub fn with<T>(&mut self, u: NodeRef, f: impl FnOnce(&mut V) -> T) -> T {
            let res = f(unsafe { self.get_mut(u) });
            self.pull_up(u);
            res
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
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
}

pub mod euler_tour_tree {
    use std::collections::HashMap;

    use super::splay;
    // use super::wbtree;

    fn rotate_to_front<S: splay::NodeSpec>(forest: &mut splay::SplayForest<S>, u: splay::NodeRef) {
        forest.splay(u);
        let left = forest.split_left(u);
        forest.merge(Some(u), left);
    }

    pub struct DynamicEulerTour<S: splay::NodeSpec> {
        pub forest: splay::SplayForest<S>,
        pub freed: Vec<splay::NodeRef>,

        pub n_verts: usize,
        pub verts: Vec<splay::NodeRef>,
        pub edges: HashMap<(u32, u32), splay::NodeRef>,
    }

    impl<S: splay::NodeSpec> DynamicEulerTour<S> {
        pub fn new(vert_nodes: impl IntoIterator<Item = S>) -> Self {
            let mut this = Self {
                forest: splay::SplayForest::new(),
                freed: vec![], // Reused deleted edge nodes

                verts: vec![],
                edges: HashMap::new(),
                n_verts: 0,
            };
            for node in vert_nodes {
                let u = this.forest.add_root(node);
                this.verts.push(u);
                this.n_verts += 1;
            }
            this
        }

        pub fn add_root(&mut self, node: S) -> splay::NodeRef {
            if let Some(u) = self.freed.pop() {
                self.forest.with(u, |u| {
                    *u = node;
                });
                u
            } else {
                self.forest.add_root(node)
            }
        }

        pub fn reroot(&mut self, u: usize) {
            rotate_to_front(&mut self.forest, self.edges[&(u as u32, u as u32)]);
        }

        pub fn find_root(&mut self, u: usize) -> splay::NodeRef {
            let mut u = self.verts[u];
            self.forest.splay(u);
            self.forest.splay_first(&mut u);
            u
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            self.find_root(u) == self.find_root(v)
        }

        pub fn contains_edge(&self, u: usize, v: usize) -> bool {
            self.edges.contains_key(&(u as u32, v as u32))
        }

        pub fn link(&mut self, u: usize, v: usize, edge_uv: S, edge_vu: S) -> bool {
            if self.is_connected(u, v) {
                return false;
            }
            let vert_u = self.verts[u];
            let vert_v = self.verts[v];
            let edge_uv = self.add_root(edge_uv);
            let edge_vu = self.add_root(edge_vu);
            self.edges.insert((u as u32, v as u32), edge_uv);
            self.edges.insert((v as u32, u as u32), edge_vu);

            rotate_to_front(&mut self.forest, vert_u);
            rotate_to_front(&mut self.forest, vert_v);
            let lhs = self.forest.merge_nonnull(vert_u, edge_uv);
            let rhs = self.forest.merge_nonnull(vert_v, edge_vu);
            self.forest.merge_nonnull(lhs, rhs);
            true
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            let (Some(edge_uv), Some(edge_vu)) = (
                self.edges.remove(&(u as u32, v as u32)),
                self.edges.remove(&(v as u32, u as u32)),
            ) else {
                return false;
            };

            rotate_to_front(&mut self.forest, edge_uv);
            self.forest.split_right(edge_uv);
            self.forest.split_left(edge_vu);
            self.forest.split_right(edge_vu);
            self.freed.push(edge_uv);
            self.freed.push(edge_vu);
            true
        }
    }
}

pub mod graph_connectivity {
    // Online dynamic connectivity in graphs
    // https://codeforces.com/blog/entry/128556
    use std::collections::{hash_map, HashMap, HashSet};

    use super::debug;
    use super::euler_tour_tree::DynamicEulerTour;
    use super::splay;

    fn ordered_pair(a: u32, b: u32) -> (u32, u32) {
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum NodeState {
        FloatingTreeEdge = 2,
        VertWithBackEdge = 1,
    }

    pub struct Node {
        size: u32,
        edge: (u32, u32),
        state: Option<NodeState>,
        next_node: (Option<NodeState>, (u32, u32)),
        link: splay::Link,
    }

    impl Default for Node {
        fn default() -> Self {
            Self {
                size: 1,
                state: None,
                next_node: (None, (0, 0)),
                edge: (0, 0),
                link: splay::Link::default(),
            }
        }
    }

    impl Node {
        fn new(u: usize, v: usize, state: Option<NodeState>) -> Self {
            let edge = (u as u32, v as u32);
            Self {
                edge,
                state,
                next_node: (state, edge),
                ..Self::default()
            }
        }

        fn edge(u: usize, v: usize) -> Self {
            Self::new(u, v, Some(NodeState::FloatingTreeEdge))
        }

        fn vert(u: usize) -> Self {
            Self::new(u, u, None)
        }
    }

    impl splay::IntrusiveNode for Node {
        fn link(&self) -> &splay::Link {
            &self.link
        }
        fn link_mut(&mut self) -> &mut splay::Link {
            &mut self.link
        }
    }

    impl splay::NodeSpec for Node {
        fn pull_up(&mut self, children: [Option<&mut Self>; 2]) {
            self.size = 1;
            self.next_node = (self.state, self.edge);
            for c in children.iter().flatten() {
                self.size += c.size;
                if self.next_node.0 < c.next_node.0 {
                    self.next_node = c.next_node;
                }
            }
        }
    }

    pub struct Conn {
        pub n_verts: usize,
        pub n_levels: usize,
        pub level: HashMap<(u32, u32), u32>,
        pub spanning_forests: Vec<DynamicEulerTour<Node>>,
        pub adj_back_edges: Vec<HashSet<u32>>,
    }

    impl Conn {
        pub fn new(n_verts: usize) -> Self {
            let n_levels = (u32::BITS - u32::leading_zeros(n_verts as u32)) as usize;
            Self {
                n_verts,
                n_levels,
                level: HashMap::new(),
                spanning_forests: (0..n_levels)
                    .map(|_| DynamicEulerTour::new((0..n_verts).map(|u| Node::vert(u))))
                    .collect(),
                adj_back_edges: vec![HashSet::new(); n_levels * n_verts],
            }
        }

        pub fn contains_edge(&self, u: usize, v: usize) -> bool {
            self.level.contains_key(&ordered_pair(u as u32, v as u32))
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.n_verts && v < self.n_verts);
            self.spanning_forests[0].is_connected(u, v)
        }

        fn link_trees_in_level(&mut self, u: usize, v: usize, level: usize) {
            self.spanning_forests[level].link(u, v, Node::edge(u, v), Node::edge(v, u));
        }

        fn link_with_level(&mut self, u: usize, v: usize, level: usize) -> bool {
            let n_verts = self.n_verts;
            let hash_map::Entry::Vacant(e) = self.level.entry(ordered_pair(u as u32, v as u32))
            else {
                return false;
            };
            e.insert(level as u32);

            if !self.is_connected(u, v) {
                self.link_trees_in_level(u, v, level);
            } else {
                self.adj_back_edges[level * n_verts + u].insert(v as u32);
                self.adj_back_edges[level * n_verts + v].insert(u as u32);
                for s in [u, v] {
                    let hs = self.spanning_forests[level].verts[s];
                    self.spanning_forests[level].forest.splay(hs);
                    self.spanning_forests[level].forest.with(hs, |node| {
                        node.state = Some(NodeState::VertWithBackEdge);
                    });
                }
            }
            true
        }

        pub fn link(&mut self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.n_verts && v < self.n_verts);
            self.link_with_level(u, v, 0)
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.n_verts && v < self.n_verts);
            let Some(base_level) = self.level.remove(&ordered_pair(u as u32, v as u32)) else {
                return false;
            };
            let base_level = base_level as usize;
            if !self.spanning_forests[base_level].contains_edge(u, v) {
                self.adj_back_edges[base_level * self.n_verts + u].remove(&(v as u32));
                self.adj_back_edges[base_level * self.n_verts + v].remove(&(u as u32));
                for s in [u, v] {
                    let degree_s = self.adj_back_edges[base_level * self.n_verts + s].len();
                    if degree_s == 0 {
                        let hs = self.spanning_forests[base_level].verts[s];
                        self.spanning_forests[base_level].forest.splay(hs);
                        self.spanning_forests[base_level]
                            .forest
                            .with(hs, |node| node.state = None);
                    }
                }
            } else {
                for level in (0..=base_level).rev() {
                    self.spanning_forests[level].cut(u, v);
                }

                let mut replacement_edge = None;
                for level in (0..=base_level).rev() {
                    if let Some((s, t)) = replacement_edge {
                        self.link_trees_in_level(s, t, level);
                        continue;
                    }

                    let mut forest = &mut self.spanning_forests[level];
                    let hu = forest.verts[u];
                    let hv = forest.verts[v];
                    forest.forest.splay(hu);
                    forest.forest.splay(hv);
                    let (mut h_small, small) =
                        if forest.forest.get(hu).size <= forest.forest.get(hv).size {
                            (hu, u)
                        } else {
                            (hv, v)
                        };
                    let large = small ^ u ^ v;

                    // Push MSF edges down to the lower level
                    forest.forest.splay(h_small);
                    while let (Some(NodeState::FloatingTreeEdge), (s, t)) =
                        forest.forest.get(h_small).next_node
                    {
                        for (s, t) in [(s, t), (t, s)] {
                            h_small = forest.edges[&(t, s)];
                            forest.forest.splay(h_small);
                            forest.forest.with(h_small, |node| node.state = None);
                        }
                        self.level.insert(ordered_pair(s, t), level as u32 + 1);
                        self.link_trees_in_level(s as usize, t as usize, level + 1);

                        forest = &mut self.spanning_forests[level];
                    }

                    // Find a replacement edge
                    'outer: while let (Some(NodeState::VertWithBackEdge), (s, _)) =
                        forest.forest.get(h_small).next_node
                    {
                        h_small = forest.verts[s as usize];
                        forest.forest.splay(h_small);
                        while let Some(t) = self.adj_back_edges[level * self.n_verts + s as usize]
                            .iter()
                            .next()
                            .copied()
                        {
                            self.level.remove(&ordered_pair(s, t));
                            self.adj_back_edges[level * self.n_verts + s as usize].remove(&t);
                            self.adj_back_edges[level * self.n_verts + t as usize]
                                .remove(&(s as u32));
                            for r in [s, t] {
                                let degree =
                                    self.adj_back_edges[level * self.n_verts + r as usize].len();
                                if degree == 0 {
                                    let hr = self.spanning_forests[level].verts[r as usize];
                                    self.spanning_forests[level].forest.splay(hr);
                                    self.spanning_forests[level]
                                        .forest
                                        .with(hr, |node| node.state = None);
                                }
                            }
                            if self.is_connected(t as usize, large) {
                                replacement_edge = Some((s as usize, t as usize));
                                self.link_with_level(s as usize, t as usize, level);
                                break 'outer;
                            }
                            self.link_with_level(s as usize, t as usize, level + 1);
                        }

                        forest = &mut self.spanning_forests[level];
                        forest.forest.splay(h_small);
                    }
                }
            }

            self.debug_topo();
            true
        }

        pub fn debug_topo(&mut self) {
            debug::with(|| {
                for level in 0..self.n_levels {
                    print!("    Level {level}: ");
                    let mut visited = HashSet::<splay::NodeRef>::new();
                    for u in 0..self.n_verts {
                        let forest = &mut self.spanning_forests[level];
                        if !visited.insert(forest.find_root(u)) {
                            continue;
                        }
                        let hu = forest.verts[u];

                        forest.forest.splay(hu);
                        forest.forest.inorder(hu, &mut |forest, u| {
                            let node = forest.get(u);

                            let (u, v) = node.edge;
                            if u == v {
                                print!("{}", u);
                            } else {
                                print!("{:?}", (u, v));
                            }
                            if node.state.is_some() {
                                print!("*");
                            }
                            print!(" ");
                        });
                        print!("    |    ");
                    }
                    println!();

                    print!("        Back edges: ");
                    for u in 0..self.n_verts {
                        for &v in &self.adj_back_edges[level * self.n_verts + u] {
                            print!("{:?} ", (u, v));
                        }
                    }
                    println!();
                }
                println!();
            });
        }
    }
}

const DECODE_QUERY: bool = true;
// const DECODE_QUERY: bool = false;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut f = 0u64;
    let mut n_components = n as u64;
    let mut conn = graph_connectivity::Conn::new(n);
    for _ in 0..q {
        let a: u64 = input.value();
        let b: u64 = input.value();
        let (x, y) = if DECODE_QUERY {
            ((a ^ f) as usize % n, (b ^ f) as usize % n)
        } else {
            (a as usize, b as usize)
        };

        debug::with(|| {
            if x < y {
                println!("Toggle {x} {y}")
            } else {
                println!("Query {x} {y}")
            }
        });

        if x < y {
            let old = conn.is_connected(x, y);
            assert!(conn.link(x, y) || conn.cut(x, y));
            let new = conn.is_connected(x, y);
            n_components += old as u64;
            n_components -= new as u64;
        } else {
            assert!(x > y);
            writeln!(output, "{}", conn.is_connected(x, y) as u8).unwrap();
        }
        debug::with(|| println!("Components: {}", n_components));

        f += n_components as u64;
    }
}
