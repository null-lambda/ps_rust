use std::{
    cmp::{Ordering, Reverse},
    collections::HashMap,
    io::Write,
};

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
}

pub mod link_cut {
    use std::{
        fmt::{self, Debug},
        num::NonZeroU32,
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

        // type Cx;
        // fn push_down(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
        // fn pull_up(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
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
    pub struct LinkCutForest<S> {
        pub pool: Vec<S>,
    }

    impl<S: NodeSpec> LinkCutForest<S> {
        pub fn new() -> Self {
            let dummy = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            Self { pool: vec![dummy] }
        }

        pub fn add_root(&mut self, node: S) -> NodeRef {
            let idx = self.pool.len();
            self.pool.push(node);
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub fn get_node<'a>(&'a self, u: NodeRef) -> &'a S {
            &self.pool[u.get()]
        }

        pub unsafe fn get_node_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut S {
            &mut self.pool[u.get()]
        }

        pub fn with<T>(&mut self, u: NodeRef, f: impl FnOnce(&mut S) -> T) -> T {
            f(unsafe { self.get_node_mut(u) })
        }

        pub unsafe fn get_node_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut S, [Option<&'a mut S>; 2]) {
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
                let node = self.get_node_mut(u);
                let link = node.link_mut();
                if link.inv {
                    link.inv = false;
                    link.children.swap(0, 1);
                    for child in link.children.into_iter().flatten() {
                        self.get_node_mut(child).link_mut().inv ^= true;
                    }
                }

                let (node, children) = self.get_node_with_children(u);
                node.push_down(children);
            }
        }

        fn pull_up(&mut self, node: NodeRef) {
            unsafe {
                let (node, children) = self.get_node_with_children(node);
                node.pull_up(children);
            }
        }

        pub fn get_internal_parent(
            &self,
            u: NodeRef,
        ) -> Result<(NodeRef, Branch), Option<NodeRef>> {
            match self.get_node(u).link().parent {
                Some(p) => {
                    if self.get_node(p).link().children[Branch::Left.usize()] == Some(u) {
                        Ok((p, Branch::Left)) // parent on a chain
                    } else if self.get_node(p).link().children[Branch::Right.usize()] == Some(u) {
                        Ok((p, Branch::Right)) // parent on a chain
                    } else {
                        Err(Some(p)) // path-parent
                    }
                }
                None => Err(None), // true root
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.get_internal_parent(u).is_err()
        }

        fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_node_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_node_mut(child).link_mut().parent = Some(u);
            }
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            unsafe {
                let child = self.get_node_mut(u).link_mut().children[branch as usize].take()?;
                self.get_node_mut(child).link_mut().parent = None;
                Some(child)
            }
        }

        fn rotate(&mut self, u: NodeRef) {
            let (parent, branch) = self
                .get_internal_parent(u)
                .expect("Root shouldn't be rotated");
            let child = self.detach(u, branch.inv());
            if let Some(child) = child {
                self.attach(parent, child, branch);
            } else {
                self.detach(parent, branch);
            }

            match self.get_internal_parent(parent) {
                Ok((grandparent, grandbranch)) => {
                    self.attach(grandparent, u, grandbranch);
                }
                Err(path_parent) => unsafe {
                    self.get_node_mut(u).link_mut().parent = path_parent;
                },
            }
            self.attach(u, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            while let Ok((parent, branch)) = self.get_internal_parent(u) {
                if let Ok((grandparent, grandbranch)) = self.get_internal_parent(parent) {
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

        pub fn access(&mut self, u: NodeRef) {
            unsafe {
                self.splay(u);
                self.get_node_mut(u).link_mut().children[Branch::Right.usize()] = None;
                while let Some(path_parent) = self.get_internal_parent(u).unwrap_err_unchecked() {
                    self.splay(path_parent);
                    self.get_node_mut(path_parent).link_mut().children[Branch::Right.usize()] =
                        Some(u);
                    self.splay(u);
                }
            }
        }

        pub fn reroot(&mut self, u: NodeRef) {
            self.access(u);
            unsafe { self.get_node_mut(u) }.link_mut().inv ^= true;
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
            if self.get_node(child).link().children[Branch::Left.usize()].is_some() {
                self.detach(child, Branch::Left);
                self.pull_up(child);
            }
        }

        pub fn find_root(&mut self, mut u: NodeRef) -> NodeRef {
            self.access(u);
            while let Some(left) = self.get_node(u).link().children[Branch::Left.usize()] {
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
            let mut left = self.get_node(u).link().children[Branch::Left.usize()]?;
            self.push_down(left);
            while let Some(right) = self.get_node(left).link().children[Branch::Right.usize()] {
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
            self.get_node(lhs).link().parent.unwrap_or(lhs)
        }

        pub fn access_vertex_path(&mut self, path_top: NodeRef, path_bot: NodeRef) {
            self.reroot(path_top);
            self.access(path_bot);
            self.splay(path_top);
        }
    }
}
const INF: i32 = 1 << 29;

#[derive(Debug)]
struct MinPlusNode {
    value: i32,
    min: i32,
    count: u32,
    lazy_add: i32,

    link: link_cut::Link,
}

impl Default for MinPlusNode {
    fn default() -> Self {
        Self {
            value: 0,
            min: 0,
            count: 1,
            lazy_add: 0,

            link: link_cut::Link::default(),
        }
    }
}

impl MinPlusNode {
    fn edge() -> Self {
        Self::default()
    }

    fn vert() -> Self {
        Self {
            value: INF,
            ..Default::default()
        }
    }

    fn add(&mut self, delta: i32) {
        self.value += delta;
        self.min += delta;
        self.lazy_add += delta;
    }

    fn count_zero(&self) -> u32 {
        if self.min == 0 {
            self.count
        } else {
            0
        }
    }
}

impl link_cut::IntrusiveNode for MinPlusNode {
    fn link(&self) -> &link_cut::Link {
        &self.link
    }
    fn link_mut(&mut self) -> &mut link_cut::Link {
        &mut self.link
    }
}

impl link_cut::NodeSpec for MinPlusNode {
    fn push_down(&mut self, children: [Option<&mut Self>; 2]) {
        for c in children.into_iter().flatten() {
            c.add(self.lazy_add);
        }
        self.lazy_add = 0;
    }

    fn pull_up(&mut self, children: [Option<&mut Self>; 2]) {
        fn combine_mincount(lhs: (i32, u32), rhs: (i32, u32)) -> (i32, u32) {
            match lhs.0.cmp(&rhs.0) {
                Ordering::Less => lhs,
                Ordering::Greater => rhs,
                Ordering::Equal => (lhs.0, lhs.1 + rhs.1),
            }
        }

        let mut min_count = (self.value, 1);
        for c in children.into_iter().flatten() {
            min_count = combine_mincount(min_count, (c.min, c.count));
        }
        self.min = min_count.0;
        self.count = min_count.1;
    }
}

fn sorted2<T: Ord>(mut xs: [T; 2]) -> [T; 2] {
    if xs[0] > xs[1] {
        xs.swap(0, 1);
    }
    xs
}

enum LinkType {
    TreeEdge,
    BackEdge,
}

struct Dynamic2CC {
    msf: link_cut::LinkCutForest<MinPlusNode>, // Minimum spanning forest
    verts: Vec<link_cut::NodeRef>,
    edges: HashMap<[u32; 2], link_cut::NodeRef>,
    freed_edges: Vec<link_cut::NodeRef>,

    n_cut_edges: u32,

    history: Vec<LinkType>,
}

impl Dynamic2CC {
    fn new(n: u32) -> Self {
        let mut msf = link_cut::LinkCutForest::new();
        let verts = (0..n).map(|_| msf.add_root(MinPlusNode::vert())).collect();
        Self {
            msf,
            verts,
            edges: HashMap::new(),
            freed_edges: vec![],

            n_cut_edges: 0,

            history: vec![],
        }
    }

    fn link(&mut self, u: u32, v: u32) {
        let vert_u = self.verts[u as usize];
        let vert_v = self.verts[v as usize];
        if !self.msf.is_connected(vert_u, vert_v) {
            let e_uv = self
                .freed_edges
                .pop()
                .unwrap_or_else(|| self.msf.add_root(MinPlusNode::edge()));
            self.msf.link_root(vert_u, e_uv);
            self.msf.link(vert_v, e_uv);

            self.edges.insert(sorted2([u, v]), e_uv);

            self.n_cut_edges += 1;
            self.history.push(LinkType::TreeEdge);
            debug::with(|| println!("link {} {}, {}", u, v, self.n_cut_edges));
        } else {
            self.msf.access_vertex_path(vert_u, vert_v);
            self.msf.with(vert_u, |path| {
                self.n_cut_edges -= path.count_zero();
                path.add(1);
            });

            self.history.push(LinkType::BackEdge);
        }
    }

    unsafe fn pop_unchecked(&mut self, u: u32, v: u32) {
        let vert_u = self.verts[u as usize];
        let vert_v = self.verts[v as usize];
        match self.history.pop().unwrap_unchecked() {
            LinkType::TreeEdge => {
                debug::with(|| println!("pop {} {}, {}", u, v, self.n_cut_edges));
                self.n_cut_edges -= 1;

                let e_uv = self.edges.remove(&sorted2([u, v])).unwrap_unchecked();

                self.msf.reroot(e_uv);
                self.msf.cut(vert_u);
                self.msf.cut(vert_v);

                self.freed_edges.push(e_uv);
            }
            LinkType::BackEdge => {
                self.msf.access_vertex_path(vert_u, vert_v);
                self.msf.with(vert_u, |path| {
                    path.add(-1);
                    self.n_cut_edges += path.count_zero();
                });
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let t: usize = input.value();

    let bs: Vec<u32> = (0..t + 1).map(|_| input.value::<u32>() - 1).collect();
    let groups: Vec<Vec<[u32; 2]>> = (0..m)
        .map(|_| {
            let k: usize = input.value();
            let mut edges: Vec<_> = (0..k)
                .map(|_| sorted2([input.value::<u32>() - 1, input.value::<u32>() - 1]))
                .collect();
            edges.sort_unstable();
            edges
        })
        .collect();

    let mut queries: Vec<_> = bs
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let mut w = [w[0], w[1]];
            if groups[w[0] as usize].len() < groups[w[1] as usize].len() {
                w.swap(0, 1);
            }
            let s = groups[w[0] as usize].len() as u32;
            (w, s, i as u32)
        })
        .collect();
    queries.sort_unstable_by_key(|&(w, s, _)| (Reverse(s), w));

    // debug::with(|| println!("{:?}", queries));

    let mut cx = Dynamic2CC::new(n as u32);
    let mut ans = vec![0; t];
    let (mut g_prev, mut h_prev) = (!0, !0);
    for ([g, h], _, i) in queries {
        let pop_g = g != g_prev;
        let pop_h = pop_g || h != h_prev;

        if pop_h && h_prev != !0 {
            for &[u, v] in groups[h_prev as usize].iter().rev() {
                unsafe { cx.pop_unchecked(u, v) };
            }
        }
        if pop_g && g_prev != !0 {
            for &[u, v] in groups[g_prev as usize].iter().rev() {
                unsafe { cx.pop_unchecked(u, v) };
            }
        }

        // cx = Dynamic2CC::new(n as u32);
        // let pop_g = true;
        // let pop_h = true;
        // debug::with(|| {
        //     println!("{:?}", groups[g as usize]);
        //     println!("{:?}", groups[h as usize]);
        // });

        if pop_g {
            for &[u, v] in &groups[g as usize] {
                cx.link(u, v);
            }
        }
        if pop_h {
            for &[u, v] in &groups[h as usize] {
                cx.link(u, v);
            }
        }

        ans[i as usize] = cx.n_cut_edges;

        g_prev = g;
        h_prev = h;

        debug::with(|| println!("end"));
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
