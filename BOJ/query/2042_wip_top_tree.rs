use std::io::Write;

use top_tree::{ClusterCx, InternalSplay};

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
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }

    #[cfg(debug_assertions)]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(T);

    #[cfg(not(debug_assertions))]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(std::marker::PhantomData<T>);

    impl<T> Label<T> {
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

        pub fn with(&mut self, #[allow(unused_variables)] f: impl FnOnce(&mut T)) {
            #[cfg(debug_assertions)]
            f(&mut self.0)
        }

        pub fn get_mut(&mut self) -> Option<&mut T> {
            #[cfg(debug_assertions)]
            {
                Some(&mut self.0)
            }

            #[cfg(not(debug_assertions))]
            {
                None
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

    pub mod tree {
        #[derive(Clone)]
        pub struct Pretty(pub String, pub Vec<Pretty>);

        impl Pretty {
            fn fmt_rec(
                &self,
                f: &mut std::fmt::Formatter,
                prefix: &str,
                first: bool,
                last: bool,
            ) -> std::fmt::Result {
                let space = format!("{}   ", prefix);
                let bar = format!("{}|  ", prefix);
                let sep = if first && last {
                    "*--"
                } else if first {
                    "┌--"
                } else if last {
                    "└--"
                } else {
                    "+--"
                };

                let m = self.1.len();
                for i in 0..m / 2 {
                    let c = &self.1[i];
                    let prefix_ext = if first && i == 0 { &space } else { &bar };
                    c.fmt_rec(f, &prefix_ext, i == 0, i == m - 1)?;
                }

                writeln!(f, "{}{}{}", prefix, sep, self.0)?;

                for i in m / 2..m {
                    let c = &self.1[i];
                    let prefix_ext = if last && i == m - 1 { &space } else { &bar };
                    c.fmt_rec(f, &prefix_ext, i == 0, i == m - 1)?;
                }

                Ok(())
            }
        }

        impl<'a> std::fmt::Debug for Pretty {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                writeln!(f)?;
                self.fmt_rec(f, "", true, true)
            }
        }
    }
}

pub mod top_tree {
    // # Top Tree
    //
    // Circular order between rake edges does not preserved.
    //
    // ## Checklist on implementation
    // - Parent(children(u)[0]) = parent(children(u)[1]) = u
    // - Push_down before pull_up
    // - Handles should be updated immediately

    use std::{marker::PhantomData, num::NonZeroU32};

    use node::BinaryNode;

    pub const UNSET: u32 = !0;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Either<L, R> {
        Left(L),
        Right(R),
    }

    pub trait ClusterCx: Sized {
        // Vertex weight
        type V: Default;

        // Path cluster (aggregate on a subchain), as an **open** interval.
        type C;
        // Point cluster (Aggregate of light edges), as an **left-open, right-closed** interval.
        type R;

        /// Compress monoid.
        /// Left side is always the top side.
        fn id_compress() -> Self::C;

        fn compress(&self, children: [&Self::C; 2], v: &Self::V, rake: Option<&Self::R>)
            -> Self::C;

        /// Rake monoid, commutative.
        fn id_rake() -> Self::R;

        fn rake(&self, lhs: &Self::R, rhs: &Self::R) -> Self::R;

        /// Enclose the right end of a path cluster with a vertex.
        fn collapse_path(&self, c: &Self::C, vr: &Self::V) -> Self::R;

        /// Lazy propagation (implement it yourself)
        #[allow(unused_variables)]
        fn push_down_compress(
            &self,
            node: &mut Self::C,
            children: [&mut Self::C; 2],
            v: &mut Self::V,
            rake: Option<&mut Self::R>,
        ) {
        }

        #[allow(unused_variables)]
        fn push_down_rake(
            &self,
            node: &mut Self::R,
            children: Option<[&mut Self::R; 2]>,
            c: &mut Self::C,
            vr: &mut Self::V,
        ) {
        }

        fn reverse(&self, c: &Self::C) -> Self::C;
    }

    #[derive(Debug, Copy, Clone)]
    pub enum ActionRange {
        Subtree,
        Path,
    }

    /// Lazy propagation (Implement it yourself, Part II)
    pub trait Action<Cx: ClusterCx> {
        fn apply_to_compress(&mut self, compress: &mut Cx::C, range: ActionRange);
        fn apply_to_rake(&mut self, rake: &mut Cx::R);
        fn apply_to_weight(&mut self, weight: &mut Cx::V);
    }

    pub struct NodeRef<T> {
        idx: NonZeroU32,
        _phantom: std::marker::PhantomData<*mut T>,
    }

    #[derive(Debug, Clone)]
    pub struct Pool<T> {
        pub nodes: Vec<T>,
        pub free: Vec<NodeRef<T>>,
    }

    impl<T> NodeRef<T> {
        pub fn new(idx: u32) -> Self {
            Self {
                idx: NonZeroU32::new(idx).unwrap(),
                _phantom: Default::default(),
            }
        }

        pub unsafe fn dangling() -> Self {
            Self {
                idx: NonZeroU32::new(UNSET).unwrap(),
                _phantom: PhantomData,
            }
        }

        pub fn get_in<'a>(&self, pool: &'a Pool<T>) -> &'a T {
            &pool.nodes[self.idx.get() as usize]
        }

        pub fn mut_in<'a>(&self, pool: &'a mut Pool<T>) -> &'a mut T {
            &mut pool.nodes[self.idx.get() as usize]
        }

        pub unsafe fn get_many_in<'a, const N: usize>(
            indices: [Self; N],
            pool: &'a mut Pool<T>,
        ) -> [&'a mut T; N] {
            let ptr = pool.nodes.as_mut_ptr();
            indices.map(|i| &mut *ptr.add(i.idx.get() as usize))
        }
    }

    impl<T> Clone for NodeRef<T> {
        fn clone(&self) -> Self {
            Self {
                idx: self.idx,
                _phantom: Default::default(),
            }
        }
    }

    impl<T> Copy for NodeRef<T> {}

    impl<T> PartialEq for NodeRef<T> {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx
        }
    }

    impl<T> Eq for NodeRef<T> {}

    impl<T> std::fmt::Debug for NodeRef<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    pub mod node {
        use crate::debug;

        use super::*;

        #[derive(Debug)]
        pub enum Parent<Cx: ClusterCx> {
            Compress(NodeRef<Compress<Cx>>),
            Rake(NodeRef<Rake<Cx>>),
        }

        impl<Cx: ClusterCx> Clone for Parent<Cx> {
            fn clone(&self) -> Self {
                match self {
                    Parent::Compress(c) => Parent::Compress(*c),
                    Parent::Rake(r) => Parent::Rake(*r),
                }
            }
        }

        impl<Cx: ClusterCx> Copy for Parent<Cx> {}

        #[derive(Debug)]
        pub struct CompressPivot<Cx: ClusterCx> {
            pub vert: u32,
            pub children: [NodeRef<Compress<Cx>>; 2],
            pub rake_tree: Option<NodeRef<Rake<Cx>>>,
        }

        #[derive(Debug)]
        pub struct Compress<Cx: ClusterCx> {
            pub ends: [u32; 2],
            pub inv_lazy: bool,

            pub parent: Option<Parent<Cx>>,
            pub pivot: Option<CompressPivot<Cx>>,

            pub sum: Cx::C,

            pub debug_inorder: debug::Label<debug::tree::Pretty>,
        }

        #[derive(Debug)]
        pub struct Rake<Cx: ClusterCx> {
            pub parent: Parent<Cx>,
            pub children: Option<[NodeRef<Rake<Cx>>; 2]>,
            pub compress_tree: NodeRef<Compress<Cx>>,

            pub sum: Cx::R,

            pub debug_inorder: debug::Label<debug::tree::Pretty>,
        }

        pub trait BinaryNode: Sized {
            type Parent: Copy;

            unsafe fn uninit() -> Self;

            fn internal_parent(&self) -> Option<&NodeRef<Self>>;
            fn parent_mut(&mut self) -> &mut Self::Parent;
            fn is_internal_root(&self) -> bool {
                self.internal_parent().is_none()
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]>;
            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]>;

            fn debug_inorder(&mut self) -> &mut debug::Label<debug::tree::Pretty>;
        }

        impl<Cx: ClusterCx> BinaryNode for Compress<Cx> {
            type Parent = Option<Parent<Cx>>;

            unsafe fn uninit() -> Self {
                Self {
                    ends: [UNSET, UNSET],
                    inv_lazy: false,

                    parent: None,
                    pivot: None,

                    sum: Cx::id_compress(),

                    debug_inorder: debug::Label::new_with(|| {
                        debug::tree::Pretty("uninit".into(), vec![])
                    }),
                }
            }

            fn internal_parent(&self) -> Option<&NodeRef<Self>> {
                match &self.parent {
                    Some(Parent::Compress(c)) => Some(c),
                    _ => None,
                }
            }

            fn parent_mut(&mut self) -> &mut Self::Parent {
                &mut self.parent
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]> {
                Some(&self.pivot.as_ref()?.children)
            }

            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]> {
                Some(&mut self.pivot.as_mut()?.children)
            }

            fn debug_inorder(&mut self) -> &mut debug::Label<debug::tree::Pretty> {
                &mut self.debug_inorder
            }
        }

        impl<Cx: ClusterCx> Compress<Cx> {
            pub fn reverse(&mut self, cx: &Cx) {
                self.inv_lazy ^= true;
                cx.reverse(&mut self.sum);
                self.ends.swap(0, 1);
            }
        }

        impl<Cx: ClusterCx> BinaryNode for Rake<Cx> {
            type Parent = Parent<Cx>;

            unsafe fn uninit() -> Self {
                Self {
                    parent: Parent::Compress(NodeRef::dangling()),
                    children: None,
                    compress_tree: NodeRef::dangling(),
                    sum: Cx::id_rake(),

                    debug_inorder: debug::Label::new_with(|| {
                        debug::tree::Pretty("uninit".into(), vec![])
                    }),
                }
            }

            fn internal_parent(&self) -> Option<&NodeRef<Self>> {
                match &self.parent {
                    Parent::Rake(r) => Some(r),
                    _ => None,
                }
            }

            fn parent_mut(&mut self) -> &mut Self::Parent {
                &mut self.parent
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]> {
                self.children.as_ref()
            }

            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]> {
                self.children.as_mut()
            }

            fn debug_inorder(&mut self) -> &mut debug::Label<debug::tree::Pretty> {
                &mut self.debug_inorder
            }
        }
    }

    impl<T: BinaryNode> Default for Pool<T> {
        fn default() -> Self {
            Self {
                nodes: vec![unsafe { node::BinaryNode::uninit() }],
                free: vec![],
            }
        }
    }

    impl<T: BinaryNode> Pool<T> {
        fn alloc(&mut self, node: T) -> NodeRef<T> {
            let u = if let Some(u) = self.free.pop() {
                *u.mut_in(self) = node;
                u
            } else {
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                NodeRef::new(idx)
            };

            u.mut_in(self)
                .debug_inorder()
                .with(|p| p.0 = format!("{u:?}"));

            u
        }

        unsafe fn free(&mut self, u: NodeRef<T>) {
            self.free.push(u);
        }
    }

    pub struct TopTree<Cx: ClusterCx> {
        pub cx: Cx,

        // Compress tree
        pub cs: Pool<node::Compress<Cx>>,
        // Rake tree
        pub rs: Pool<node::Rake<Cx>>,

        // Vertex info
        pub n_verts: usize,
        pub weights: Vec<Cx::V>,

        // handle(v) is the only compress-node that requires the vertex information of v. There are three cases:
        // - degree(v) = 0 (Isolated vertex) => handle(v) = null
        // - degree(v) = 1 (Boundary vertex) => handle(v) = [Root of a compression tree]
        // - degree(v) >= 2 (Inner vertex) => handle(v) = [Compress-node with v as the pivot]
        pub handle: Vec<Option<NodeRef<node::Compress<Cx>>>>,
    }

    pub trait InternalSplay<T: BinaryNode> {
        fn pool(&mut self) -> &mut Pool<T>;

        fn push_down(&mut self, u: NodeRef<T>);
        fn pull_up(&mut self, u: NodeRef<T>);

        fn branch(&mut self, u: NodeRef<T>) -> Option<(NodeRef<T>, usize)> {
            let pool = self.pool();
            let &p = u.get_in(pool).internal_parent()?;
            let branch = unsafe { p.get_in(pool).children().unwrap_unchecked() }[1] == u;
            Some((p, branch as usize))
        }

        unsafe fn rotate(&mut self, u: NodeRef<T>) {
            // Converts p ->(L) u ->(R) c to u ->(R) p ->(L) c.
            // (If p ->(L) u, otherwise flip (L) <-> (R).)
            //
            // Constraints:
            // 1. u must be a non-root branch.
            // 2. push_down and pull_up for (g?), p, and u must be called beforehand.
            // Diagram:
            // ┌────────┐     ┌───────┐
            // │      g?│     │    g? │
            // │     /  │     │   /   │
            // │    p   │     │  u    │
            // │   / \  │ ==> │ / \   │
            // │  u   4 │     │0   p  │
            // │ / \    │     │   / \ │
            // │0   c   │     │  c   4│
            // └────────┘     └───────┘

            let (p, bp) = self.branch(u).unwrap_unchecked();
            let c = std::mem::replace(
                &mut u.mut_in(self.pool()).children_mut().unwrap_unchecked()[bp ^ 1],
                p,
            );
            p.mut_in(self.pool()).children_mut().unwrap_unchecked()[bp] = c;

            if let Some((g, bg)) = self.branch(p) {
                g.mut_in(self.pool()).children_mut().unwrap_unchecked()[bg as usize] = u;
            }
            let pp = *p.mut_in(self.pool()).parent_mut();
            *p.mut_in(self.pool()).parent_mut() = *c.mut_in(self.pool()).parent_mut();
            *c.mut_in(self.pool()).parent_mut() = *u.mut_in(self.pool()).parent_mut();
            *u.mut_in(self.pool()).parent_mut() = pp;
        }

        unsafe fn guarded_splay(&mut self, u: NodeRef<T>, guard: Option<NodeRef<T>>) {
            // Make u the root of the splay tree.
            // Diagram:
            //
            // Case zig-zig:
            // ┌─────────┐     ┌─────────┐
            // │      g  │     │  u      │
            // │     / \ │     │ / \     │
            // │    p   6│     │0   p    │
            // │   / \   │ ==> │   / \   │
            // │  u   4  │     │  2   g  │
            // │ / \     │     │     / \ │
            // │0   2    │     │    4   6│
            // └─────────┘     └─────────┘
            //
            // Case zig-zag:
            // ┌───────┐
            // │    g  │     ┌───────────┐
            // │   / \ │     │     u     │
            // │  p   6│     │   /   \   │
            // │ / \   │ ==> │  p     g  │
            // │0   u  │     │ / \   / \ │
            // │   / \ │     │0   2 4   6│
            // │   2  4│     └───────────┘
            // └───────┘
            //
            // Case zig:
            // ┌────────┐     ┌───────┐
            // │    p   │     │  u    │
            // │   / \  │     │ / \   │
            // │  u   4 │ ==> │0   p  │
            // │ / \    │     │   / \ │
            // │0   c   │     │  c   4│
            // └────────┘     └───────┘

            unsafe {
                while let Some((p, bp)) = self.branch(u).filter(|&(p, _)| Some(p) != guard) {
                    if let Some((g, bg)) = self.branch(p).filter(|&(g, _)| Some(g) != guard) {
                        self.push_down(g);
                        self.push_down(p);
                        self.push_down(u);

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
            }
        }

        fn splay(&mut self, u: NodeRef<T>) {
            unsafe { self.guarded_splay(u, None) };
        }
    }

    impl<Cx: ClusterCx> InternalSplay<node::Compress<Cx>> for TopTree<Cx> {
        fn pool(&mut self) -> &mut Pool<node::Compress<Cx>> {
            &mut self.cs
        }

        fn push_down(&mut self, u: NodeRef<node::Compress<Cx>>) {
            unsafe {
                let u_mut = u.mut_in(&mut self.cs);
                if let Some(pivot) = &mut u_mut.pivot {
                    let inv_lazy = std::mem::take(&mut u_mut.inv_lazy);

                    let vert = pivot.vert;
                    let rake_tree = pivot.rake_tree.map(|r| &mut r.mut_in(&mut self.rs).sum);

                    if inv_lazy {
                        pivot.children.swap(0, 1);
                    }

                    let [u, l, r] = NodeRef::get_many_in(
                        [u, pivot.children[0], pivot.children[1]],
                        &mut self.cs,
                    );

                    if inv_lazy {
                        l.reverse(&self.cx);
                        r.reverse(&self.cx);
                    }

                    self.cx.push_down_compress(
                        &mut u.sum,
                        [&mut l.sum, &mut r.sum],
                        &mut self.weights[vert as usize],
                        rake_tree,
                    );
                }
            }
        }

        fn pull_up(&mut self, u: NodeRef<node::Compress<Cx>>) {
            unsafe {
                {
                    let u_mut = u.mut_in(&mut self.cs);
                    let ends = u_mut.ends;
                    u_mut
                        .debug_inorder()
                        .get_mut()
                        .map(|p| p.0 = format!("{u:?}- ends:{:?}", ends));
                }

                if let Some(pivot) = &u.get_in(&self.cs).pivot {
                    let rake_tree = pivot.rake_tree;
                    let rake_tree_sum = rake_tree.map(|r| &r.get_in(&self.rs).sum);

                    let [u_mut, l, r] = NodeRef::get_many_in(
                        [u, pivot.children[0], pivot.children[1]],
                        &mut self.cs,
                    );

                    debug_assert_eq!(l.ends[1], r.ends[0], "u={u:?} {:?} {:?}", l.ends, r.ends);
                    let vert = l.ends[1];
                    u_mut.ends = [l.ends[0], r.ends[1]];
                    u_mut.pivot.as_mut().unwrap_unchecked().vert = vert;
                    self.handle[vert as usize] = Some(u);

                    u_mut.sum = self.cx.compress(
                        [&l.sum, &r.sum],
                        &mut self.weights[vert as usize],
                        rake_tree_sum,
                    );

                    {
                        let ends = u_mut.ends;
                        u_mut
                            .debug_inorder()
                            .get_mut()
                            .zip(l.debug_inorder().get_mut())
                            .zip(r.debug_inorder().get_mut())
                            .map(|((p, l), r)| {
                                p.0 = format!("{u:?}- ends:{:?}", ends);
                                if let Some(rake_tree) = rake_tree {
                                    p.0 = format!("{u:?}-(r{rake_tree:?})");
                                }
                                p.1 = vec![l.clone(), r.clone()];
                            });
                    }
                }

                let u_mut = u.mut_in(&mut self.cs);
                if u_mut.internal_parent().is_none() {
                    self.handle[u_mut.ends[0] as usize] = Some(u);
                    self.handle[u_mut.ends[1] as usize] = Some(u);
                }
            }
        }
    }

    impl<Cx: ClusterCx> InternalSplay<node::Rake<Cx>> for TopTree<Cx> {
        fn pool(&mut self) -> &mut Pool<node::Rake<Cx>> {
            &mut self.rs
        }

        fn push_down(&mut self, u: NodeRef<node::Rake<Cx>>) {
            unsafe {
                let u_mut = u.mut_in(&mut self.rs);
                let (u_mut, children) = if let Some(cs) = u_mut.children {
                    let [u_mut_new, l, r] = NodeRef::get_many_in([u, cs[0], cs[1]], &mut self.rs);
                    (u_mut_new, Some([l, r].map(|c| &mut c.sum)))
                } else {
                    (u_mut, None)
                };

                let compress_tree = &mut u_mut.compress_tree.mut_in(&mut self.cs);
                self.cx.push_down_rake(
                    &mut u_mut.sum,
                    children,
                    &mut compress_tree.sum,
                    &mut self.weights[compress_tree.ends[1] as usize],
                );
            }
        }

        fn pull_up(&mut self, u: NodeRef<node::Rake<Cx>>) {
            unsafe {
                let u_mut = u.mut_in(&mut self.rs);
                let compress_tree = u_mut.compress_tree.mut_in(&mut self.cs);
                u_mut.sum = self.cx.collapse_path(
                    &compress_tree.sum,
                    &self.weights[compress_tree.ends[1] as usize],
                );

                if let Some(cs) = u_mut.children {
                    let [u_mut, l, r] = NodeRef::get_many_in([u, cs[0], cs[1]], &mut self.rs);
                    u_mut.sum = self.cx.rake(&u_mut.sum, &self.cx.rake(&l.sum, &r.sum));
                }
            }
        }
    }

    impl<Cx: ClusterCx> TopTree<Cx> {
        pub fn new(weights: impl IntoIterator<Item = Cx::V>, cx: Cx) -> Self {
            let weights: Vec<_> = weights.into_iter().collect();
            let n_verts = weights.len();
            Self {
                cx,

                cs: Pool::default(),
                rs: Pool::default(),

                n_verts,
                weights,
                handle: vec![None; n_verts],
            }
        }

        // Connect all chains between root the and u, and splay handle(u).
        pub fn access(&mut self, u: usize) -> NodeRef<node::Compress<Cx>> {
            todo!()
        }

        pub fn soft_expose(&mut self, u: usize, v: usize) -> Option<NodeRef<node::Compress<Cx>>> {
            todo!()
        }

        pub fn reroot(&mut self, u: usize) {
            todo!()
        }

        pub fn link(&mut self, u: usize, v: usize, e: Cx::C) -> bool {
            todo!()
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            todo!()
        }

        pub fn sum_path(&mut self, u: usize, v: usize) -> (&Cx::V, &Cx::C, &Cx::V) {
            todo!()
        }

        // pub fn sum_subtree(&mut self, u: usize, v: usize) -> (&Cx::V, &Cx::R) {
        //     todo!()
        // }

        // pub fn apply_path(&mut self, u: usize, v: usize, action: impl Action<Cx>) -> bool {
        //     todo!()
        // }

        // pub fn apply_subtree(&mut self, u: usize, p: usize, action: impl Action<Cx>) -> bool {
        //     todo!()
        // }

        pub fn debug_cons_chain(
            &mut self,
            nodes: impl IntoIterator<Item = Cx::C>,
        ) -> Option<NodeRef<node::Compress<Cx>>> {
            let mut nodes = nodes.into_iter().enumerate();

            let (u, c) = nodes.next()?;
            let mut lhs = self.cs.alloc(node::Compress {
                ends: [u as u32, u as u32 + 1],
                sum: c,
                ..unsafe { node::Compress::uninit() }
            });

            for (u, c) in nodes {
                let rhs = self.cs.alloc(node::Compress {
                    ends: [u as u32, u as u32 + 1],
                    sum: c,
                    ..unsafe { node::Compress::uninit() }
                });

                let c = self.cs.alloc(node::Compress {
                    pivot: Some(node::CompressPivot {
                        vert: UNSET,
                        children: [lhs, rhs],
                        rake_tree: None,
                    }),
                    ..unsafe { node::Compress::uninit() }
                });

                lhs.mut_in(&mut self.cs).parent = Some(node::Parent::Compress(c));
                rhs.mut_in(&mut self.cs).parent = Some(node::Parent::Compress(c));
                self.pull_up(c);
                lhs = c;
            }

            Some(lhs)
        }
    }

    ///// A morphism of clusters, with common weight type V. There are two main use cases:
    ///// - Supporting both path and subtree sum queries.
    /////   Path sums do not propagate from rake trees, whereas subtree sums do.
    /////   Therefore, we need to store both path and subtree aggregates in your clusters,
    /////   and the projection helps reduce computation time efficiently for each sum query.
    ///// - Nesting another data structure within nodes (e.g., sets, segment trees, ropes, ... ).
    /////   The user has control over querying a specific node before performing the summation.
    ///// Some set of combinators are provided: identity, tuple and path-sum.
    /////
    ///// TODO: modify most of the sum_ functions to accept an additional reducer.
    //pub trait Reducer<Dom: ClusterCx> {
    //    type Co: ClusterCx<V = Dom::V>;
    //    fn co(&self) -> &Self::Co;
    //    fn map_compress(
    //        &self,
    //        c: &<Dom as ClusterCx>::Compress,
    //    ) -> <Self::Co as ClusterCx>::Compress;
    //    fn map_rake(&self, r: &<Dom as ClusterCx>::Rake) -> <Self::Co as ClusterCx>::Rake;
    //}

    ///// An identity.
    /////
    ///// # Examples
    /////
    ///// ```
    ///// let cx = || { ... }
    ///// let mut stt = StaticTopTree::from_edges(n, edges, root, cx());
    ///// ...
    ///// let total_sum = stt.sum_all(Id(cx()));
    ///// ```
    //pub struct Id<Cx>(pub Cx);
    //impl<Cx: ClusterCx> Reducer<Cx> for Id<Cx> {
    //    type Co = Cx;
    //    fn co(&self) -> &Self::Co {
    //        &self.0
    //    }
    //    fn map_compress(&self, c: &<Cx as ClusterCx>::Compress) -> <Cx as ClusterCx>::Compress {
    //        c.clone()
    //    }
    //    fn map_rake(&self, r: &<Cx as ClusterCx>::Rake) -> <Cx as ClusterCx>::Rake {
    //        r.clone()
    //    }
    //}
}

#[derive(Debug)]
struct Additive;

impl ClusterCx for Additive {
    type V = i64;

    type C = i64;
    type R = ();

    fn id_compress() -> Self::C {
        0
    }
    fn compress(&self, children: [&Self::C; 2], v: &Self::V, _: Option<&Self::R>) -> Self::C {
        children[0] + v + children[1]
    }

    fn id_rake() -> Self::R {}
    fn rake(&self, _: &Self::R, _: &Self::R) -> Self::R {}

    fn collapse_path(&self, _: &Self::C, _: &Self::V) -> Self::R {}
    fn reverse(&self, c: &Self::C) -> Self::C {
        *c
    }
}

#[test]
fn test_linear() {
    let n = 9;

    let mut tt = top_tree::TopTree::new((0..n).map(|u| u as i64), Additive);
    tt.debug_cons_chain((0..n - 1).map(|_| 0)).unwrap();

    for r0 in 0..n {
        use top_tree::InternalSplay;
        let hr0 = tt.handle[r0].unwrap();
        tt.splay(hr0);

        println!(
            "handle({r0}) = {hr0:?}, splay[handle({r0})]: {:?}",
            hr0.get_in(&tt.cs)
        );
    }

    // println!("{:?}", tt.weights);
    // for (u, c) in tt.cs.nodes.iter().enumerate().skip(1) {
    //     println!("cs[{}] = {:?}", u, c);
    // }

    // for (u, h) in tt.handle.iter().enumerate() {
    //     println!("handle[v{}] = {:?}", u, h);
    // }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();

    let mut tt = top_tree::TopTree::new((0..n).map(|_| input.value::<i64>()), Additive);
    tt.debug_cons_chain((0..n - 1).map(|_| 0));
    for _ in 0..m + k {
        match input.token() {
            "1" => {
                let b = input.u32() as usize - 1;
                let c: i64 = input.value();

                if let Some(hb) = tt.handle[b] {
                    tt.splay(hb);
                    tt.weights[b] = c;
                    tt.push_down(hb);
                    tt.pull_up(hb);
                } else {
                    tt.weights[b] = c;
                }
            }
            "2" => {
                let b = input.u32() as usize - 1;
                let c = input.u32() as usize - 1;
                debug_assert!(b <= c);

                let mut ans = 0i64;
                if b == c {
                    ans = tt.weights[b];
                } else {
                    ans += tt.weights[b];
                    ans += tt.weights[c];

                    let hb = tt.handle[b].unwrap();
                    let hc = tt.handle[c].unwrap();
                    let h_range = match (b == 0, c == n - 1) {
                        (true, true) => {
                            tt.splay(hb);
                            hb
                        }
                        (false, true) => {
                            tt.splay(hb);
                            hb.get_in(&tt.cs).pivot.as_ref().unwrap().children[1]
                        }
                        (true, false) => {
                            tt.splay(hc);
                            hc.get_in(&tt.cs).pivot.as_ref().unwrap().children[0]
                        }
                        (false, false) => {
                            tt.splay(hb);
                            unsafe { tt.guarded_splay(hc, Some(hb)) };

                            hc.get_in(&tt.cs).pivot.as_ref().unwrap().children[0]
                        }
                    };

                    ans += h_range.get_in(&tt.cs).sum;
                }

                writeln!(output, "{ans}").unwrap();
            }
            _ => panic!(),
        }

        debug::with(|| {
            println!("weights {:?}", tt.weights);
            for (u, c) in tt.cs.nodes.iter().enumerate().skip(1) {
                println!("cs[{}] = {:?}", u, c.sum);
            }
        });
    }
}
