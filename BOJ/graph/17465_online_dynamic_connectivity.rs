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

mod rustc_hash {
    /* MIT LICENSE
     *
     * Copyright (c) The Rust Project Contributors
     *
     * Permission is hereby granted, free of charge, to any
     * person obtaining a copy of this software and associated
     * documentation files (the "Software"), to deal in the
     * Software without restriction, including without
     * limitation the rights to use, copy, modify, merge,
     * publish, distribute, sublicense, and/or sell copies of
     * the Software, and to permit persons to whom the Software
     * is furnished to do so, subject to the following
     * conditions:
     *
     * The above copyright notice and this permission notice
     * shall be included in all copies or substantial portions
     * of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
     * ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
     * TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
     * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
     * SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
     * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
     * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
     * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     * DEALINGS IN THE SOFTWARE.
     */

    use core::hash::{BuildHasher, Hasher};
    use std::collections::{HashMap, HashSet};

    pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
    pub type FxHashSet<V> = HashSet<V, FxBuildHasher>;

    #[derive(Clone)]
    pub struct FxHasher {
        hash: usize,
    }

    #[cfg(target_pointer_width = "64")]
    const K: usize = 0xf1357aea2e62a9c5;
    #[cfg(target_pointer_width = "32")]
    const K: usize = 0x93d765dd;

    impl FxHasher {
        pub const fn with_seed(seed: usize) -> FxHasher {
            FxHasher { hash: seed }
        }

        pub const fn default() -> FxHasher {
            FxHasher { hash: 0 }
        }
    }

    impl Default for FxHasher {
        #[inline]
        fn default() -> FxHasher {
            Self::default()
        }
    }

    impl FxHasher {
        #[inline]
        fn add_to_hash(&mut self, i: usize) {
            self.hash = self.hash.wrapping_add(i).wrapping_mul(K);
        }
    }

    impl Hasher for FxHasher {
        #[inline]
        fn write(&mut self, bytes: &[u8]) {
            self.write_u64(hash_bytes(bytes));
        }

        #[inline]
        fn write_u8(&mut self, i: u8) {
            self.add_to_hash(i as usize);
        }

        #[inline]
        fn write_u16(&mut self, i: u16) {
            self.add_to_hash(i as usize);
        }

        #[inline]
        fn write_u32(&mut self, i: u32) {
            self.add_to_hash(i as usize);
        }

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.add_to_hash(i as usize);
            #[cfg(target_pointer_width = "32")]
            self.add_to_hash((i >> 32) as usize);
        }

        #[inline]
        fn write_u128(&mut self, i: u128) {
            self.add_to_hash(i as usize);
            #[cfg(target_pointer_width = "32")]
            self.add_to_hash((i >> 32) as usize);
            self.add_to_hash((i >> 64) as usize);
            #[cfg(target_pointer_width = "32")]
            self.add_to_hash((i >> 96) as usize);
        }

        #[inline]
        fn write_usize(&mut self, i: usize) {
            self.add_to_hash(i);
        }

        #[inline]
        fn finish(&self) -> u64 {
            #[cfg(target_pointer_width = "64")]
            const ROTATE: u32 = 26;
            #[cfg(target_pointer_width = "32")]
            const ROTATE: u32 = 15;

            self.hash.rotate_left(ROTATE) as u64
        }
    }

    const SEED1: u64 = 0x243f6a8885a308d3;
    const SEED2: u64 = 0x13198a2e03707344;
    const PREVENT_TRIVIAL_ZERO_COLLAPSE: u64 = 0xa4093822299f31d0;

    #[inline]
    fn multiply_mix(x: u64, y: u64) -> u64 {
        if cfg!(any(
            all(
                target_pointer_width = "64",
                not(any(target_arch = "sparc64", target_arch = "wasm64")),
            ),
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_family = "wasm", target_feature = "wide-arithmetic"),
        )) {
            let full = (x as u128).wrapping_mul(y as u128);
            let lo = full as u64;
            let hi = (full >> 64) as u64;

            lo ^ hi
        } else {
            let lx = x as u32;
            let ly = y as u32;
            let hx = (x >> 32) as u32;
            let hy = (y >> 32) as u32;

            let afull = (lx as u64).wrapping_mul(hy as u64);
            let bfull = (hx as u64).wrapping_mul(ly as u64);

            afull ^ bfull.rotate_right(32)
        }
    }

    #[inline]
    fn hash_bytes(bytes: &[u8]) -> u64 {
        let len = bytes.len();
        let mut s0 = SEED1;
        let mut s1 = SEED2;

        if len <= 16 {
            if len >= 8 {
                s0 ^= u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                s1 ^= u64::from_le_bytes(bytes[len - 8..].try_into().unwrap());
            } else if len >= 4 {
                s0 ^= u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as u64;
                s1 ^= u32::from_le_bytes(bytes[len - 4..].try_into().unwrap()) as u64;
            } else if len > 0 {
                let lo = bytes[0];
                let mid = bytes[len / 2];
                let hi = bytes[len - 1];
                s0 ^= lo as u64;
                s1 ^= ((hi as u64) << 8) | mid as u64;
            }
        } else {
            let mut off = 0;
            while off < len - 16 {
                let x = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                let y = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());

                let t = multiply_mix(s0 ^ x, PREVENT_TRIVIAL_ZERO_COLLAPSE ^ y);
                s0 = s1;
                s1 = t;
                off += 16;
            }

            let suffix = &bytes[len - 16..];
            s0 ^= u64::from_le_bytes(suffix[0..8].try_into().unwrap());
            s1 ^= u64::from_le_bytes(suffix[8..16].try_into().unwrap());
        }

        multiply_mix(s0, s1) ^ (len as u64)
    }

    #[derive(Copy, Clone, Default)]
    pub struct FxBuildHasher;

    impl BuildHasher for FxBuildHasher {
        type Hasher = FxHasher;
        fn build_hasher(&self) -> FxHasher {
            FxHasher::default()
        }
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
        ops::{Index, IndexMut},
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
        fn usize(&self) -> usize {
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

    impl<V> Index<NodeRef> for SplayForest<V> {
        type Output = V;

        #[inline(always)]
        fn index(&self, index: NodeRef) -> &Self::Output {
            unsafe { &self.pool[index.usize()].assume_init_ref() }
        }
    }

    impl<V> IndexMut<NodeRef> for SplayForest<V> {
        #[inline(always)]
        fn index_mut(&mut self, index: NodeRef) -> &mut Self::Output {
            unsafe { self.pool[index.usize()].assume_init_mut() }
        }
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

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut V, [Option<&'a mut V>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = (&mut *pool_ptr.add(u.usize())).assume_init_mut();
                let children = node.link().children.map(|child| {
                    child.map(|child| (&mut *pool_ptr.add(child.usize())).assume_init_mut())
                });
                (node, children)
            }
        }

        pub fn with<T>(&mut self, u: NodeRef, f: impl FnOnce(&mut V) -> T) -> T {
            let res = f(&mut self[u]);
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

        pub fn branch(&self, u: NodeRef) -> Option<(NodeRef, Branch)> {
            let p = self[u].link().parent?;
            if self[p].link().children[Branch::Left.usize()] == Some(u) {
                Some((p, Branch::Left))
            } else if self[p].link().children[Branch::Right.usize()] == Some(u) {
                Some((p, Branch::Right))
            } else {
                None
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.branch(u).is_none()
        }

        fn attach(&mut self, u: NodeRef, c: NodeRef, branch: Branch) {
            debug_assert_ne!(u, c);
            self[u].link_mut().children[branch as usize] = Some(c);
            self[c].link_mut().parent = Some(u);
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            let child = self[u].link_mut().children[branch as usize].take()?;
            self[child].link_mut().parent = None;
            Some(child)
        }

        unsafe fn rotate(&mut self, u: NodeRef) {
            let (p, b) = self.branch(u).unwrap_unchecked();

            if let Some(c) = self[u].link_mut().children[b.inv().usize()].replace(p) {
                self[p].link_mut().children[b as usize] = Some(c);
                self[c].link_mut().parent = Some(p);
            } else if let Some(c) = self[p].link_mut().children[b.usize()].take() {
                self[c].link_mut().parent = None;
            }

            if let Some((g, bg)) = self.branch(p) {
                self[g].link_mut().children[bg.usize()] = Some(u);
                self[u].link_mut().parent = Some(g);
            } else {
                self[u].link_mut().parent = None;
            }
            self[p].link_mut().parent = Some(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            unsafe {
                while let Some((p, bp)) = self.branch(u) {
                    if let Some((g, bg)) = self.branch(p) {
                        self.push_down(g);
                        self.push_down(p);
                        self.push_down(u);
                        if bp != bg {
                            self.rotate(u); // zig-zig
                        } else {
                            self.rotate(p); // zig-zag
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
                    next(self, u).and_then(|branch| self[u].link().children[branch.usize()])
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
            if let Some(left) = self[u].link().children[Branch::Left.usize()] {
                self.inorder(left, visitor);
            }
            visitor(self, u);
            if let Some(right) = self[u].link().children[Branch::Right.usize()] {
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
    use crate::rustc_hash::FxHashMap as HashMap;

    use super::splay;
    // use super::wbtree;

    fn rotate_to_front<S: splay::NodeSpec>(forest: &mut splay::SplayForest<S>, u: splay::NodeRef) {
        forest.splay(u);
        let left = forest.split_left(u);
        forest.merge(Some(u), left);
    }

    pub struct DynamicEulerTour<S: splay::NodeSpec> {
        pub pool: splay::SplayForest<S>,
        pub freed: Vec<splay::NodeRef>,

        pub n_verts: usize,
        pub verts: Vec<splay::NodeRef>,
        pub edges: HashMap<[u32; 2], splay::NodeRef>,
    }

    impl<S: splay::NodeSpec> DynamicEulerTour<S> {
        pub fn new(vert_nodes: impl IntoIterator<Item = S>) -> Self {
            let mut this = Self {
                pool: splay::SplayForest::new(),
                freed: vec![], // Recycle deleted edge nodes

                verts: vec![],
                edges: HashMap::default(),
                n_verts: 0,
            };
            for node in vert_nodes {
                let u = this.pool.add_root(node);
                this.verts.push(u);
                this.n_verts += 1;
            }
            this
        }

        pub fn add_root(&mut self, node: S) -> splay::NodeRef {
            if let Some(u) = self.freed.pop() {
                self.pool.with(u, |u| {
                    *u = node;
                });
                u
            } else {
                self.pool.add_root(node)
            }
        }

        pub fn reroot(&mut self, u: usize) {
            rotate_to_front(&mut self.pool, self.edges[&[u as u32, u as u32]]);
        }

        pub fn find_root(&mut self, u: usize) -> splay::NodeRef {
            let mut u = self.verts[u];
            self.pool.splay(u);
            self.pool.splay_first(&mut u);
            u
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            self.find_root(u) == self.find_root(v)
        }

        pub fn contains_edge(&self, u: usize, v: usize) -> bool {
            self.edges.contains_key(&[u as u32, v as u32])
        }

        pub fn link(&mut self, u: usize, v: usize, edge_uv: S, edge_vu: S) -> bool {
            if self.is_connected(u, v) {
                return false;
            }
            let vert_u = self.verts[u];
            let vert_v = self.verts[v];
            let edge_uv = self.add_root(edge_uv);
            let edge_vu = self.add_root(edge_vu);
            self.edges.insert([u as u32, v as u32], edge_uv);
            self.edges.insert([v as u32, u as u32], edge_vu);

            rotate_to_front(&mut self.pool, vert_u);
            rotate_to_front(&mut self.pool, vert_v);
            let lhs = self.pool.merge_nonnull(vert_u, edge_uv);
            let rhs = self.pool.merge_nonnull(vert_v, edge_vu);
            self.pool.merge_nonnull(lhs, rhs);
            true
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            let (Some(edge_uv), Some(edge_vu)) = (
                self.edges.remove(&[u as u32, v as u32]),
                self.edges.remove(&[v as u32, u as u32]),
            ) else {
                return false;
            };

            rotate_to_front(&mut self.pool, edge_uv);
            self.pool.split_right(edge_uv);
            self.pool.split_left(edge_vu);
            self.pool.split_right(edge_vu);
            self.freed.push(edge_uv);
            self.freed.push(edge_vu);
            true
        }
    }
}

pub mod graph_connectivity {
    // Online dynamic connectivity in graphs
    // https://codeforces.com/blog/entry/128556
    use crate::rustc_hash::FxHashMap as HashMap;
    use crate::rustc_hash::FxHashSet as HashSet;
    use std::collections::hash_map;

    use super::euler_tour_tree::DynamicEulerTour;
    use super::splay;

    fn sorted2<T: PartialOrd>(mut xs: [T; 2]) -> [T; 2] {
        if xs[0] > xs[1] {
            xs.swap(0, 1);
        }
        xs
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum NodeState {
        FloatingTreeEdge = 0b10,
        VertWithBackEdge = 0b01,
        None = 0,
    }

    pub struct Node {
        size: u32,
        edge: [u32; 2],
        state: NodeState,
        subtree_flag: u8,
        link: splay::Link,
    }

    impl Default for Node {
        fn default() -> Self {
            Self {
                size: 1,
                state: NodeState::None,
                subtree_flag: 0,
                edge: [0, 0],
                link: splay::Link::default(),
            }
        }
    }

    impl Node {
        fn new(u: usize, v: usize, state: NodeState) -> Self {
            let edge = [u as u32, v as u32];
            Self {
                edge,
                state,
                subtree_flag: state as u8,
                ..Self::default()
            }
        }

        fn edge(u: usize, v: usize) -> Self {
            Self::new(u, v, NodeState::FloatingTreeEdge)
        }

        fn vert(u: usize) -> Self {
            Self::new(u, u, NodeState::None)
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
            self.subtree_flag = self.state as u8;

            let mut left_flag = 0;
            if let Some(c) = children[0].as_ref() {
                self.size += c.size;
                self.subtree_flag |= c.subtree_flag;
                left_flag = c.subtree_flag;
            }
            if let Some(c) = children[1].as_ref() {
                self.size += c.size;
                self.subtree_flag |= c.subtree_flag;
            }
            self.subtree_flag &= (1 << 4) - 1;
            self.subtree_flag |= left_flag << 4;
        }
    }

    pub struct Conn {
        pub n_verts: usize,
        pub n_levels: usize,
        pub level: HashMap<[u32; 2], u32>,
        pub spanning_forests: Vec<DynamicEulerTour<Node>>,
        pub adj_back_edges: Vec<HashSet<u32>>,
    }

    impl Conn {
        pub fn new(n_verts: usize) -> Self {
            let n_levels = (u32::BITS - u32::leading_zeros(n_verts as u32)) as usize;
            Self {
                n_verts,
                n_levels,
                level: HashMap::default(),
                spanning_forests: (0..n_levels)
                    .map(|_| DynamicEulerTour::new((0..n_verts).map(|u| Node::vert(u))))
                    .collect(),
                adj_back_edges: vec![HashSet::default(); n_levels * n_verts],
            }
        }

        pub fn contains_edge(&self, u: usize, v: usize) -> bool {
            self.level.contains_key(&sorted2([u as u32, v as u32]))
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.n_verts && v < self.n_verts);
            self.spanning_forests[0].is_connected(u, v)
        }

        fn link_trees_in_level(&mut self, u: usize, v: usize, level: usize) {
            self.spanning_forests[level].link(u, v, Node::edge(u, v), Node::edge(v, u));
        }

        fn link_in_level(&mut self, u: usize, v: usize, level: usize) -> bool {
            let n_verts = self.n_verts;
            let hash_map::Entry::Vacant(e) = self.level.entry(sorted2([u as u32, v as u32])) else {
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
                    self.spanning_forests[level].pool.splay(hs);
                    self.spanning_forests[level].pool.with(hs, |node| {
                        node.state = NodeState::VertWithBackEdge;
                    });
                }
            }
            true
        }

        pub fn link(&mut self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.n_verts && v < self.n_verts);
            self.link_in_level(u, v, 0)
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            debug_assert!(u < self.n_verts && v < self.n_verts);
            let Some(base_level) = self.level.remove(&sorted2([u as u32, v as u32])) else {
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
                        self.spanning_forests[base_level].pool.splay(hs);
                        self.spanning_forests[base_level]
                            .pool
                            .with(hs, |node| node.state = NodeState::None);
                    }
                }
            } else {
                for level in (0..=base_level).rev() {
                    self.spanning_forests[level].cut(u, v);
                }

                let mut replacement_edge: Option<[u32; 2]> = None;
                for level in (0..=base_level).rev() {
                    if let Some([s, t]) = replacement_edge {
                        self.link_trees_in_level(s as usize, t as usize, level);
                        continue;
                    }

                    let mut ett = &mut self.spanning_forests[level];
                    let hu = ett.verts[u];
                    let hv = ett.verts[v];
                    ett.pool.splay(hu);
                    ett.pool.splay(hv);
                    let (mut h_small, small) = if ett.pool[hu].size <= ett.pool[hv].size {
                        (hu, u)
                    } else {
                        (hv, v)
                    };
                    let large = small ^ u ^ v;

                    // Push MSF edges down to the lower level
                    ett.pool.splay(h_small);
                    while ett.pool[h_small].subtree_flag & NodeState::FloatingTreeEdge as u8 != 0 {
                        ett.pool.splay_by(&mut h_small, |forest, u| {
                            let node = &forest[u];
                            if node.state == NodeState::FloatingTreeEdge {
                                None
                            } else if node.subtree_flag & ((NodeState::FloatingTreeEdge as u8) << 4)
                                != 0
                            {
                                Some(splay::Branch::Left)
                            } else {
                                Some(splay::Branch::Right)
                            }
                        });
                        let [s, t] = ett.pool[h_small].edge;
                        ett.pool.with(h_small, |node| node.state = NodeState::None);

                        if s < t {
                            self.level.insert(sorted2([s, t]), level as u32 + 1);
                            self.link_trees_in_level(s as usize, t as usize, level + 1);
                        }

                        ett = &mut self.spanning_forests[level];
                    }

                    // Find a replacement edge
                    'outer: while ett.pool[h_small].subtree_flag & NodeState::VertWithBackEdge as u8
                        != 0
                    {
                        ett.pool.splay_by(&mut h_small, |pool, u| {
                            let node = &pool[u];
                            if node.state == NodeState::VertWithBackEdge {
                                None
                            } else if node.subtree_flag & ((NodeState::VertWithBackEdge as u8) << 4)
                                != 0
                            {
                                Some(splay::Branch::Left)
                            } else {
                                Some(splay::Branch::Right)
                            }
                        });
                        let [s, _] = ett.pool[h_small].edge;

                        while let Some(t) = self.adj_back_edges[level * self.n_verts + s as usize]
                            .iter()
                            .next()
                            .copied()
                        {
                            self.level.remove(&sorted2([s, t]));
                            self.adj_back_edges[level * self.n_verts + s as usize].remove(&t);
                            self.adj_back_edges[level * self.n_verts + t as usize].remove(&s);
                            for r in [s, t] {
                                let degree =
                                    self.adj_back_edges[level * self.n_verts + r as usize].len();
                                if degree == 0 {
                                    ett = &mut self.spanning_forests[level];

                                    let hr = ett.verts[r as usize];
                                    ett.pool.splay(hr);
                                    ett.pool.with(hr, |node| node.state = NodeState::None);
                                }
                            }
                            if self.is_connected(t as usize, large) {
                                replacement_edge = Some([s, t]);
                                self.link_in_level(s as usize, t as usize, level);
                                break 'outer;
                            }
                            self.link_in_level(s as usize, t as usize, level + 1);
                        }

                        ett = &mut self.spanning_forests[level];
                        ett.pool.splay(h_small);
                    }
                }
            }

            true
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

        f += n_components as u64;
    }
}
