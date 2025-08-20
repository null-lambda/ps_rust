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
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
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
        hash: u64,
    }

    const K: u64 = 0xf1357aea2e62a9c5;

    impl FxHasher {
        pub const fn with_seed(seed: u64) -> FxHasher {
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
        fn add_to_hash(&mut self, i: u64) {
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
            self.add_to_hash(i as u64);
        }

        #[inline]
        fn write_u16(&mut self, i: u16) {
            self.add_to_hash(i as u64);
        }

        #[inline]
        fn write_u32(&mut self, i: u32) {
            self.add_to_hash(i as u64);
        }

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.add_to_hash(i as u64);
        }

        #[inline]
        fn write_u128(&mut self, i: u128) {
            self.add_to_hash(i as u64);
            self.add_to_hash((i >> 64) as u64);
        }

        #[inline]
        fn write_usize(&mut self, i: usize) {
            self.add_to_hash(i as u64);
        }

        #[inline]
        fn finish(&self) -> u64 {
            const ROTATE: u32 = 26;
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

pub mod full_conn {
    // Full dynamic connectivity through the Cluster Forest with Local Trees.
    //
    // ## Implementation details
    //   - Edge edge is assigned a level in [0, lv_max],
    //     where level 0 corresponds to roots and level lv_max to leaves (vertices).
    //   - Size invariant: Forall cluster node, n_vert <= 2^(max_level - level).
    //   - Path-compression: A cluster node is materialized when it contains an edge at a given level,
    //     and dematerialized when it does not.
    //
    // ## Reference
    //   - Holm, Jacob, Kristian de Lichtenberg, and Mikkel Thorup.
    //     'Poly-Logarithmic Deterministic Fully-Dynamic Algorithms for Connectivity, Minimum Spanning Tree, 2-Edge, and Biconnectivity'. 2001.
    //     [https://doi.org/10.1145/502090.502095]
    //   - Thorup, Mikkel.
    //     'Near-Optimal Fully-Dynamic Graph Connectivity'. 2000.
    //     [https://doi.org/10.1145/335305.335345]
    //   - Wulff-Nilsen, Christian.
    //     'Faster Deterministic Fully-Dynamic Graph Connectivity'. 2013.
    //     [https://doi.org/10.1137/1.9781611973105.126]
    //   - Man, Quinten De, Laxman Dhulipala, Adam Karczmarz, Jakub Łącki, Julian Shun, and Zhongqi Wang.
    //     'Towards Scalable and Practical Batch-Dynamic Connectivity'. 2024.
    //     [http://arxiv.org/abs/2411.11781]

    use std::{
        cell::Cell,
        cmp::{Ordering, Reverse},
        collections::BinaryHeap,
        ops::{Index, IndexMut},
    };

    use crate::rustc_hash::FxHashMap as HashMap;
    use crate::rustc_hash::FxHashSet as HashSet;

    mod cmp {
        // The equalizer of all things
        use std::cmp::Ordering;

        #[repr(transparent)]
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

    pub const fn log2_floor_p1(n: u32) -> u32 {
        u32::BITS - u32::leading_zeros(n)
    }

    pub const fn rank(n: u32) -> u8 {
        (u32::BITS - 1 - u32::leading_zeros(n)) as u8
    }

    pub const N_MAX: usize = 1e6 as usize + 10;
    pub const L_MAX: usize =
        (usize::BITS - usize::leading_zeros(N_MAX.next_power_of_two())) as usize;

    fn sorted2<T: PartialOrd>(mut xs: [T; 2]) -> [T; 2] {
        if xs[0] > xs[1] {
            xs.swap(0, 1);
        }
        xs
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeRef(u32);

    #[repr(u32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NodeType {
        Leaf = 0 << 30,
        Local = 1 << 30,
        Cluster = 2 << 30,
    }

    impl NodeRef {
        const IDX_MASK: u32 = (1 << 30) - 1;

        #[inline(always)]
        const fn new(ty: NodeType, u: u32) -> Self {
            debug_assert!(u & !Self::IDX_MASK == 0);
            Self(ty as u32 | u)
        }

        #[inline(always)]
        fn ty(&self) -> NodeType {
            match self.0 >> 30 {
                0 => NodeType::Leaf,
                1 => NodeType::Local,
                2 => NodeType::Cluster,
                _ => panic!(),
            }
        }

        #[inline(always)]
        fn usize(&self) -> usize {
            (self.0 & !(0b11 << 30)) as usize
        }
    }

    impl std::fmt::Debug for NodeRef {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if *self == UNSET {
                write!(f, "UNSET")
            } else {
                write!(f, "{:?}({:?})", self.ty(), self.usize())
            }
        }
    }

    const UNSET: NodeRef = NodeRef(u32::MAX);

    impl Default for NodeRef {
        fn default() -> Self {
            UNSET
        }
    }

    pub trait Node {
        const TY: NodeType;
    }

    #[derive(Clone, Debug, Default)]
    struct ClusterNode {
        parent: NodeRef,
        children: [NodeRef; 2],
        lv: u8,

        emask: u32,
        n_vert: u32,

        marker: Cell<u8>,
    }

    #[derive(Clone, Debug, Default)]
    struct LocalNode {
        parent: NodeRef,
        children: [NodeRef; 2],

        emask: u32,
        n_vert: u32,
    }

    // Mark edges that might affect connectivity
    // (part of a spanning subgraph, not necessarily a tree).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum EdgeType {
        Back = 0,
        Span = 1,
    }

    type AdjSet = HashMap<u32, Cell<EdgeType>>;
    type AdjSetIter<'a> = std::collections::hash_map::Iter<'a, u32, Cell<EdgeType>>;

    #[derive(Clone, Debug, Default)]
    struct LeafNode {
        parent: NodeRef,

        // Flat-set of neighbors
        neighbors: Vec<AdjSet>,
        emask: u32,

        marker: Cell<u8>,
    }

    impl Node for ClusterNode {
        const TY: NodeType = NodeType::Cluster;
    }

    impl Node for LocalNode {
        const TY: NodeType = NodeType::Local;
    }

    impl Node for LeafNode {
        const TY: NodeType = NodeType::Leaf;
    }

    impl LeafNode {
        fn get(&self, lv: u8) -> Option<&AdjSet> {
            if self.emask & (1 << lv) == 0 {
                return None;
            }

            let k = (self.emask & ((1 << lv) - 1)).count_ones() as usize;
            Some(&self.neighbors[k])
        }

        fn get_or_default(&mut self, lv: u8) -> &mut AdjSet {
            let k = (self.emask & ((1 << lv) - 1)).count_ones() as usize;
            if self.emask & (1 << lv) == 0 {
                self.emask |= 1 << lv;
                self.neighbors.insert(k, AdjSet::default());
            }

            &mut self.neighbors[k]
        }

        fn remove_if(&mut self, lv: u8, pred: impl FnOnce(&mut AdjSet) -> bool) -> bool {
            assert!(self.emask & (1 << lv) != 0);
            let k = (self.emask & ((1 << lv) - 1)).count_ones() as usize;
            if pred(&mut self.neighbors[k]) {
                self.emask ^= 1 << lv;
                self.neighbors.remove(k);
                true
            } else {
                false
            }
        }

        fn find_level(&self, v_adj: u32) -> Option<u8> {
            let mut rows = self.neighbors.iter().rev();
            for lv in (0..32).rev() {
                if self.emask & (1 << lv) != 0 {
                    if rows.next().unwrap().contains_key(&v_adj) {
                        return Some(lv);
                    }
                }
            }

            None
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct Pool<T> {
        nodes: Vec<T>,
        free: Vec<NodeRef>,
    }

    impl<T: Node> Pool<T> {
        fn alloc(&mut self, node: T) -> NodeRef {
            let u = if let Some(u) = self.free.pop() {
                self[u] = node;
                u
            } else {
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                NodeRef::new(T::TY, idx)
            };
            u
        }

        fn mark_free(&mut self, u: NodeRef) {
            debug_assert_eq!(T::TY, u.ty());
            self.free.push(u);
        }

        fn capacity(&self) -> usize {
            self.nodes.len()
        }
    }

    impl<T> Index<NodeRef> for Pool<T> {
        type Output = T;
        fn index(&self, index: NodeRef) -> &Self::Output {
            &self.nodes[index.usize()]
        }
    }

    impl<T> IndexMut<NodeRef> for Pool<T> {
        fn index_mut(&mut self, index: NodeRef) -> &mut Self::Output {
            &mut self.nodes[index.usize()]
        }
    }

    #[derive(Clone, Debug)]
    pub struct ClusterForest {
        lv_max: u8,

        cluster: Pool<ClusterNode>,
        local: Pool<LocalNode>,
        leaf: Vec<LeafNode>,
    }

    impl ClusterForest {
        pub fn new(n_verts: usize) -> Self {
            assert!(n_verts >= 1);
            let lv_max = (usize::BITS - usize::leading_zeros(n_verts.next_power_of_two())) as u8;

            Self {
                lv_max,

                cluster: Pool::default(),
                local: Pool::default(),
                leaf: vec![LeafNode::default(); n_verts],
            }
        }

        pub fn n_verts(&self) -> usize {
            self.leaf.len()
        }

        fn parent(&self, s: NodeRef) -> NodeRef {
            match s.ty() {
                NodeType::Cluster => self.cluster[s].parent,
                NodeType::Local => self.local[s].parent,
                NodeType::Leaf => self.leaf[s.usize()].parent,
            }
        }

        fn parent_mut(&mut self, s: NodeRef) -> &mut NodeRef {
            match s.ty() {
                NodeType::Cluster => &mut self.cluster[s].parent,
                NodeType::Local => &mut self.local[s].parent,
                NodeType::Leaf => &mut self.leaf[s.usize()].parent,
            }
        }

        fn children(&self, s: NodeRef) -> [NodeRef; 2] {
            match s.ty() {
                NodeType::Cluster => self.cluster[s].children,
                NodeType::Local => self.local[s].children,
                NodeType::Leaf => [UNSET; 2],
            }
        }

        fn children_mut(&mut self, s: NodeRef) -> &mut [NodeRef; 2] {
            match s.ty() {
                NodeType::Cluster => &mut self.cluster[s].children,
                NodeType::Local => &mut self.local[s].children,
                NodeType::Leaf => panic!(),
            }
        }

        fn level(&self, s: NodeRef) -> u8 {
            match s.ty() {
                NodeType::Leaf => self.lv_max,
                NodeType::Local => panic!(),
                NodeType::Cluster => self.cluster[s].lv,
            }
        }

        fn emask(&self, s: NodeRef) -> u32 {
            match s.ty() {
                NodeType::Cluster => self.cluster[s].emask,
                NodeType::Local => self.local[s].emask,
                NodeType::Leaf => self.leaf[s.usize()].emask,
            }
        }

        fn n_vert(&self, s: NodeRef) -> u32 {
            match s.ty() {
                NodeType::Cluster => self.cluster[s].n_vert,
                NodeType::Local => self.local[s].n_vert,
                NodeType::Leaf => 1,
            }
        }

        fn marker(&self, s: NodeRef) -> &Cell<u8> {
            match s.ty() {
                NodeType::Cluster => &self.cluster[s].marker,
                NodeType::Local => panic!(),
                NodeType::Leaf => &self.leaf[s.usize()].marker,
            }
        }

        #[inline]
        fn push_down_link(&mut self, s: NodeRef) {
            match s.ty() {
                NodeType::Cluster => {
                    let [c0, c1] = self.cluster[s].children;
                    if c0 != UNSET {
                        *self.parent_mut(c0) = s;
                    }
                    *self.parent_mut(c1) = s;
                }
                NodeType::Local => {
                    let [c0, c1] = self.local[s].children;
                    if c0 != UNSET {
                        *self.parent_mut(c0) = s;
                    }
                    *self.parent_mut(c1) = s;
                }
                NodeType::Leaf => {}
            }
        }

        fn pull_up(&mut self, s: NodeRef) {
            match s.ty() {
                NodeType::Cluster => {
                    let [c0, c1] = self.cluster[s].children;
                    self.cluster[s].n_vert = self.n_vert(c1);
                    self.cluster[s].emask = self.emask(c1);
                    if c0 != UNSET {
                        self.cluster[s].n_vert += self.n_vert(c0);
                        self.cluster[s].emask |= self.emask(c0);
                    }
                }
                NodeType::Local => {
                    let [c0, c1] = self.local[s].children;
                    self.local[s].n_vert = self.n_vert(c1);
                    self.local[s].emask = self.emask(c1);
                    if c0 != UNSET {
                        self.local[s].n_vert += self.n_vert(c0);
                        self.local[s].emask |= self.emask(c0);
                    }
                }
                NodeType::Leaf => {}
            }
        }

        fn ascend_to_root(&self, mut s: NodeRef) -> NodeRef {
            loop {
                let p = self.parent(s);
                if p == UNSET {
                    break s;
                }
                s = p;
            }
        }

        fn pull_up_to_guard(&mut self, mut s: NodeRef, guard: NodeRef) -> NodeRef {
            loop {
                let p = self.parent(s);
                if p == guard {
                    break s;
                }
                s = p;
                self.pull_up(s);
            }
        }

        #[inline(always)]
        fn pull_up_to_root(&mut self, s: NodeRef) -> NodeRef {
            self.pull_up_to_guard(s, UNSET)
        }

        fn verify_strict_path_compression_invariant(&self, u: u32) {
            let mut s = NodeRef::new(NodeType::Leaf, u);

            let mut levels_mask = 0u32;
            loop {
                let p = self.parent(s);
                if p == UNSET {
                    break;
                }
                s = p;
                if s.ty() == NodeType::Cluster {
                    assert!(self.cluster[s].emask & (1 << self.cluster[s].lv) != 0);
                    levels_mask |= 1 << self.level(s);
                }
            }

            let l = &self.leaf[u as usize];
            assert_eq!(
                l.emask,
                l.emask & levels_mask,
                "{:020b} {:020b}",
                l.emask,
                levels_mask,
            );
        }

        fn verify_weak_path_compression_invariant(&self, u: u32) {
            let mut s = NodeRef::new(NodeType::Leaf, u);
            let mut levels_mask = 0u32;
            let mut pc = vec![];
            loop {
                let p = self.parent(s);
                if p == UNSET {
                    break;
                }
                s = p;
                if s.ty() == NodeType::Cluster {
                    pc.push(s);
                    levels_mask |= 1 << self.level(s);
                }
            }

            let l = &self.leaf[u as usize];
            assert_eq!(
                l.emask,
                l.emask & levels_mask,
                "{:020b} {:020b} {:?}",
                l.emask,
                levels_mask,
                pc
            );
        }

        fn verify_size_invariant(&self, u: u32) {
            let mut s = NodeRef::new(NodeType::Leaf, u);
            loop {
                s = self.parent(s);
                if s == UNSET {
                    break;
                }
                if s.ty() == NodeType::Cluster {
                    assert!(self.cluster[s].n_vert <= 1 << self.lv_max - self.cluster[s].lv);
                }
            }
        }

        fn verify_link(&mut self, u: u32, guard: NodeRef) {
            let mut s = NodeRef::new(NodeType::Leaf, u);
            if s == guard {
                return;
            }

            loop {
                let p = self.parent(s);
                if p == guard {
                    break;
                }
                if p == UNSET {
                    if s.ty() == NodeType::Local {
                        panic!("u={u}, s={s:?}");
                    }
                    break;
                }

                s = p;
                let [c0, c1] = match s.ty() {
                    NodeType::Leaf => [UNSET; 2],
                    NodeType::Local => self.local[s].children,
                    NodeType::Cluster => self.cluster[s].children,
                };
                assert!(
                    c0 == UNSET || self.parent(c0) == s,
                    "{:?}",
                    (s, c0, self.parent(c0))
                );
                assert!(
                    c1 != UNSET && self.parent(c1) == s,
                    "{:?}",
                    (s, c1, self.parent(c1))
                );

                let prev = (self.emask(s),);
                self.pull_up(s);
                let curr = (self.emask(s),);
                assert_eq!(prev, curr, "{:?} {:?}", s, self.parent(s));
            }
        }

        fn verify_path_to_root(&mut self, u: u32) {
            // self.verify_weak_path_compression_invariant(u);

            // self.verify_strict_path_compression_invariant(u);
            // self.verify_size_invariant(u);
            // self.verify_link(u, UNSET);
        }

        // Returns the topmost cluster node (including leaf) materialized in a level >= lv
        pub fn ascend_to_level(&self, u: u32, lv: u8) -> (NodeRef, NodeRef) {
            let mut cs = NodeRef::new(NodeType::Leaf, u);

            let mut cp = self.parent(cs);
            while cp != UNSET {
                match cp.ty() {
                    NodeType::Leaf => unreachable!(),
                    NodeType::Local => cp = self.local[cp].parent,
                    NodeType::Cluster => {
                        let lv_p = self.cluster[cp].lv;
                        if lv_p < lv {
                            break;
                        }

                        cs = cp;
                        cp = self.cluster[cp].parent;
                    }
                }
            }

            debug_assert!(cp == UNSET || self.cluster[cp].lv < lv);
            debug_assert!(cs.ty() == NodeType::Leaf || self.cluster[cs].lv >= lv);
            (cp, cs)
        }

        pub fn lca_cluster(&self, u: u32, v: u32) -> Option<NodeRef> {
            let mut ru = NodeRef::new(NodeType::Leaf, u);
            let mut rv = NodeRef::new(NodeType::Leaf, v);

            let ascend_to_cluster = |s: &mut NodeRef| {
                *s = self.parent(*s);
                if *s == UNSET {
                    return;
                }

                while s.ty() == NodeType::Local {
                    *s = self.local[*s].parent;
                }
            };

            loop {
                match self.level(ru).cmp(&self.level(rv)) {
                    Ordering::Less => {
                        ascend_to_cluster(&mut rv);
                        if rv == UNSET {
                            return None;
                        }
                    }
                    Ordering::Greater => {
                        ascend_to_cluster(&mut ru);
                        if ru == UNSET {
                            return None;
                        }
                    }
                    Ordering::Equal => {
                        ascend_to_cluster(&mut ru);
                        if ru == UNSET {
                            return None;
                        }

                        ascend_to_cluster(&mut rv);
                        if rv == UNSET {
                            return None;
                        }
                    }
                }

                if ru == rv {
                    return Some(ru);
                }
            }
        }

        fn extract_rank_forest_rev(&mut self, mut s: NodeRef) -> Vec<NodeRef> {
            assert_eq!(s.ty(), NodeType::Cluster);
            let mut local_roots = vec![];

            let [c0, c1] = self.cluster[s].children;
            local_roots.push(c1);
            s = c0;

            while s != UNSET {
                match s.ty() {
                    NodeType::Local => {
                        let [c0, c1] = self.local[s].children;
                        local_roots.push(c1);
                        self.local.mark_free(s);
                        s = c0;
                    }
                    _ => panic!(),
                }
            }

            local_roots
        }

        fn merge_ordered_rank_forest(
            &mut self,
            px: impl Iterator<Item = NodeRef>,
            py: impl Iterator<Item = NodeRef>,
            mut yield_with: impl FnMut(&mut Self, NodeRef),
        ) {
            let local_join = |this: &mut Self, (rx, x): (u8, NodeRef), (_ry, y): (u8, NodeRef)| {
                let rz = rx + 1;

                let z = this.local.alloc(LocalNode {
                    parent: UNSET,
                    children: [x, y],
                    ..Default::default()
                });

                this.push_down_link(z);
                this.pull_up(z);

                (rz, z)
            };

            let mut carry: Option<(u8, NodeRef)> = None;
            let mut px = px.peekable();
            let mut py = py.peekable();
            loop {
                const R_INF: u8 = u8::MAX;

                let r0 = carry.map_or(R_INF, |(r, _)| r);
                let r1 = px.peek().map_or(R_INF, |&u| rank(self.n_vert(u)));
                let r2 = py.peek().map_or(R_INF, |&u| rank(self.n_vert(u)));
                let r_min = r0.min(r1).min(r2).min(R_INF - 1);

                let mut p0 = || (carry.take().unwrap());
                let mut p1 = || (r1, px.next().unwrap());
                let mut p2 = || (r2, py.next().unwrap());
                match (r0 == r_min, r1 == r_min, r2 == r_min) {
                    (false, false, false) => break,
                    (true, false, false) => yield_with(self, p0().1),
                    (false, true, false) => yield_with(self, p1().1),
                    (false, false, true) => yield_with(self, p2().1),
                    (true, true, false) => carry = Some(local_join(self, p0(), p1())),
                    (false, true, true) => carry = Some(local_join(self, p1(), p2())),
                    (true, false, true) => carry = Some(local_join(self, p0(), p2())),
                    (true, true, true) => {
                        yield_with(self, p0().1);
                        carry = Some(local_join(self, p1(), p2()));
                    }
                }
            }
        }

        pub fn materialize(&mut self, s: &mut NodeRef, lv: u8) -> bool {
            let (lv_s, g) = match s.ty() {
                NodeType::Leaf => (self.lv_max, self.leaf[s.usize()].parent),
                NodeType::Local => panic!(),
                NodeType::Cluster => (self.cluster[*s].lv, self.cluster[*s].parent),
            };

            assert!(lv_s >= lv);
            if lv_s == lv {
                return false;
            }

            let p = self.cluster.alloc(ClusterNode {
                parent: g,
                children: [UNSET, *s],
                lv,
                ..Default::default()
            });
            self.push_down_link(p);
            self.pull_up(p);

            if g != UNSET {
                let c = self.children_mut(g);
                let branch = (c[1] == *s) as usize;
                c[branch] = p;
            }

            *s = p;
            true
        }

        pub fn try_dematerialize(&mut self, s: &mut NodeRef) -> bool {
            debug_assert_eq!(s.ty(), NodeType::Cluster);

            let lv = self.cluster[*s].lv;
            if self.cluster[*s].emask & (1 << lv) != 0 {
                return false;
            }

            let p = self.cluster[*s].parent;
            let [c0, c1] = self.cluster[*s].children;
            debug_assert!(c0 == UNSET && c1.ty() != NodeType::Local);

            *self.parent_mut(c1) = p;
            if p != UNSET {
                let pcs = self.children_mut(p);
                let branch = (pcs[1] == *s) as usize;
                pcs[branch] = c1;
            }
            self.cluster.mark_free(*s);

            *s = c1;
            true
        }

        pub fn merge_cluster(&mut self, cx: &mut NodeRef, mut cy: NodeRef, lv: u8) {
            self.materialize(cx, lv);
            self.materialize(&mut cy, lv);

            debug_assert!(self.parent(cy) == UNSET, "Detach cy first");

            // Delete rv
            let px = self.extract_rank_forest_rev(*cx);
            let py = self.extract_rank_forest_rev(cy);
            self.cluster.mark_free(cy);

            // Merge
            let local_join = |this: &mut Self, x: NodeRef, y: NodeRef| {
                let z = this.local.alloc(LocalNode {
                    parent: UNSET,
                    children: [x, y],
                    ..Default::default()
                });

                this.push_down_link(z);
                this.pull_up(z);
                z
            };

            let mut acc = [UNSET; 2];
            self.merge_ordered_rank_forest(
                px.into_iter().rev(),
                py.into_iter().rev(),
                |this, s| {
                    if acc[1] != UNSET {
                        acc[0] = local_join(this, acc[0], acc[1]);
                    }
                    acc[1] = s;
                },
            );

            debug_assert!(acc[1] != UNSET);
            self.cluster[*cx].children = acc;
            self.push_down_link(*cx);
            self.pull_up(*cx);
        }

        fn detach_clusters(&mut self, cp: NodeRef, cxs: &[NodeRef]) {
            debug_assert_eq!(cp.ty(), NodeType::Cluster);
            let pp = self.extract_rank_forest_rev(cp);
            for &r in &pp {
                *self.parent_mut(r) = UNSET;
            }

            let mut pq = BinaryHeap::new();

            let mut px = vec![];
            let mut banned = HashSet::default();
            for &cx in cxs {
                debug_assert!(cx.ty() != NodeType::Local);
                let mut s = cx;
                let mut p = self.parent(s);
                while p != UNSET {
                    self.local.mark_free(p);

                    let cs = self.local[p].children;
                    let branch = (s == cs[1]) as usize;
                    *self.parent_mut(cs[branch ^ 1]) = UNSET;
                    px.push(cs[branch ^ 1]);

                    banned.insert(s);

                    s = p;
                    p = self.local[s].parent;
                }

                banned.insert(s);

                *self.parent_mut(cx) = UNSET;
            }

            for t in px {
                if !banned.contains(&t) {
                    pq.push((Reverse(rank(self.n_vert(t))), cmp::Trivial(t)));
                }
            }

            for t in pp {
                if !banned.contains(&t) {
                    pq.push((Reverse(rank(self.n_vert(t))), cmp::Trivial(t)));
                }
            }

            // Merge.
            // Unlike `merge_cluster_into`, rank balance along path `px` may be broken,
            // so we require an explicit sorting.
            let local_join = |this: &mut Self, (rx, x): (u8, NodeRef), (_ry, y): (u8, NodeRef)| {
                let rz = rx + 1;

                let z = this.local.alloc(LocalNode {
                    parent: UNSET,
                    children: [x, y],
                    ..Default::default()
                });

                this.push_down_link(z);
                this.pull_up(z);

                (rz, z)
            };

            let mut acc = [UNSET; 2];
            let mut push = |this: &mut Self, s: NodeRef| {
                if acc[1] != UNSET {
                    acc[0] = local_join(this, (0, acc[0]), (0, acc[1])).1;
                }
                acc[1] = s;
            };

            while let Some((Reverse(r0), cmp::Trivial(t0))) = pq.pop() {
                if pq.is_empty() {
                    push(self, t0);
                    break;
                }

                let (Reverse(r1), cmp::Trivial(t1)) = *pq.peek().unwrap();
                if r0 != r1 {
                    push(self, t0);
                    continue;
                }

                pq.pop();
                let (rz, tz) = local_join(self, (r0, t0), (r1, t1));
                pq.push((Reverse(rz), cmp::Trivial(tz)));
            }

            debug_assert!(acc[1] != UNSET);
            self.cluster[cp].children = acc;
            self.push_down_link(cp);
            self.pull_up(cp);
        }

        pub fn link(&mut self, u: u32, v: u32) -> bool {
            if u == v {
                return false;
            }

            if self.leaf[u as usize]
                .neighbors
                .iter()
                .any(|vs| vs.contains_key(&v))
            {
                return false;
            }

            if let Some(r) = self.lca_cluster(u, v) {
                let lv = self.level(r);

                for [u, v] in [[u, v], [v, u]] {
                    self.leaf[u as usize]
                        .get_or_default(lv)
                        .insert(v, EdgeType::Back.into());
                }

                self.pull_up_to_root(NodeRef::new(NodeType::Leaf, u));
                self.pull_up_to_root(NodeRef::new(NodeType::Leaf, v));
            } else {
                for [u, v] in [[u, v], [v, u]] {
                    self.leaf[u as usize]
                        .get_or_default(0)
                        .insert(v, EdgeType::Span.into());
                }

                let mut ru = self.pull_up_to_root(NodeRef::new(NodeType::Leaf, u));
                let rv = self.pull_up_to_root(NodeRef::new(NodeType::Leaf, v));
                self.merge_cluster(&mut ru, rv, 0);
            };

            self.verify_path_to_root(u);
            self.verify_path_to_root(v);

            true
        }

        fn promote_edge(&mut self, u: u32, v: u32, lv: u8, guard: NodeRef) {
            for [x, y] in [[u, v], [v, u]] {
                let fx = &mut self.leaf[x as usize];

                let emask_prev = fx.emask;

                let mut ty = Cell::new(EdgeType::Span); // Dummy
                fx.remove_if(lv, |row| {
                    ty = row.remove(&y).unwrap();
                    row.is_empty()
                });

                assert!(fx.get_or_default(lv + 1).insert(y, ty).is_none());

                if emask_prev != fx.emask {
                    self.pull_up_to_guard(NodeRef::new(NodeType::Leaf, x), guard);
                }
            }
        }

        pub fn cut(&mut self, mut u: u32, mut v: u32) -> bool {
            let Some(lv) = self.leaf[u as usize].find_level(v) else {
                return false;
            };

            let mut lv = lv;
            let (mut cp, mut cu) = self.ascend_to_level(u, lv + 1);
            let (cp_alt, mut cv) = self.ascend_to_level(v, lv + 1);

            assert_eq!(cp, cp_alt);
            assert_eq!(self.cluster[cp].lv, lv);

            self.verify_path_to_root(u);
            self.verify_path_to_root(v);

            let mut ty = [EdgeType::Span; 2]; // Dummy
            if self.leaf[u as usize].remove_if(lv, |row| {
                ty[0] = row.remove(&v).unwrap().get();
                row.is_empty()
            }) {
                self.pull_up_to_root(NodeRef::new(NodeType::Leaf, u));
            }
            if self.leaf[v as usize].remove_if(lv, |row| {
                ty[1] = row.remove(&u).unwrap().get();
                row.is_empty()
            }) {
                self.pull_up_to_root(NodeRef::new(NodeType::Leaf, v));
            }

            // [u, v] is a self-loop in the cluster multigraph.
            if cu == cv {
                return true;
            }

            // [u, v] is a back-edge.
            // Because markers are updated on half-edges, both directions must be checked.
            if ty[0] == EdgeType::Back && ty[1] == EdgeType::Back {
                return true;
            }

            loop {
                let mut trav_u = LazyTraversal::new(&self, 1, u, lv);
                let mut trav_v = LazyTraversal::new(&self, 2, v, lv);

                let mut has_replacement_edge = false;
                let mut terminal_side;
                loop {
                    terminal_side = 0u8;
                    match trav_u.next() {
                        Ok(()) => {}
                        Err(SearchResult::Empty) => break,
                        Err(SearchResult::Replacement) => {
                            has_replacement_edge = true;
                            break;
                        }
                    }

                    terminal_side = 1u8;
                    match trav_v.next() {
                        Ok(()) => {}
                        Err(SearchResult::Empty) => break,
                        Err(SearchResult::Replacement) => {
                            has_replacement_edge = true;
                            break;
                        }
                    }
                }

                if has_replacement_edge {
                    // Case 1. If replacement edge is found, the connectivity remains unchanged.
                    //         Assume trav_u.n_vert <= trav_v.n_vert.
                    //         Promote all edges in trav_u to the next level (except the replacement edge).

                    if trav_u.n_vert > trav_v.n_vert {
                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut cu, &mut cv);
                        std::mem::swap(&mut trav_u, &mut trav_v);

                        debug_assert!(
                            trav_u.n_vert <= 1 << (self.lv_max - lv - 1),
                            "Size invariant violation"
                        );
                    }
                    trav_u.clear_markers();
                    trav_v.clear_markers();

                    let LazyTraversal {
                        visited_edges,
                        bfs_cluster,
                        ..
                    } = trav_u;

                    // Promote all edges in trav_u from lv to lv + 1
                    for [x, y] in visited_edges {
                        self.promote_edge(x, y, lv, cp);
                    }
                    self.pull_up(cp);

                    // Merge all nodes in `trav_u` from `cp`.
                    self.detach_clusters(cp, &bfs_cluster);
                    let mut cq = bfs_cluster[0];
                    for &ct in &bfs_cluster[1..] {
                        self.merge_cluster(&mut cq, ct, lv + 1);
                    }
                    if self.emask(cq) & (1 << lv + 1) != 0 {
                        // Check `verify_weak_path_compression_invariant`
                        self.materialize(&mut cq, lv + 1);
                    }

                    assert!(self.materialize(&mut cq, lv));
                    self.merge_cluster(&mut cp, cq, lv);

                    self.pull_up_to_root(cp);
                    self.verify_path_to_root(u);
                    self.verify_path_to_root(v);

                    return true;
                } else {
                    // Case 2. If there is no replacement edge in the current level,
                    //         promote the smaller component in the cluster multigraph to the next level,
                    //         And continue replacement edge search in the upper level.

                    if terminal_side == 1 {
                        // Let u be a terminal side.
                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut cu, &mut cv);
                        std::mem::swap(&mut trav_u, &mut trav_v);
                    }
                    if trav_u.n_vert > 1 << (self.lv_max - lv - 1) {
                        // Again, let u be a side that obeys the size constraint.
                        trav_v.exhaust();

                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut cu, &mut cv);
                        std::mem::swap(&mut trav_u, &mut trav_v);

                        debug_assert!(
                            trav_u.n_vert <= 1 << (self.lv_max - lv - 1),
                            "Size invariant violation"
                        );
                    }
                    trav_u.clear_markers();
                    trav_v.clear_markers();

                    let LazyTraversal {
                        visited_edges,
                        bfs_cluster,
                        ..
                    } = trav_u;

                    // Promote all edges in trav_u from lv to lv + 1
                    for [x, y] in visited_edges {
                        self.promote_edge(x, y, lv, cp);
                    }
                    self.pull_up(cp);

                    // ## Diagram
                    // lv<=i-1    lv=i        lv=i+1
                    //                  +-- |------|
                    //                  +-- |trav_u|
                    //                  +-- |------|
                    //      cg ---- cp -+
                    //                  +-- |------|
                    //                  +-- |trav_v|
                    //                  +-- |------|
                    //
                    // Detach all nodes in `trav_u` from `cp`, then merge them as `cq` at level `lv + 1`.
                    //
                    // lv<=i-1    lv=i        lv=i+1
                    //          +-----------------cq
                    //      cg -+-- cp -+-- |------|
                    //                  +-- |trav_v|
                    //                  +-- |------|

                    self.detach_clusters(cp, &bfs_cluster);
                    let mut cq = bfs_cluster[0];
                    for &ct in &bfs_cluster[1..] {
                        self.merge_cluster(&mut cq, ct, lv + 1);
                    }

                    let mut cg = self.parent(cp);
                    let mut sub_lv = !0;
                    while cg != UNSET {
                        self.pull_up(cg);
                        match cg.ty() {
                            NodeType::Leaf => unreachable!(),
                            NodeType::Local => cg = self.local[cg].parent,
                            NodeType::Cluster => {
                                sub_lv = self.cluster[cg].lv;
                                break;
                            }
                        }
                    }
                    if cg != UNSET {
                        assert!(self.materialize(&mut cq, sub_lv));
                        self.merge_cluster(&mut cg, cq, sub_lv);
                    }

                    self.try_dematerialize(&mut cp);

                    self.verify_link(u, cg);
                    self.verify_link(v, cg);

                    if cg != UNSET {
                        lv = sub_lv;
                        cp = cg;
                        cu = cq;
                        cv = cp;
                        continue;
                    } else {
                        if cg != UNSET {
                            self.pull_up(cg);
                            self.pull_up_to_root(cg);
                        }
                        self.verify_path_to_root(u);
                        self.verify_path_to_root(v);
                        return true;
                    }
                }
            }
        }

        pub fn is_connected(&self, u: u32, v: u32) -> bool {
            let ru = self.ascend_to_root(NodeRef::new(NodeType::Leaf, u));
            let rv = self.ascend_to_root(NodeRef::new(NodeType::Leaf, v));
            ru == rv
        }

        pub fn debug_mem(&self, output: &mut impl std::io::Write) {
            writeln!(output, "cluster.cap {}", self.cluster.capacity()).unwrap();
            writeln!(output, "local.cap {}", self.local.capacity()).unwrap();
        }

        pub fn debug_topo(&self) {
            let mut roots = HashSet::default();
            for u in 0..self.n_verts() {
                let leaf = NodeRef::new(NodeType::Leaf, u as u32);
                roots.insert(self.ascend_to_root(leaf));
            }

            let mut roots: Vec<_> = roots.into_iter().collect();
            roots.sort_unstable_by_key(|r| r.usize());

            fn dfs_rec(cf: &ClusterForest, u: NodeRef, depth: usize) {
                let indent = "  ".repeat(depth);

                let emask = cf.emask(u);
                let lv = match u.ty() {
                    NodeType::Local => "".into(),
                    _ => format!("    lv({:?})", cf.level(u)),
                };
                let _ = println!("{}{:?}    {:06b}    {}", indent, u, emask, lv);

                for c in cf.children(u) {
                    if c != UNSET {
                        dfs_rec(cf, c, depth + 1);
                    }
                }
            }

            for r in roots {
                println!("Root {:?}", r);
                dfs_rec(self, r, 0);
                println!();
            }
        }
    }

    enum SearchResult {
        Replacement,
        Empty,
    }

    struct LazyTraversal<'a> {
        color: u8,
        owner: &'a ClusterForest,
        level: u8,

        // BFS on cluster graph,  formed by clusters in level + 1 and edges in level.
        bfs_cluster: Vec<NodeRef>,
        timer: usize,

        // Pseudo-DFS on the current cluster tree
        tree_stack: Vec<NodeRef>,

        // Iterate adjacent edges in level
        current_leaf: Option<u32>,
        current_edge: Option<AdjSetIter<'a>>,

        // Additional aggregates
        n_vert: u32,
        visited_edges: HashSet<[u32; 2]>,
    }

    impl<'a> LazyTraversal<'a> {
        fn new(owner: &'a ClusterForest, color: u8, u: u32, level: u8) -> Self {
            let (_, ru) = owner.ascend_to_level(u, level + 1);
            let n_verts = owner.n_vert(ru);
            owner.marker(ru).set(color);

            Self {
                color,
                owner,
                level,

                bfs_cluster: vec![ru],
                timer: 0,

                tree_stack: vec![],

                current_leaf: None,
                current_edge: None,

                n_vert: n_verts,
                visited_edges: Default::default(),
            }
        }

        fn next(&mut self) -> Result<(), SearchResult> {
            loop {
                if let Some(edge_iter) = &mut self.current_edge {
                    if let Some((&v, ty)) = edge_iter.next() {
                        let u = unsafe { self.current_leaf.unwrap_unchecked() };
                        let half_edge = [u, v];

                        let (_, rv) = self.owner.ascend_to_level(v, self.level + 1);
                        let c = self.owner.marker(rv);

                        return if c.get() == self.color {
                            self.visited_edges.insert(sorted2(half_edge));

                            Ok(())
                        } else if c.get() == 0 {
                            c.set(self.color);
                            self.bfs_cluster.push(rv);
                            self.n_vert += self.owner.n_vert(rv);
                            self.visited_edges.insert(sorted2(half_edge));
                            ty.set(EdgeType::Span);

                            Ok(())
                        } else {
                            ty.set(EdgeType::Span);

                            Err(SearchResult::Replacement)
                        };
                    }

                    self.current_edge = None;
                    self.current_leaf = None;
                }

                'leaf_search: loop {
                    while let Some(r) = self.tree_stack.pop() {
                        match r.ty() {
                            NodeType::Leaf => {
                                if let Some(adj) = self.owner.leaf[r.usize()].get(self.level) {
                                    self.current_leaf = Some(r.usize() as u32);
                                    self.current_edge = Some(adj.iter());
                                    break 'leaf_search;
                                }
                            }
                            NodeType::Local => {
                                if self.owner.local[r].emask & (1 << self.level) == 0 {
                                    continue;
                                }

                                let [c0, c1] = self.owner.local[r].children;
                                if c0 != UNSET {
                                    self.tree_stack.push(c0);
                                }
                                self.tree_stack.push(c1);
                            }
                            NodeType::Cluster => {
                                if self.owner.cluster[r].emask & (1 << self.level) == 0 {
                                    continue;
                                }

                                let [c0, c1] = self.owner.cluster[r].children;
                                if c0 != UNSET {
                                    self.tree_stack.push(c0);
                                }
                                self.tree_stack.push(c1);
                            }
                        }
                    }

                    if let Some(&r) = self.bfs_cluster.get(self.timer) {
                        self.timer += 1;
                        self.tree_stack.push(r);
                        continue;
                    }

                    return Err(SearchResult::Empty);
                }
            }
        }

        fn exhaust(&mut self) {
            while self.next().is_ok() {}
        }

        fn clear_markers(&mut self) {
            for &t in &self.bfs_cluster {
                self.owner.marker(t).set(0);
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut conn = full_conn::ClusterForest::new(n);
    for _ in 0..m {
        let cmd = input.token();
        let x = input.value::<u32>() - 1;
        let y = input.value::<u32>() - 1;

        match cmd {
            "1" => {
                conn.link(x, y);
            }
            "2" => {
                conn.cut(x, y);
            }
            "3" => writeln!(output, "{}", conn.is_connected(x, y) as u8).unwrap(),
            _ => panic!(),
        }
    }
}
