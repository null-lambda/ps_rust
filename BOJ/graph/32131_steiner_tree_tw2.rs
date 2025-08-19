use std::{collections::HashMap, io::Write};

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

mod dset {
    use std::{cell::Cell, mem};

    #[derive(Clone)]
    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

pub mod map {
    use std::{hash::Hash, mem::MaybeUninit};

    pub enum AdaptiveHashSet<K, const STACK_CAP: usize> {
        Small([MaybeUninit<K>; STACK_CAP], usize),
        Large(std::collections::HashSet<K>),
    }

    impl<K, const STACK_CAP: usize> Drop for AdaptiveHashSet<K, STACK_CAP> {
        fn drop(&mut self) {
            match self {
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe { arr[i].assume_init_drop() }
                    }
                }
                _ => {}
            }
        }
    }

    impl<K: Clone, const STACK_CAP: usize> Clone for AdaptiveHashSet<K, STACK_CAP> {
        fn clone(&self) -> Self {
            match self {
                Self::Small(arr, size) => {
                    let mut cloned = std::array::from_fn(|_| MaybeUninit::uninit());
                    for i in 0..*size {
                        cloned[i] = MaybeUninit::new(unsafe { arr[i].assume_init_ref().clone() });
                    }
                    Self::Small(cloned, *size)
                }
                Self::Large(set) => Self::Large(set.clone()),
            }
        }
    }

    impl<K, const STACK_CAP: usize> Default for AdaptiveHashSet<K, STACK_CAP> {
        fn default() -> Self {
            Self::Small(std::array::from_fn(|_| MaybeUninit::uninit()), 0)
        }
    }

    impl<K: Eq + Hash, const STACK_CAP: usize> AdaptiveHashSet<K, STACK_CAP> {
        pub fn len(&self) -> usize {
            match self {
                Self::Small(_, size) => *size,
                Self::Large(set) => set.len(),
            }
        }

        pub fn contains(&self, key: &K) -> bool {
            match self {
                Self::Small(arr, size) => arr[..*size]
                    .iter()
                    .find(|&x| unsafe { x.assume_init_ref() } == key)
                    .is_some(),
                Self::Large(set) => set.contains(key),
            }
        }

        pub fn insert(&mut self, key: K) -> bool {
            if self.contains(&key) {
                return false;
            }
            match self {
                Self::Small(arr, size) => {
                    if arr[..*size]
                        .iter()
                        .find(|&x| unsafe { x.assume_init_ref() } == &key)
                        .is_some()
                    {
                        return false;
                    }

                    if *size < STACK_CAP {
                        arr[*size] = MaybeUninit::new(key);
                        *size += 1;
                    } else {
                        let arr =
                            std::mem::replace(arr, std::array::from_fn(|_| MaybeUninit::uninit()));
                        *size = 0; // Prevent `drop` call on arr elements
                        *self = Self::Large(
                            arr.into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .chain(Some(key))
                                .collect(),
                        );
                    }
                    true
                }
                Self::Large(set) => set.insert(key),
            }
        }

        pub fn remove(&mut self, key: &K) -> bool {
            match self {
                Self::Small(_, 0) => false,
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe {
                            if arr[i].assume_init_ref() == key {
                                *size -= 1;
                                arr[i].assume_init_drop();
                                arr[i] = std::mem::replace(&mut arr[*size], MaybeUninit::uninit());
                                return true;
                            }
                        }
                    }

                    false
                }
                Self::Large(set) => set.remove(key),
            }
        }

        pub fn for_each(&mut self, mut visitor: impl FnMut(&K)) {
            match self {
                Self::Small(arr, size) => {
                    arr[..*size]
                        .iter()
                        .for_each(|x| visitor(unsafe { x.assume_init_ref() }));
                }
                Self::Large(set) => set.iter().for_each(visitor),
            }
        }
    }
}

pub mod tree_decomp {
    use std::collections::VecDeque;

    // pub type HashSet<T> = std::collections::HashSet<T>;
    pub type HashSet<T> = crate::map::AdaptiveHashSet<T, 5>;

    pub const UNSET: u32 = u32::MAX;

    // Tree decomposition of treewidth 2.
    #[derive(Clone)]
    pub struct TW2 {
        // Perfect elimination ordering in the chordal completion
        pub topological_order: Vec<u32>,
        pub t_in: Vec<u32>,
        pub parents: Vec<[u32; 2]>,
    }

    impl TW2 {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = [u32; 2]>,
        ) -> Option<Self> {
            let mut neighbors = vec![HashSet::default(); n_verts];
            for [u, v] in edges {
                neighbors[u as usize].insert(v);
                neighbors[v as usize].insert(u);
            }

            let mut visited = vec![false; n_verts];
            let mut parents = vec![[UNSET; 2]; n_verts];

            let mut topological_order = vec![];
            let mut t_in = vec![UNSET; n_verts];
            let mut root = None;

            let mut queue: [_; 3] = std::array::from_fn(|_| VecDeque::new());
            for u in 0..n_verts {
                let d = neighbors[u].len();
                if d <= 2 {
                    visited[u] = true;
                    queue[d].push_back(u as u32);
                }
            }

            while let Some(u) = (0..=2).flat_map(|i| queue[i].pop_front()).next() {
                t_in[u as usize] = topological_order.len() as u32;
                topological_order.push(u);

                match neighbors[u as usize].len() {
                    0 => {
                        if let Some(old_root) = root {
                            parents[old_root as usize][0] = u;
                        }
                        root = Some(u);
                    }
                    1 => {
                        let mut p = UNSET;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| p = v);
                        neighbors[p as usize].remove(&u);

                        parents[u as usize][0] = p;

                        if !visited[p as usize] && neighbors[p as usize].len() <= 2 {
                            visited[p as usize] = true;
                            queue[neighbors[p as usize].len()].push_back(p);
                        }
                    }
                    2 => {
                        let mut ps = [UNSET; 2];
                        let mut i = 0;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| {
                            ps[i] = v;
                            i += 1;
                        });
                        let [p, q] = ps;

                        neighbors[p as usize].remove(&u);
                        neighbors[q as usize].remove(&u);

                        neighbors[p as usize].insert(q);
                        neighbors[q as usize].insert(p);

                        parents[u as usize] = [p, q];

                        for w in [p, q] {
                            if !visited[w as usize] && neighbors[w as usize].len() <= 2 {
                                visited[w as usize] = true;
                                queue[neighbors[w as usize].len()].push_back(w);
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }

            if topological_order.len() != n_verts {
                return None;
            }
            assert_eq!(root, topological_order.iter().last().copied());

            for u in 0..n_verts {
                let ps = &mut parents[u];
                if ps[1] != UNSET && t_in[ps[0] as usize] > t_in[ps[1] as usize] {
                    ps.swap(0, 1);
                }
            }

            Some(Self {
                parents,
                topological_order,
                t_in,
            })
        }
    }
}

use tree_decomp::UNSET;
const TW: usize = 2;
const K: usize = TW + 1;

type W = i64;
type Bag = [u32; K];

fn xor_bag(lhs: Bag, rhs: Bag) -> Bag {
    std::array::from_fn(|i| lhs[i] ^ rhs[i])
}

const INF: W = 2e13 as i64;

fn bag_len(bag: &Bag) -> usize {
    (0..K).find(|&i| bag[i] == UNSET).unwrap_or(K)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NaiveConnState(Vec<u8>);

impl NaiveConnState {
    fn normalized(&self) -> Self {
        let mut res = self.clone();
        let mut trans: HashMap<u8, u8> = Default::default();
        trans.insert(0, 0);
        for x in &mut res.0 {
            let l = trans.len() as u8;
            *x = *trans.entry(*x).or_insert_with(|| l);
        }
        res
    }

    fn pop(&self) -> Option<Self> {
        let mut res = self.clone();
        let last = res.0.pop().unwrap();
        let valid = last == 0 || res.0.iter().any(|&x| x == last);
        valid.then_some(res)
    }

    fn push(&self, x: u8) -> Self {
        let mut res = self.clone();
        res.0.push(x);
        res.normalized()
    }

    fn swap(&self, i: usize, j: usize) -> Self {
        let mut res = self.clone();
        res.0.swap(i, j);
        res.normalized()
    }

    fn link(&self, i: usize, j: usize) -> Option<Self> {
        let mut res = self.clone();
        if i == j {
            return Some(res);
        }

        let ci = res.0[i];
        let cj = res.0[j];
        if ci == 0 || cj == 0 || ci == cj {
            return None;
        }

        for x in &mut res.0 {
            if *x == cj {
                *x = ci;
            }
        }
        Some(res.normalized())
    }
}

type SparseMat = Vec<[u8; 2]>;
type SparseTensorD3 = Vec<[u8; 3]>;

#[derive(Default)]
struct TransLUT {
    states: [Vec<NaiveConnState>; K + 1],
    inv: [HashMap<NaiveConnState, u8>; K + 1],

    pop: [SparseMat; K + 1],
    push: [SparseMat; K],

    swap: [[[SparseMat; K]; K]; K + 1],
    link: [[[SparseMat; K]; K]; K + 1],
    join: [SparseTensorD3; K + 1],

    pivot_filter: [[Vec<u8>; K]; K + 1],
}

impl TransLUT {
    fn new() -> Self {
        let mut states: [Vec<NaiveConnState>; K + 1] = Default::default();

        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    states[3].push(NaiveConnState(vec![a, b, c]).normalized());
                }
            }
        }
        states[3].sort_unstable();
        states[3].dedup();

        for m in (0..=2).rev() {
            states[m] = states[m + 1].iter().filter_map(|s| s.pop()).collect();
            states[m].sort_unstable();
            states[m].dedup();
        }

        let inv = std::array::from_fn(|i| {
            let mut res = HashMap::new();
            for (i, x) in states[i].iter().enumerate() {
                res.entry(x.clone()).or_insert_with(|| i as u8);
            }
            res
        });

        let mut pop: [SparseMat; K + 1] = Default::default();
        let mut push: [SparseMat; K] = Default::default();
        for m in 1..=3 {
            for (iu, u) in states[m].iter().enumerate() {
                pop[m].extend(u.pop().map(|v| [iu as u8, inv[m - 1][&v]]));
            }

            for (iu, u) in states[m - 1].iter().enumerate() {
                push[m - 1].push([iu as u8, inv[m][&u.push(0)]]);
                push[m - 1].push([iu as u8, inv[m][&u.push(42)]]);
            }
        }

        let mut swap: [[[SparseMat; K]; K]; K + 1] = Default::default();
        let mut link: [[[SparseMat; K]; K]; K + 1] = Default::default();
        for m in 0..=3 {
            for i in 0..m {
                for j in 0..m {
                    for (iu, u) in states[m].iter().enumerate() {
                        swap[m][i][j].push([iu as u8, inv[m][&u.swap(i, j)]]);
                        link[m][i][j].extend(u.link(i, j).map(|v| [iu as u8, inv[m][&v]]));
                    }
                }
            }
        }

        let mut join: [SparseTensorD3; K + 1] = Default::default();
        for m in 0..=3 {
            for (iu, u) in states[m].iter().enumerate() {
                'outer: for (iv, v) in states[m].iter().enumerate() {
                    let mut dset = dset::DisjointSet::new((m + 1) * 2);
                    for j in 0..m {
                        match (u.0[j], v.0[j]) {
                            (0, 0) => {}
                            (0, _) | (_, 0) => continue 'outer,
                            _ => {
                                dset.merge(u.0[j] as usize, v.0[j] as usize + m);
                            }
                        }
                    }

                    let mut w = u.clone();
                    for x in &mut w.0 {
                        *x = dset.find_root(*x as usize) as u8;
                    }
                    w = w.normalized();
                    join[m].push([iu as u8, iv as u8, inv[m][&w]]);
                }
            }
        }

        let mut pivot_filter: [[Vec<u8>; K]; K + 1] = Default::default();
        for m in 0..=3 {
            for p in 0..m {
                for (iu, u) in states[m].iter().enumerate() {
                    if u.0[p as usize] == 0 {
                        pivot_filter[m][p].push(iu as u8);
                    }
                }
            }
        }

        Self {
            states,
            inv,

            pop,
            push,

            swap,
            link,

            pivot_filter,

            join,
        }
    }
}

const MAX_STATES: usize = 15;
type Dp = [W; MAX_STATES];

struct Cx {
    edges: HashMap<[u32; 2], W>,
    is_pivot: Vec<bool>,
}

#[derive(Clone, Copy, Debug)]
struct NodeAgg {
    bag: Bag,
    dp: Dp,
}

// Let us introduce some nice tree decompositions...
impl NodeAgg {
    fn bag_len(&self) -> usize {
        bag_len(&self.bag)
    }

    fn empty() -> Self {
        let mut dp = [INF; MAX_STATES];
        dp[0] = 0;
        Self {
            bag: [UNSET; 3],
            dp,
        }
    }

    fn new(bag: Bag, lut: &TransLUT, cx: &Cx, link: bool) -> Self {
        let m = bag_len(&bag);
        let mut res = Self::empty();
        for i in 0..m {
            res.push(&lut, bag[i]);
        }

        if link {
            for i in 0..m {
                for j in i + 1..m {
                    res.link(&lut, &cx, i, j);
                }
            }
        }

        res
    }

    // Apply sparse matrix
    fn transform_dp(&mut self, trans: &[[u8; 2]], f: impl Fn(&W) -> W) {
        let mut next = [INF; MAX_STATES];
        for &[u, v] in trans {
            next[v as usize] = next[v as usize].min(f(&self.dp[u as usize]));
        }
        self.dp = next
    }

    // Process forget node
    fn pop(&mut self, lut: &TransLUT) {
        let m = self.bag_len();
        self.bag[m - 1] = UNSET;
        self.transform_dp(&lut.pop[m], |&x| x);
    }

    // Process introduce node
    fn push(&mut self, lut: &TransLUT, u: u32) {
        let m = self.bag_len();
        self.bag[m] = u;
        self.transform_dp(&lut.push[m], |&x| x);
    }

    // Our representation of bag is ordered.
    fn swap(&mut self, lut: &TransLUT, i: usize, j: usize) {
        let m = self.bag_len();
        if i == j {
            return;
        }

        self.bag.swap(i, j);
        self.transform_dp(&lut.swap[m][i][j], |&x| x);
    }

    fn link(&mut self, lut: &TransLUT, cx: &Cx, i: usize, j: usize) {
        let m = self.bag_len();
        if i == j {
            return;
        }

        let u = self.bag[i];
        let v = self.bag[j];
        let Some(&w) = cx.edges.get(&sorted2([u, v])) else {
            return;
        };

        let mut next = self.dp.clone();
        let trans = &lut.link[m][i][j];
        for &[u, v] in trans {
            next[v as usize] = next[v as usize].min(&self.dp[u as usize] + w);
        }
        self.dp = next;
    }

    fn pivot_filter(&mut self, lut: &TransLUT, cx: &Cx) {
        let m = self.bag_len();
        for i in 0..m {
            if cx.is_pivot[self.bag[i] as usize] {
                for &x in &lut.pivot_filter[m][i] {
                    self.dp[x as usize] = INF;
                }
            }
        }
    }

    // Process join node
    fn join(&mut self, lut: &TransLUT, other: Self) {
        assert_eq!(self.bag, other.bag);
        let m = self.bag_len();

        let mut next = [INF; MAX_STATES];
        for &[iu, iv, iw] in &lut.join[m] {
            next[iw as usize] = next[iw as usize].min(self.dp[iu as usize] + other.dp[iv as usize]);
        }
        self.dp = next;
    }

    fn lift(&mut self, lut: &TransLUT, cx: &Cx, target: Bag) {
        for i in (0..self.bag_len()).rev() {
            let u = self.bag[i];
            if target.contains(&u) {
                continue;
            }

            let m = self.bag_len();
            self.swap(&lut, i, m - 1);
            self.pop(&lut);
        }

        for i in 0..bag_len(&target) {
            let v = target[i];

            if let Some(j) = self.bag.iter().position(|&u| u as u32 == v) {
                self.swap(&lut, i, j);
                continue;
            }

            let m = self.bag_len();
            self.push(&lut, v);
            for j in 0..m {
                self.link(&lut, &cx, j, m);
            }
            self.swap(&lut, i, m)
        }
    }

    fn mul(&mut self, factor: W) {
        for x in &mut self.dp {
            *x *= factor;
        }
    }
}

fn sorted2<T: PartialOrd>(mut xs: [T; 2]) -> [T; 2] {
    if xs[0] > xs[1] {
        xs.swap(0, 1);
    }
    xs
}

fn xor_traversal(
    mut degree: Vec<u32>,
    mut xor_neighbors: Vec<u32>,
    root: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = degree.len();
    degree[root as usize] += 2;

    let mut toposort = vec![];

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            xor_neighbors[p as usize] ^= u;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            toposort.push(u);

            u = p;
        }
    }
    toposort.push(root);

    let mut parent = xor_neighbors;
    parent[root as usize] = root;
    (toposort, parent)
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let edges: HashMap<[u32; 2], W> = (0..m)
        .map(|_| {
            (
                sorted2([input.value::<u32>() - 1, input.value::<u32>() - 1]),
                input.value(),
            )
        })
        .collect();

    let k: usize = input.value();
    let mut is_pivot = vec![false; n];
    let mut root = UNSET;
    for _ in 0..k {
        let u = input.value::<u32>() - 1;
        is_pivot[u as usize] = true;
        root = u;
    }
    root = 5;

    let td = tree_decomp::TW2::from_edges(n, edges.keys().copied()).unwrap();

    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];
    degree[root as usize] += 2;
    for u in 0..n as u32 {
        let p = td.parents[u as usize][0];
        if p == UNSET {
            continue;
        }
        degree[p as usize] += 1;
        degree[u as usize] += 1;
        xor_neighbors[u as usize] ^= p;
        xor_neighbors[p as usize] ^= u;
    }

    let (toposort, parent) = xor_traversal(degree, xor_neighbors, root);
    assert!(toposort.len() == n);

    let mut indegree = vec![0u32; n];
    for u in 0..n as u32 {
        let p = parent[u as usize];
        if p == u {
            continue;
        }
        indegree[p as usize] += 1;
    }

    let lut = TransLUT::new();
    let cx = Cx { edges, is_pivot };
    let mut dp: Vec<_> = (0..n)
        .map(|u| {
            let mut bag = [UNSET; K];
            bag[0] = u as u32;
            bag[1] = td.parents[u][0];
            bag[2] = td.parents[u][1];

            NodeAgg::new(bag, &lut, &cx, indegree[u as usize] == 0)
        })
        .collect();

    // for m in 0..=3 {
    //     println!("states[{m}]: {:?}", lut.states[m]);
    // }

    // println!("{:?}", root);
    // println!("parent {:?}", parent);
    // println!("indegree {:?}", indegree);
    // println!("toposort {:?}", toposort);

    for &u in &toposort {
        dp[u as usize].pivot_filter(&lut, &cx);
        // println!("dp {:?}", dp[u as usize]);

        let p = parent[u as usize];
        if u == p {
            break;
        }

        let mut dp_u = dp[u as usize];
        dp_u.lift(&lut, &cx, dp[p as usize].bag);
        // println!("    lifted {:?}", dp_u);

        dp[p as usize].join(&lut, dp_u);
    }

    // println!("{:?}", dp[root as usize]);

    let mut dp_root = dp[root as usize];
    let m = dp_root.bag_len();
    let i = dp_root.bag.iter().position(|&u| u == root).unwrap();
    dp_root.swap(&lut, i, 0);
    for _ in 0..m - 1 {
        dp_root.pop(&lut);
    }

    let ans = dp_root.dp[lut.inv[1][&NaiveConnState(vec![1])] as usize];
    writeln!(output, "{}", ans).unwrap();
}
