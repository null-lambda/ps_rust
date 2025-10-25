use std::io::Write;

use bitset::BitVec;
use matroid_inter::{ExchangeOracle, UNSET};

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
    use std::cell::Cell;

    thread_local! {
        static WORK: Cell<u64> = Cell::new(0);
    }

    pub fn work() {
        WORK.with(|work| work.set(work.get() + 1));
    }

    pub fn get_work() -> u64 {
        WORK.with(|work| work.get())
    }
}

pub mod bitset {
    pub type B = u64;
    pub const BLOCK_BITS: usize = B::BITS as usize;

    #[derive(Clone)]
    pub struct BitVec {
        masks: Vec<B>,
        size: usize,
    }

    impl BitVec {
        pub fn len(&self) -> usize {
            self.size
        }

        pub fn with_size(n: usize) -> Self {
            Self {
                masks: vec![B::default(); n.div_ceil(BLOCK_BITS)],
                size: n,
            }
        }

        pub fn get(&self, i: usize) -> bool {
            assert!(i < self.size);
            let (b, s) = (i / BLOCK_BITS, i % BLOCK_BITS);
            (self.masks[b] >> s) & 1 != 0
        }

        pub fn set(&mut self, i: usize, value: bool) {
            assert!(i < self.size);
            let (b, s) = (i / BLOCK_BITS, i % BLOCK_BITS);
            if !value {
                self.masks[b] &= !(1 << s);
            } else {
                self.masks[b] |= 1 << s;
            }
        }

        pub fn toggle(&mut self, i: usize) {
            assert!(i < self.size);
            let (b, s) = (i / BLOCK_BITS, i % BLOCK_BITS);
            self.masks[b] ^= 1 << s;
        }

        pub fn count_ones(&self) -> u32 {
            self.masks.iter().map(|&m| m.count_ones()).sum()
        }
    }

    impl FromIterator<bool> for BitVec {
        fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
            let iter = iter.into_iter();
            let (lower, upper) = iter.size_hint();
            let mut masks = Vec::with_capacity(upper.unwrap_or(lower).div_ceil(BLOCK_BITS));

            let mut mask = B::default();
            let mut s = 0;
            let mut size = 0;
            for bit in iter {
                if bit {
                    mask |= 1 << s;
                }
                s += 1;

                if s == BLOCK_BITS {
                    masks.push(mask);
                    size += s;
                    mask = 0;
                    s = 0;
                }
            }
            if s != 0 {
                size += s;
                masks.push(mask);
            }

            BitVec { masks, size }
        }
    }

    impl std::fmt::Debug for BitVec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[")?;
            for i in 0..self.size {
                write!(f, "{}", self.get(i) as u8)?;
            }
            write!(f, "]")?;
            Ok(())
        }
    }
}

pub mod matroid_inter {
    pub(crate) type BitVec = crate::bitset::BitVec;

    pub const UNSET: u32 = u32::MAX;

    // An abstract query structure for building an exchange graph.
    // Use lazy or amortized evaluation if possible.
    pub trait ExchangeOracle {
        fn len(&self) -> usize;

        fn load_indep_set(&mut self, indep_set: &BitVec);

        // Test whether I U {i} is independent.
        fn can_insert(&mut self, i: usize) -> bool;

        // Test whether I - {i} + {j} is indepdendent.
        fn can_exchange(&mut self, _i: usize, _j: usize) -> bool {
            unimplemented!()
        }

        // Assuming i in I, visit all exchangable j.
        fn left_exchange(&mut self, indep_set: &BitVec, i: usize, mut visitor: impl FnMut(usize)) {
            if !indep_set.get(i) {
                return;
            }
            for j in 0..self.len() {
                if !indep_set.get(j) && self.can_exchange(i, j) {
                    visitor(j);
                }
            }
        }

        // Assuming j not in I, visit all exchangable j.
        fn right_exchange(&mut self, indep_set: &BitVec, j: usize, mut visitor: impl FnMut(usize)) {
            if indep_set.get(j) {
                return;
            }
            for i in 0..self.len() {
                if indep_set.get(i) && self.can_exchange(i, j) {
                    visitor(i);
                }
            }
        }
    }

    pub fn inter(m1: &mut impl ExchangeOracle, m2: &mut impl ExchangeOracle) -> (BitVec, usize) {
        assert_eq!(m1.len(), m2.len());
        let mut set = BitVec::with_size(m1.len());
        let mut rank = 0;
        while augment(m1, m2, &mut set) {
            rank += 1;
        }
        (set, rank)
    }

    fn ascend_to_root(parent: &[u32], mut u: usize, mut visitor: impl FnMut(usize)) {
        loop {
            visitor(u);

            if u == parent[u] as usize {
                break;
            }
            u = parent[u] as usize;
        }
    }

    fn augment(
        m1: &mut impl ExchangeOracle,
        m2: &mut impl ExchangeOracle,
        indep_set: &mut BitVec,
    ) -> bool {
        let n = m1.len();
        m1.load_indep_set(&indep_set);
        m2.load_indep_set(&indep_set);

        let mut parent = vec![UNSET; n];
        let mut bfs = vec![];
        for i in 0..n {
            if !indep_set.get(i) && m1.can_insert(i) {
                bfs.push(i as u32);
                parent[i] = i as u32;
            }
        }

        let is_dest: Vec<bool> = (0..n)
            .map(|i| !indep_set.get(i) && m2.can_insert(i))
            .collect();
        let mut timer = 0;

        while let Some(u) = bfs.get(timer).map(|&u| u as usize) {
            timer += 1;

            if is_dest[u] {
                ascend_to_root(&parent, u as usize, |u| indep_set.toggle(u));
                return true;
            }

            let mut try_enqueue = |v| {
                super::debug::work();
                if parent[v] == UNSET {
                    parent[v] = u as u32;
                    bfs.push(v as u32);
                }
            };

            m1.left_exchange(&indep_set, u, |v| try_enqueue(v));
            m2.right_exchange(&indep_set, u, |v| try_enqueue(v));
        }

        false
    }
}

#[derive(Clone)]
pub struct DisjointForest {
    link: Vec<i32>,
}

impl DisjointForest {
    pub fn new(n: usize) -> Self {
        Self { link: vec![-1; n] }
    }

    pub fn root(&mut self, u: u32) -> u32 {
        let p = self.link[u as usize];
        if p >= 0 {
            let root = self.root(p as u32);
            self.link[u as usize] = root as i32;
            root
        } else {
            u
        }
    }

    pub fn link(&mut self, mut u: u32, mut p: u32) -> bool {
        u = self.root(u);
        p = self.root(p);

        self.link[u as usize] = p as i32;
        self.link[p as usize] = -1;
        true
    }
}

pub struct BridgeCover {
    pub edges: Vec<[u32; 2]>,

    pub parent: Vec<u32>,
    pub parent_edge: Vec<u32>,
    pub depth: Vec<u32>,
    pub root: Vec<u32>,

    pub bccs: DisjointForest,
}

impl BridgeCover {
    fn new(n_verts: usize, edges: Vec<[u32; 2]>) -> Self {
        Self {
            edges,

            parent: vec![UNSET; n_verts],
            parent_edge: vec![UNSET; n_verts],
            depth: vec![UNSET; n_verts],
            root: vec![UNSET; n_verts],

            bccs: DisjointForest::new(n_verts),
        }
    }

    fn build(
        &mut self,
        include_edge: impl Fn(u32) -> bool,
        mut notify_covered_edge: impl FnMut(u32),
    ) {
        let n_verts = self.parent.len();

        let mut head = vec![0u32; n_verts + 1];
        for e in 0..self.edges.len() as u32 {
            if !include_edge(e) {
                continue;
            }
            let [u, v] = self.edges[e as usize];

            head[u as usize + 1] += 1;
            head[v as usize + 1] += 1;
        }
        for i in 2..n_verts + 1 {
            head[i] += head[i - 1];
        }

        let n_links = head[n_verts] as usize;
        let mut cursor = head[..n_verts].to_vec();
        let mut links = vec![(UNSET, UNSET); n_links];
        for e in 0..self.edges.len() as u32 {
            if !include_edge(e) {
                continue;
            }
            let [u, v] = self.edges[e as usize];

            links[cursor[u as usize] as usize] = (v, e);
            cursor[u as usize] += 1;
            links[cursor[v as usize] as usize] = (u, e);
            cursor[v as usize] += 1;
        }

        self.parent.fill(UNSET);

        let mut bfs = vec![];
        let mut timer = 0;
        let mut non_tree_edges = vec![];
        for r in 0..n_verts as u32 {
            if self.parent[r as usize] != UNSET {
                continue;
            }

            self.parent[r as usize] = r;
            self.parent_edge[r as usize] = UNSET;
            self.depth[r as usize] = 0;
            bfs.push(r);

            while let Some(&u) = bfs.get(timer as usize) {
                self.root[u as usize] = r;
                timer += 1;

                for ie in head[u as usize]..head[u as usize + 1] {
                    let (v, e) = links[ie as usize];
                    if e == self.parent_edge[u as usize] {
                        continue;
                    }
                    if self.parent[v as usize] == UNSET {
                        self.parent[v as usize] = u;
                        self.parent_edge[v as usize] = e;
                        self.depth[v as usize] = self.depth[u as usize] + 1;
                        bfs.push(v);
                    } else if u < v
                    /* Loose tie-breaking. Self-loops? */
                    {
                        non_tree_edges.push(e);
                    }
                }
            }
        }

        self.bccs = DisjointForest::new(n_verts);
        for e in non_tree_edges {
            self.cover(e, &mut notify_covered_edge);
        }
    }

    fn is_connected(&self, u: u32, v: u32) -> bool {
        self.root[u as usize] == self.root[v as usize]
    }

    fn cover(&mut self, e: u32, mut notify_covered_edge: impl FnMut(u32)) -> bool {
        let [mut u, mut v] = self.edges[e as usize];
        if !self.is_connected(u, v) {
            return false;
        }

        u = self.bccs.root(u);
        v = self.bccs.root(v);
        while u != v {
            if self.depth[u as usize] < self.depth[v as usize] {
                std::mem::swap(&mut u, &mut v);
            }

            let p = self.parent[u as usize];
            notify_covered_edge(self.parent_edge[u as usize]);

            self.bccs.link(u, p);

            let rm = self.bccs.root(u);
            u = rm;
        }

        true
    }
}

pub struct LeftCographicMatroid {
    pub inner: BridgeCover,
    pub lazy_yield: Vec<u32>,
}

impl LeftCographicMatroid {
    pub fn new(n_verts: usize, edges: Vec<[u32; 2]>) -> Option<Self> {
        let mut this = Self {
            inner: BridgeCover::new(n_verts, edges),

            lazy_yield: vec![],
        };

        this.inner.build(|_| true, |_| {});
        if (1..n_verts).any(|v| !this.inner.is_connected(0, v as u32)) {
            return None;
        }

        Some(this)
    }
}

impl ExchangeOracle for LeftCographicMatroid {
    fn len(&self) -> usize {
        self.inner.edges.len()
    }

    fn load_indep_set(&mut self, indep_set: &BitVec) {
        self.inner
            .build(|e| !indep_set.get(e as usize), |e| self.lazy_yield.push(e));
        self.lazy_yield.clear();
    }

    fn can_insert(&mut self, i: usize) -> bool {
        let [u, v] = self.inner.edges[i];
        self.inner.bccs.root(u) == self.inner.bccs.root(v)
    }

    fn left_exchange(&mut self, indep_set: &BitVec, i: usize, mut visitor: impl FnMut(usize)) {
        if !indep_set.get(i) {
            return;
        }

        while let Some(j) = self.lazy_yield.pop() {
            assert!(!indep_set.get(j as usize));
            visitor(j as usize);
        }
        self.inner.cover(i as u32, |e| visitor(e as usize));
    }
}

pub struct RightGraphicMatroid {
    pub inner: BridgeCover,
    pub yielded_all: bool,
}

impl RightGraphicMatroid {
    pub fn new(n_verts: usize, edges: Vec<[u32; 2]>) -> Self {
        Self {
            inner: BridgeCover::new(n_verts, edges),
            yielded_all: false,
        }
    }
}

impl ExchangeOracle for RightGraphicMatroid {
    fn len(&self) -> usize {
        self.inner.edges.len()
    }

    fn load_indep_set(&mut self, indep_set: &BitVec) {
        self.inner
            .build(|e| indep_set.get(e as usize), |_e| panic!());
        self.yielded_all = false;
    }

    fn can_insert(&mut self, i: usize) -> bool {
        let [u, v] = self.inner.edges[i];
        self.inner.root[u as usize] != self.inner.root[v as usize]
    }

    fn right_exchange(&mut self, indep_set: &BitVec, j: usize, mut visitor: impl FnMut(usize)) {
        if indep_set.get(j) || self.yielded_all {
            return;
        }

        if !self.inner.cover(j as u32, |e| visitor(e as usize)) {
            self.yielded_all = true;
            for i in 0..self.len() {
                if indep_set.get(i) {
                    visitor(i);
                }
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m = 2 * n - 2;
    let edges = (0..2 * n - 2)
        .map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1])
        .collect::<Vec<_>>();

    let Some(mut m1) = LeftCographicMatroid::new(n, edges.clone()) else {
        writeln!(output, "NO").ok();
        return;
    };
    let mut m2 = RightGraphicMatroid::new(n, edges);
    let (set, rank) = matroid_inter::inter(&mut m1, &mut m2);
    if rank == n - 1 {
        writeln!(output, "YES").ok();

        let ans = unsafe {
            String::from_utf8_unchecked(
                (0..m)
                    .map(|e| if set.get(e) { b'R' } else { b'B' })
                    .collect::<Vec<_>>(),
            )
        };
        writeln!(output, "{}", ans).ok();
    } else {
        writeln!(output, "NO").ok();
    }

    eprintln!("total work: {:?}", debug::get_work());
}
