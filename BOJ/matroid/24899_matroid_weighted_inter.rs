use std::io::Write;

use buffered_io::BufReadExt;
use dset::DisjointSet;
use matroid_inter::{BitVec, ExchangeOracle};

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

mod dset {
    #[derive(Clone)]
    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        link: Vec<i32>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self { link: vec![-1; n] }
        }

        pub fn root_with_size(&mut self, u: u32) -> (u32, u32) {
            let p = self.link[u as usize];
            if p >= 0 {
                let (root, size) = self.root_with_size(p as u32);
                self.link[u as usize] = root as i32;
                (root, size)
            } else {
                (u, (-p) as u32)
            }
        }

        pub fn root(&mut self, u: u32) -> u32 {
            self.root_with_size(u).0
        }

        pub fn merge(&mut self, u: u32, v: u32) -> bool {
            let (mut u, size_u) = self.root_with_size(u);
            let (mut v, size_v) = self.root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                std::mem::swap(&mut u, &mut v);
            }
            self.link[v as usize] = u as i32;
            self.link[u as usize] = -((size_u + size_v) as i32);
            true
        }
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
    use std::collections::VecDeque;

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

    fn ascend_to_root(parent: &[u32], mut u: usize, mut visitor: impl FnMut(usize)) {
        loop {
            visitor(u);

            if u == parent[u] as usize {
                break;
            }
            u = parent[u] as usize;
        }
    }

    pub fn inter(m1: &mut impl ExchangeOracle, m2: &mut impl ExchangeOracle) -> (BitVec, usize) {
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

        assert_eq!(m1.len(), m2.len());
        let mut set = BitVec::with_size(m1.len());
        let mut rank = 0;
        while augment(m1, m2, &mut set) {
            rank += 1;
        }
        (set, rank)
    }

    pub type W = i64;
    pub const W_INF: W = W::MAX / 3;

    pub fn inter_max_weight(
        m1: &mut impl ExchangeOracle,
        m2: &mut impl ExchangeOracle,
        weights: &[W],
        mut yield_weight: impl FnMut(W),
    ) -> (BitVec, usize) {
        fn augment_spfa(
            m1: &mut impl ExchangeOracle,
            m2: &mut impl ExchangeOracle,
            weights: &[W],
            indep_set: &mut BitVec,
        ) -> Option<W> {
            let n = m1.len();
            m1.load_indep_set(&indep_set);
            m2.load_indep_set(&indep_set);

            let mut bfs = vec![];
            let mut parent = vec![UNSET; n];

            let mut spfa_queue = VecDeque::new();
            let mut on_queue = vec![false; n];
            let mut dist = vec![(W_INF, 0u32); n];

            for u in 0..n {
                if !indep_set.get(u) && m1.can_insert(u) {
                    bfs.push(u as u32);
                    parent[u] = u as u32;
                    dist[u] = (-weights[u], 0);

                    spfa_queue.push_back(u as u32);
                    on_queue[u as usize] = true;
                }
            }

            let is_dest: Vec<bool> = (0..n)
                .map(|u| !indep_set.get(u) && m2.can_insert(u))
                .collect();

            let mut timer = 0;
            let mut adj = vec![vec![]; n];
            while let Some(u) = bfs.get(timer).map(|&u| u as usize) {
                timer += 1;

                let mut try_enqueue = |v| {
                    if parent[v] == UNSET {
                        parent[v] = u as u32;
                        bfs.push(v as u32);
                    }
                };

                m1.left_exchange(&indep_set, u, |v| {
                    try_enqueue(v);
                    assert!(!indep_set.get(v));
                    adj[u].push((v as u32, -weights[v]));
                });
                m2.right_exchange(&indep_set, u, |v| {
                    try_enqueue(v);
                    assert!(indep_set.get(v));
                    adj[u].push((v as u32, weights[v]));
                });
            }

            while let Some(u) = spfa_queue.pop_front() {
                on_queue[u as usize] = false;

                for &(v, w) in &adj[u as usize] {
                    let dv_new = (dist[u as usize].0 + w, dist[u as usize].1 + 1);
                    if dv_new < dist[v as usize] {
                        dist[v as usize] = dv_new;
                        parent[v as usize] = u;

                        if !on_queue[v as usize] {
                            spfa_queue.push_back(v);
                            on_queue[v as usize] = true;
                        }
                    }
                }
            }

            let u = (0..n)
                .filter(|&u| is_dest[u] && dist[u].0 < W_INF)
                .min_by_key(|&u| dist[u])?;
            ascend_to_root(&parent, u as usize, |u| indep_set.toggle(u));
            Some(-dist[u].0)
        }

        assert_eq!(m1.len(), m2.len());
        assert_eq!(m1.len(), weights.len());
        let mut set = BitVec::with_size(m1.len());
        let mut rank = 0;

        let mut w = 0;
        while let Some(dw) = augment_spfa(m1, m2, &weights, &mut set) {
            rank += 1;
            w += dw;
            yield_weight(w);
        }
        (set, rank)
    }
}

struct M1 {
    edges: Vec<[u32; 2]>,
    color: Vec<u32>,
    cap: Vec<u32>,

    conn: DisjointSet,
    residual: Vec<u32>,
}

impl M1 {
    fn new(edges: Vec<[u32; 2]>, color: Vec<u32>, cap: Vec<u32>) -> Self {
        Self {
            edges,
            color,
            cap,

            conn: DisjointSet::new(0),
            residual: vec![],
        }
    }
}

impl ExchangeOracle for M1 {
    fn len(&self) -> usize {
        self.edges.len()
    }

    fn load_indep_set(&mut self, indep_set: &BitVec) {
        let n = self.cap.len();
        self.conn = DisjointSet::new(n);
        self.residual = self.cap.clone();
        for e in 0..self.len() {
            if !indep_set.get(e) {
                continue;
            }

            let [u, v] = self.edges[e];
            self.conn.merge(u, v);
            self.residual[self.color[e as usize] as usize] -= 1;
        }
    }

    fn can_insert(&mut self, i: usize) -> bool {
        let [u, v] = self.edges[i];
        self.conn.root(u) != self.conn.root(v)
            && self.residual[self.color[i as usize] as usize] >= 1
    }

    // Assuming i in I, visit all exchangable j.
    fn left_exchange(&mut self, indep_set: &BitVec, i: usize, mut visitor: impl FnMut(usize)) {
        if !indep_set.get(i) {
            return;
        }

        let n = self.cap.len();
        let mut conn = DisjointSet::new(n);
        for e in 0..self.len() {
            if !indep_set.get(e) || e == i {
                continue;
            }

            let [u, v] = self.edges[e];
            conn.merge(u, v);
        }

        for j in 0..self.len() {
            if indep_set.get(j) {
                continue;
            }

            let [u, v] = self.edges[j];
            if conn.root(u) != conn.root(v)
                && (self.color[i as usize] == self.color[j as usize]
                    || self.residual[self.color[j as usize] as usize] >= 1)
            {
                visitor(j);
            }
        }
    }
}

struct M2 {
    color: Vec<u32>,
    cap: Vec<u32>,

    residual: Vec<u32>,
}

impl M2 {
    fn new(color: Vec<u32>, cap: Vec<u32>) -> Self {
        Self {
            color,
            cap,

            residual: vec![],
        }
    }
}

impl ExchangeOracle for M2 {
    fn len(&self) -> usize {
        self.color.len()
    }

    fn load_indep_set(&mut self, indep_set: &BitVec) {
        self.residual = self.cap.clone();
        for e in 0..self.len() {
            if !indep_set.get(e) {
                continue;
            }

            self.residual[self.color[e as usize] as usize] -= 1;
        }
    }

    fn can_insert(&mut self, i: usize) -> bool {
        self.residual[self.color[i as usize] as usize] >= 1
    }

    fn can_exchange(&mut self, i: usize, j: usize) -> bool {
        self.color[i as usize] == self.color[j as usize]
            || self.residual[self.color[j as usize] as usize] >= 1
    }
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let x: usize = input.value();
    let y: usize = input.value();

    let inf_cap = 1 << 30;

    let mut rs = vec![];
    for _ in 0..x {
        let u = input.value::<u32>() - 1;
        rs.push(u);
    }

    let mut rcap = vec![inf_cap; n];
    for u in rs {
        rcap[u as usize] = input.value::<u32>();
    }

    let mut lcap = vec![inf_cap; n];
    for _ in 0..y {
        let u = input.value::<u32>() - 1;
        lcap[u as usize] = 1;
    }

    let mut edges = vec![];
    let mut weights = vec![];
    let mut lcolor = vec![];
    let mut rcolor = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w = input.value::<i64>();
        if lcap[u as usize] < inf_cap && lcap[v as usize] < inf_cap
            || rcap[u as usize] < inf_cap && rcap[v as usize] < inf_cap
        {
            continue;
        }

        edges.push([u, v]);
        weights.push(-w);
        lcolor.push(if lcap[u as usize] < inf_cap { u } else { v });
        rcolor.push(if rcap[u as usize] < inf_cap { u } else { v });
    }

    let mut mat1 = M1::new(edges, lcolor, lcap);
    let mut mat2 = M2::new(rcolor, rcap);
    let mut ws = vec![];
    let (_set, _rank) =
        matroid_inter::inter_max_weight(&mut mat1, &mut mat2, &weights, |w| ws.push(w));
    ws.resize(n - 1, 1);
    for i in 0..n - 1 {
        writeln!(output, "{}", -ws[i]).unwrap();
    }
}
