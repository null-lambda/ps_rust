use std::{collections::HashMap, io::Write};

use matroid_inter::{GraphicMatroid, PartitionMatroid};

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
    type BitVec = crate::bitset::BitVec;

    const UNSET: u32 = u32::MAX;

    // A query structure for building an exchange graph, lazily if possible.
    pub trait ExchangeOracle {
        fn len(&self) -> usize;

        fn load_indep_set(&mut self, indep_set: &BitVec);

        // Test whether I U {i} is independent.
        fn can_insert(&mut self, i: usize) -> bool;

        // Test whether I - {i} + {j} is indepdendent.
        // It is recommended to cache
        fn can_exchange(&mut self, i: usize, j: usize) -> bool;
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
                if parent[v] == UNSET {
                    parent[v] = u as u32;
                    bfs.push(v as u32);
                }
            };

            if indep_set.get(u) {
                for v in 0..n {
                    if !indep_set.get(v) && m1.can_exchange(u, v) {
                        try_enqueue(v);
                    }
                }
            } else {
                for v in 0..n {
                    if indep_set.get(v) && m2.can_exchange(v, u) {
                        try_enqueue(v);
                    }
                }
            }
        }

        false
    }

    pub struct GraphicMatroid {
        pub edges: Vec<[u32; 2]>,

        pub parent: Vec<(u32, u32)>,
        pub root: Vec<u32>,
        pub depth: Vec<u32>,

        pub in_circuit: Vec<Option<BitVec>>,
    }

    impl GraphicMatroid {
        pub fn new(n_verts: usize, edges: impl IntoIterator<Item = [u32; 2]>) -> Self {
            let edges = Vec::from_iter(edges);
            let n_edges = edges.len();

            Self {
                edges,

                parent: vec![(UNSET, UNSET); n_verts],
                root: vec![UNSET; n_verts],
                depth: vec![0; n_verts],

                in_circuit: vec![None; n_edges],
            }
        }
    }

    impl ExchangeOracle for GraphicMatroid {
        fn len(&self) -> usize {
            self.edges.len()
        }

        fn load_indep_set(&mut self, indep_set: &BitVec) {
            let n_verts = self.parent.len();

            self.in_circuit.fill(None);

            let mut head = vec![0u32; n_verts + 1];
            for e in 0..self.edges.len() {
                if !indep_set.get(e) {
                    continue;
                }
                let [u, v] = self.edges[e];

                head[u as usize + 1] += 1;
                head[v as usize + 1] += 1;
            }
            for i in 2..n_verts + 1 {
                head[i] += head[i - 1];
            }

            let n_links = head[n_verts] as usize;
            let mut cursor = head[..n_verts].to_vec();
            let mut links = vec![(UNSET, UNSET); n_links];
            for e in 0..self.edges.len() {
                if !indep_set.get(e) {
                    continue;
                }
                let [u, v] = self.edges[e];

                links[cursor[u as usize] as usize] = (v, e as u32);
                cursor[u as usize] += 1;
                links[cursor[v as usize] as usize] = (u, e as u32);
                cursor[v as usize] += 1;
            }

            let neighbors = |u| &links[head[u] as usize..head[u + 1] as usize];

            let mut bfs = vec![];
            self.parent.fill((UNSET, UNSET));

            for u in 0..n_verts {
                if self.parent[u].0 != UNSET {
                    continue;
                }
                self.parent[u].0 = u as u32;
                self.root[u] = u as u32;
                self.depth[u] = 0;

                bfs.clear();
                bfs.push(u as u32);
                let mut timer = 0;
                while let Some(&u) = bfs.get(timer) {
                    timer += 1;
                    for &(v, e) in neighbors(u as usize) {
                        if !indep_set.get(e as usize) || self.parent[v as usize].0 != UNSET {
                            continue;
                        }
                        bfs.push(v);
                        self.parent[v as usize] = (u, e);
                        self.root[v as usize] = self.root[u as usize];
                        self.depth[v as usize] = self.depth[u as usize] + 1;
                    }
                }
            }
        }

        fn can_insert(&mut self, i: usize) -> bool {
            let [u, v] = self.edges[i];
            self.root[u as usize] != self.root[v as usize]
        }

        fn can_exchange(&mut self, i: usize, j: usize) -> bool {
            let [mut u, mut v] = self.edges[j];
            if self.root[u as usize] != self.root[v as usize] {
                return true;
            }

            self.in_circuit[j]
                .get_or_insert_with(|| {
                    let mut row = BitVec::with_size(self.edges.len());

                    if self.depth[u as usize] < self.depth[v as usize] {
                        std::mem::swap(&mut u, &mut v);
                    }

                    for _ in 0..self.depth[u as usize] - self.depth[v as usize] {
                        let (pu, eu) = self.parent[u as usize];
                        u = pu;
                        row.set(eu as usize, true);
                    }
                    while u != v {
                        let (pu, eu) = self.parent[u as usize];
                        let (pv, ev) = self.parent[v as usize];
                        u = pu;
                        v = pv;
                        row.set(eu as usize, true);
                        row.set(ev as usize, true);
                    }

                    row
                })
                .get(i)
        }
    }

    pub type Color = u32;
    pub type Cap = u8;

    pub struct PartitionMatroid {
        pub color: Vec<Color>,
        pub cap: Vec<Cap>,

        pub residual_cap: Vec<Cap>,
    }

    impl PartitionMatroid {
        pub fn new(colors: Vec<Color>, cap: Vec<Cap>) -> Self {
            Self {
                color: colors,
                cap: cap.clone(),

                residual_cap: cap,
            }
        }
    }

    impl ExchangeOracle for PartitionMatroid {
        fn len(&self) -> usize {
            self.color.len()
        }

        fn load_indep_set(&mut self, indep_set: &BitVec) {
            self.residual_cap = self.cap.clone();
            for i in 0..self.len() {
                if indep_set.get(i) {
                    self.residual_cap[self.color[i] as usize] -= 1;
                }
            }
        }

        fn can_insert(&mut self, i: usize) -> bool {
            self.residual_cap[self.color[i] as usize] >= 1
        }

        fn can_exchange(&mut self, i: usize, j: usize) -> bool {
            self.color[i] == self.color[j] || self.can_insert(j)
        }
    }

    //pub struct LinearMatroid {
    //    // TODO
    //}

    //pub struct GF2LinearMatroid {
    //    // TODO
    //}
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    loop {
        let r: usize = input.value();
        if r == 0 {
            return;
        }

        let mut index_map = HashMap::<u32, u32>::new();
        let mut get_index = |u: u32| {
            let i = index_map.len();
            *index_map.entry(u).or_insert_with(|| i as u32)
        };

        let edges: Vec<_> = (0..2 * r)
            .map(|_| std::array::from_fn(|_| get_index(input.value())))
            .collect();
        let n = index_map.len();
        let colors: Vec<_> = (0..r as u32 * 2).map(|c| c / 2).collect();

        let mut m1 = GraphicMatroid::new(n, edges);
        let mut m2 = PartitionMatroid::new(colors, vec![1; 2 * r]);
        let (_, rank) = matroid_inter::inter(&mut m1, &mut m2);
        let ans = rank * 2;
        writeln!(output, "{}", ans).ok();
    }
}
