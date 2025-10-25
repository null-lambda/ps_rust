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
