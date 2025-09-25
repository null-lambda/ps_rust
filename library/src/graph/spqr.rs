pub mod spqr {
    /// Static triconnectivity for **simple** graphs via SPQR Tree
    ///
    /// ## References
    /// J. E. Hopcroft and R. E. Tarjan, 'Dividing a Graph into Triconnected Components', 1973.
    /// C. Gutwenger and P. Mutzel, 'A linear time implementation of SPQR-trees', 2000.
    /// C. Gutwenger, 'Application of SPQR-trees in the planarization approach for drawing graphs', 2010.
    use self::jagged::CSR;
    use self::linked_list::CyclicListPool;
    use std::cmp::Ordering;

    pub const UNSET: u32 = u32::MAX;
    pub const INACTIVE: u32 = u32::MAX - 1;
    const INF: u32 = u32::MAX;
    const TSTACK_SEP: [u32; 3] = [UNSET; 3];

    // Sort with a key in 0..key_bound
    pub fn bucket_sort<T: Copy + Default>(
        xs: &mut [T],
        key_bound: u32,
        mut key: impl FnMut(&T) -> u32,
    ) {
        const UNSET: u32 = u32::MAX;

        // Forward-star
        let mut head = vec![UNSET; key_bound as usize];
        let mut link = vec![(UNSET, T::default()); xs.len()];

        for (ix, &x) in xs.iter().enumerate().rev() {
            let k = key(&x);
            debug_assert!(k < key_bound);

            link[ix as usize] = (head[k as usize], x);
            head[k as usize] = ix as u32;
        }

        let mut i = 0;
        for k in 0..key_bound {
            let mut u = head[k as usize];
            while u != UNSET {
                let (u_next, x) = link[u as usize];
                xs[i] = x;
                i += 1;

                u = u_next;
            }
        }
    }

    fn inv_perm(perm: &[u32]) -> Vec<u32> {
        let mut res = vec![UNSET; perm.len()];
        for u in 0..perm.len() as u32 {
            res[perm[u as usize] as usize] = u;
        }
        res
    }

    pub mod jagged {
        use std::fmt::Debug;
        use std::iter::FromIterator;
        use std::mem::MaybeUninit;
        use std::ops::{Index, IndexMut};

        // Compressed sparse row format, for static jagged array
        // Provides good locality for graph traversal
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
        pub struct CSR<T> {
            pub links: Vec<T>,
            head: Vec<u32>,
        }

        impl<T> Default for CSR<T> {
            fn default() -> Self {
                Self {
                    links: vec![],
                    head: vec![0],
                }
            }
        }

        impl<T: Clone> CSR<T> {
            pub fn from_pairs_rstable(
                n: usize,
                pairs: impl Iterator<Item = (u32, T)> + Clone,
            ) -> Self {
                let mut head = vec![0u32; n + 1];

                for (u, _) in pairs.clone() {
                    debug_assert!(u < n as u32);
                    head[u as usize] += 1;
                }
                for i in 0..n {
                    head[i + 1] += head[i];
                }
                let mut data: Vec<_> = (0..head[n]).map(|_| MaybeUninit::uninit()).collect();

                for (u, v) in pairs {
                    head[u as usize] -= 1;
                    data[head[u as usize] as usize] = MaybeUninit::new(v.clone());
                }

                // Rustc is likely to perform in‑place iteration without new allocation.
                // [https://doc.rust-lang.org/stable/std/iter/trait.FromIterator.html#impl-FromIterator%3CT%3E-for-Vec%3CT%3E]
                let data = data
                    .into_iter()
                    .map(|x| unsafe { x.assume_init() })
                    .collect();

                CSR { links: data, head }
            }
        }

        impl<T, I> FromIterator<I> for CSR<T>
        where
            I: IntoIterator<Item = T>,
        {
            fn from_iter<J>(iter: J) -> Self
            where
                J: IntoIterator<Item = I>,
            {
                let mut data = vec![];
                let mut head = vec![];
                head.push(0);

                let mut cnt = 0;
                for row in iter {
                    data.extend(row.into_iter().inspect(|_| cnt += 1));
                    head.push(cnt);
                }
                CSR { links: data, head }
            }
        }

        impl<T> CSR<T> {
            pub fn len(&self) -> usize {
                self.head.len() - 1
            }

            pub fn edge_range(&self, index: usize) -> std::ops::Range<usize> {
                self.head[index] as usize..self.head[index as usize + 1] as usize
            }

            pub fn push(&mut self) {
                self.head.push(*self.head.last().unwrap());
            }

            pub fn push_to_last_row(&mut self, v: T) {
                assert!(self.len() >= 1);
                *self.head.last_mut().unwrap() += 1;
                self.links.push(v);
            }
        }

        impl<T> Index<usize> for CSR<T> {
            type Output = [T];

            fn index(&self, index: usize) -> &Self::Output {
                &self.links[self.edge_range(index)]
            }
        }

        impl<T> IndexMut<usize> for CSR<T> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                let es = self.edge_range(index);
                &mut self.links[es]
            }
        }

        impl<T> Debug for CSR<T>
        where
            T: Debug,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
                v.fmt(f)
            }
        }
    }

    #[allow(unused)]
    mod linked_list {
        // Arena-allocated pool of cyclic doubly linked lists,
        // representing a directed permutation graph.
        #[derive(Clone, Debug, Default)]
        pub struct CyclicListPool {
            links: Vec<[u32; 2]>,
        }

        impl CyclicListPool {
            pub fn with_size(n_nodes: usize) -> Self {
                Self {
                    links: (0..n_nodes as u32).map(|u| [u, u]).collect(),
                }
            }

            pub fn add_node(&mut self) -> u32 {
                let u = self.links.len() as u32;
                self.links.push([u, u]);
                u
            }

            pub fn is_isolated(&self, u: u32) -> bool {
                self.links[u as usize][0] == u
            }

            pub fn next(&self, u: u32) -> u32 {
                self.links[u as usize][1]
            }

            pub fn prev(&self, u: u32) -> u32 {
                self.links[u as usize][0]
            }

            pub fn isolate(&mut self, u: u32) {
                let [a, b] = self.links[u as usize];
                if a == u {
                    return;
                }

                self.links[a as usize][1] = b;
                self.links[b as usize][0] = a;
                self.links[u as usize] = [u, u];
            }

            pub fn insert_left(&mut self, pivot: u32, u: u32) {
                debug_assert!(self.is_isolated(u));
                let [a, _] = self.links[pivot as usize];
                self.links[a as usize][1] = u;
                self.links[u as usize] = [a, pivot];
                self.links[pivot as usize][0] = u;
            }

            pub fn insert_right(&mut self, pivot: u32, u: u32) {
                debug_assert!(self.is_isolated(u));
                let [_, a] = self.links[pivot as usize];
                self.links[pivot as usize][1] = u;
                self.links[u as usize] = [pivot, a];
                self.links[a as usize][0] = u;
            }

            pub fn pop_right(&mut self, u: u32) -> Option<u32> {
                let [_, v] = self.links[u as usize];
                if u == v {
                    return None;
                }

                self.isolate(v);
                Some(v)
            }

            pub fn for_each(&mut self, entry: u32, mut visitor: impl FnMut(u32)) {
                let mut u = entry;
                loop {
                    visitor(u);

                    u = self.next(u);
                    if u == entry {
                        return;
                    }
                }
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum NodeType {
        S, // Series (a simple cycle)
        P, // Parallel (a dipole with edge multiplictiy >= 3)
        R, // Rigid (a 3-vertex-connected graph that is neither series or parallel)
    }

    pub struct SPQRForest {
        pub n_verts: usize,
        pub n_real_edges: usize,

        pub dfs: DFSForest,
        pub bc: BCForest,
        pub skel: SkeletonGraph,

        pub split_comps: CSR<u32>,
        pub ty: Vec<NodeType>,
    }

    pub struct DFSForest {
        pub parent: Vec<u32>,
        pub t_in: Vec<u32>,
        pub low1: Vec<u32>, // Lowest euler index on a subtree's back edge
        pub low2: Vec<u32>, // Second lowest ...
        pub size: Vec<u32>,
        pub tour: Vec<u32>,
    }

    pub struct BCForest {
        // BC-node indexing:
        // - C nodes: 0..n_verts, directly correspond to original graph vertices)
        // - B nodes: n_verts..
        pub parent: Vec<u32>,
        pub degree: Vec<u32>,
    }

    pub struct SkeletonGraph {
        // Skeleton edge indexing:
        // - Real edges: 0..n_real_edges
        // - Virtual edges: n_real_edges..
        pub edges: Vec<[u32; 2]>,

        // Index of parent spqr node of an edge
        pub color: Vec<u32>,

        pub parent: Vec<u32>,
        pub real_children: CSR<u32>,
    }

    struct BicompCx<'a> {
        neighbors: &'a CSR<u32>,
        dfs: &'a mut DFSForest,
        bc: &'a mut BCForest,
        skel: &'a mut SkeletonGraph,

        timer: u32,
        current_edge: Vec<u32>,

        vstack: Vec<u32>,
        estack: Vec<[u32; 2]>,
    }

    impl<'a> BicompCx<'a> {
        pub fn new(
            n_verts: usize,
            neighbors: &'a CSR<u32>,
            dfs: &'a mut DFSForest,
            bc: &'a mut BCForest,
            skel: &'a mut SkeletonGraph,
        ) -> Self {
            BicompCx {
                neighbors,
                dfs,
                bc,
                skel,

                timer: 0,
                current_edge: (0..n_verts)
                    .map(|u| neighbors.edge_range(u).start as u32)
                    .collect(),

                vstack: vec![],
                estack: vec![],
            }
        }

        pub fn run(mut self) {
            let n_verts = self.current_edge.len();

            for u in 0..n_verts as u32 {
                if self.dfs.t_in[u as usize] != INF {
                    continue;
                }
                self.dfs(u);
            }

            let n_bc_nodes = self.bc.parent.len();
            self.bc.degree = vec![0; n_bc_nodes];
            for u in 0..n_bc_nodes as u32 {
                let p = self.bc.parent[u as usize];
                if p != u {
                    self.bc.degree[p as usize] += 1;
                    self.bc.degree[u as usize] += 1;
                }
            }
        }

        fn dfs(&mut self, root: u32) {
            let mut u = root as u32;
            loop {
                let p = self.dfs.parent[u as usize];
                let iv = self.current_edge[u as usize];
                self.current_edge[u as usize] += 1;
                if iv == self.neighbors.edge_range(u as usize).start as u32 {
                    self.enter_node(u);
                }
                if iv == self.neighbors.edge_range(u as usize).end as u32 {
                    if p == u {
                        break;
                    }

                    self.exit_tree_edge(p, u);
                    u = p;
                    continue;
                }

                let v = self.neighbors.links[iv as usize];
                if v == p {
                    continue;
                }

                if self.dfs.t_in[v as usize] == INF {
                    self.enter_tree_edge(u, v);
                    u = v;
                } else if self.dfs.t_in[v as usize] < self.dfs.t_in[u as usize] {
                    self.visit_back_edge(u, v);
                }
            }
        }

        fn enter_node(&mut self, u: u32) {
            self.vstack.push(u);
            self.dfs.t_in[u as usize] = self.timer;
            self.dfs.low1[u as usize] = self.timer;
            self.dfs.low2[u as usize] = self.timer;
            self.timer += 1;
        }

        fn enter_tree_edge(&mut self, u: u32, v: u32) {
            self.estack.push([u, v]);
            self.dfs.parent[v as usize] = u;
        }

        fn exit_tree_edge(&mut self, u: u32, v: u32) {
            if self.dfs.low1[v as usize] >= self.dfs.t_in[u as usize] {
                self.make_block_node(v, u);
            } else {
                self.dfs.size[u as usize] += self.dfs.size[v as usize];
            }

            match self.dfs.low1[u as usize].cmp(&self.dfs.low1[v as usize]) {
                Ordering::Less => {
                    self.dfs.low2[u as usize] =
                        self.dfs.low2[u as usize].min(self.dfs.low1[v as usize])
                }
                Ordering::Equal => {
                    self.dfs.low2[u as usize] =
                        self.dfs.low2[u as usize].min(self.dfs.low2[v as usize])
                }
                Ordering::Greater => {
                    self.dfs.low2[u as usize] =
                        self.dfs.low1[u as usize].min(self.dfs.low2[v as usize]);
                    self.dfs.low1[u as usize] = self.dfs.low1[v as usize];
                }
            }
        }

        fn visit_back_edge(&mut self, u: u32, v: u32) {
            self.estack.push([u, v]);
            match self.dfs.low1[u as usize].cmp(&self.dfs.t_in[v as usize]) {
                Ordering::Less => {
                    self.dfs.low2[u as usize] =
                        self.dfs.low2[u as usize].min(self.dfs.t_in[v as usize]);
                }
                Ordering::Equal => {}
                Ordering::Greater => {
                    self.dfs.low2[u as usize] = self.dfs.low1[u as usize];
                    self.dfs.low1[u as usize] = self.dfs.t_in[v as usize];
                }
            }
        }

        // Create a block node indexed by `b`.
        // Also, make `b` a virtual copy of the node `p`, which is the
        // new root of the dfs tree from BCC edges. It is garunteed from the
        // properties of the DFS tree and BCC that `u` is the only child of `b`.
        fn make_block_node(&mut self, u: u32, p: u32) -> u32 {
            debug_assert_eq!(self.dfs.parent[u as usize], p);
            let b = self.bc.parent.len() as u32;

            self.dfs.t_in.push(self.dfs.t_in[p as usize]);
            self.dfs
                .low1
                .push(self.dfs.t_in[p as usize].min(self.dfs.low1[u as usize]));
            self.dfs
                .low2
                .push(self.dfs.t_in[p as usize].min(self.dfs.low2[u as usize]));
            self.dfs.size.push(self.dfs.size[u as usize] + 1);
            self.bc.parent.push(p);
            while let Some(c) = self.vstack.pop() {
                self.bc.parent[c as usize] = b;
                if c == u {
                    break;
                }
            }

            while let Some([mut x, mut y]) = self.estack.pop() {
                if x == p {
                    x = b;
                }
                if y == p {
                    y = b;
                }

                self.skel.edges.push([x, y]);
                if [x, y] == [b, u] {
                    break;
                }
            }

            b
        }
    }

    impl SkeletonGraph {
        fn reorder_edges(&mut self, n_bc_nodes: usize, dfs: &DFSForest) {
            bucket_sort(&mut self.edges, 3 * n_bc_nodes as u32, |&[u, v]| {
                if dfs.t_in[u as usize] > dfs.t_in[v as usize] {
                    // Back edge
                    3 * dfs.t_in[v as usize] + 1
                } else if dfs.t_in[u as usize] > dfs.low2[v as usize] {
                    3 * dfs.low1[v as usize]
                } else {
                    3 * dfs.low1[v as usize] + 2
                }
            });

            self.real_children = CSR::from_pairs_rstable(
                n_bc_nodes,
                self.edges
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(e, &[u, _v])| (u, e as u32)),
            );
        }
    }

    struct EarDecompCx<'a> {
        n_verts: usize,
        dfs: &'a mut DFSForest,
        skel: &'a mut SkeletonGraph,

        rtimer: u32,
        t_in_new: Vec<u32>,
        t_base: u32,
        current_edge: Vec<u32>,
        t_trans: Vec<u32>,

        active_parent: Vec<u32>,
        active_parent_edge: Vec<u32>,
        active_children: CyclicListPool,
        unvisited_tree_edges: Vec<u32>,
        high_list: CyclicListPool,

        // Ear decomposition
        is_path_top: Vec<bool>,
        init_path: bool,
    }

    impl<'a> EarDecompCx<'a> {
        pub fn new(n_verts: usize, dfs: &'a mut DFSForest, skel: &'a mut SkeletonGraph) -> Self {
            let n_bc_nodes = skel.real_children.len();

            let mut this = EarDecompCx {
                n_verts,
                dfs,

                rtimer: 0,
                t_in_new: vec![],
                t_base: 0,
                current_edge: (0..n_bc_nodes)
                    .map(|u| skel.real_children.edge_range(u).start as u32)
                    .collect(),
                t_trans: vec![UNSET; n_bc_nodes],

                active_parent: (0..n_bc_nodes as u32).collect(),
                active_parent_edge: vec![UNSET; n_bc_nodes],
                active_children: CyclicListPool::with_size(n_bc_nodes + skel.edges.len()),
                unvisited_tree_edges: vec![0u32; n_bc_nodes],
                high_list: CyclicListPool::with_size(n_bc_nodes + skel.edges.len()),

                is_path_top: vec![false; skel.edges.len()],
                init_path: true,

                skel,
            };

            this.skel.parent = (0..n_bc_nodes as u32).collect();
            this.t_in_new = this.dfs.size.clone();
            for e in 0..this.skel.edges.len() as u32 {
                let [u, v] = this.skel.edges[e as usize];
                if this.dfs.t_in[u as usize] < this.dfs.t_in[v as usize] {
                    this.skel.parent[v as usize] = u;
                    this.active_parent[v as usize] = u;
                    this.active_parent_edge[v as usize] = e;
                    this.unvisited_tree_edges[u as usize] += 1;
                }
                this.active_children.insert_left(u, e + n_bc_nodes as u32);
            }

            this
        }

        fn run(&mut self) {
            let n_bc_nodes = self.skel.real_children.len();

            for b in (self.n_verts as u32..n_bc_nodes as u32).rev() {
                self.t_base += self.dfs.size[b as usize];
                self.rtimer = self.t_base;
                self.dfs(b);
            }

            for u in 0..self.n_verts as u32 {
                if self.skel.parent[u as usize] == u {
                    // Process isolated vertex to ensure that `t_in` is a valid permutation
                    self.t_in_new[u as usize] = self.t_base;
                    self.dfs.low1[u as usize] = self.t_base;
                    self.dfs.low2[u as usize] = self.t_base;
                    self.t_base += 1;
                }
            }

            self.dfs.tour = inv_perm(&self.t_in_new);
            self.dfs.t_in = std::mem::take(&mut self.t_in_new);
        }

        fn dfs(&mut self, block_root: u32) {
            let mut u = block_root as u32;
            loop {
                let p = self.skel.parent[u as usize];

                let ie = self.current_edge[u as usize];
                self.current_edge[u as usize] += 1;

                if ie == self.skel.real_children.edge_range(u as usize).start as u32 {
                    self.enter_node(u);
                }
                if ie == self.skel.real_children.edge_range(u as usize).end as u32 {
                    if p == u {
                        break;
                    }

                    self.exit_tree_edge(p, u);
                    u = p;
                    continue;
                }

                let e = self.skel.real_children.links[ie as usize];
                let [_, v] = self.skel.edges[e as usize];
                if self.skel.parent[v as usize] == u {
                    self.enter_tree_edge(u, v, e);
                    u = v;
                } else {
                    self.enter_back_edge(u, v, e);
                }
            }
        }

        fn enter_node(&mut self, u: u32) {
            self.t_in_new[u as usize] = self.rtimer - self.dfs.size[u as usize];

            self.t_trans[self.dfs.t_in[u as usize] as usize] = self.t_in_new[u as usize];
            self.dfs.low1[u as usize] = self.t_trans[self.dfs.low1[u as usize] as usize];
            self.dfs.low2[u as usize] = self.t_trans[self.dfs.low2[u as usize] as usize];
        }

        fn enter_tree_edge(&mut self, _u: u32, _v: u32, e: u32) {
            self.check_path_top(e);
        }

        fn exit_tree_edge(&mut self, _u: u32, _v: u32) {
            self.rtimer -= 1;
        }

        fn enter_back_edge(&mut self, _u: u32, v: u32, e: u32) {
            self.check_path_top(e);

            self.init_path = true;
            let n_bc_nodes = self.skel.real_children.len();
            self.high_list.insert_left(v, e + n_bc_nodes as u32);
        }

        fn check_path_top(&mut self, e: u32) {
            self.is_path_top[e as usize] = self.init_path;
            self.init_path = false;
        }
    }

    struct TutteDecompCx<'a> {
        n_verts: usize,
        n_real_edges: usize,
        dfs: &'a DFSForest,
        skel: &'a mut SkeletonGraph,

        current_edge: Vec<u32>,

        active_parent: Vec<u32>,
        active_parent_edge: Vec<u32>,
        active_children: CyclicListPool,
        unvisited_tree_edges: Vec<u32>,
        high_list: CyclicListPool,
        is_path_top: Vec<bool>,
        degree: Vec<u32>,

        split_comps: CSR<u32>,
        ty: Vec<NodeType>,

        estack: Vec<u32>,
        tstack: Vec<[u32; 3]>,
    }

    impl<'a> TutteDecompCx<'a> {
        pub fn new(prev_cx: EarDecompCx<'a>) -> Self {
            let EarDecompCx {
                n_verts,
                dfs,
                skel,

                active_parent,
                active_parent_edge,
                active_children,
                unvisited_tree_edges,
                high_list,

                is_path_top,
                ..
            } = prev_cx;

            let n_bc_nodes = skel.real_children.len();

            let current_edge = (0..n_bc_nodes)
                .map(|u| skel.real_children.edge_range(u).start as u32)
                .collect();

            let mut degree = vec![0u32; n_bc_nodes];
            for &[u, v] in &skel.edges {
                degree[u as usize] += 1;
                degree[v as usize] += 1;
            }

            TutteDecompCx {
                n_verts,
                n_real_edges: skel.edges.len(),
                dfs,
                skel,

                current_edge,

                active_parent,
                active_parent_edge,
                active_children,
                unvisited_tree_edges,
                high_list,
                is_path_top,
                degree,

                split_comps: CSR::default(),
                ty: vec![],

                estack: vec![],
                tstack: vec![],
            }
        }

        fn run(&mut self) {
            for b in (self.n_verts as u32..self.n_bc_nodes() as u32).rev() {
                if self.is_root_of_cut_edge(b) {
                    continue;
                }

                self.tstack.push(TSTACK_SEP);
                self.dfs(b);

                self.add_comp([UNSET; 2], NodeType::R);

                while let Some(e) = self.estack.pop() {
                    self.move_edge_to_last_comp(e);
                }
            }

            for s in 0..self.split_comps.len() {
                let degree = self.split_comps[s].len()
                    + (self.skel.edges[s + self.n_real_edges][0] != UNSET) as usize;
                if self.ty[s] == NodeType::R && degree <= 3 {
                    self.ty[s] = NodeType::S;
                }
            }
        }

        fn dfs(&mut self, block_root: u32) {
            let mut u = block_root as u32;
            loop {
                let p = self.skel.parent[u as usize];

                let ie = self.current_edge[u as usize];
                self.current_edge[u as usize] += 1;

                if ie == self.skel.real_children.edge_range(u as usize).end as u32 {
                    if p == u {
                        break;
                    }

                    let ep =
                        self.skel.real_children.links[self.current_edge[p as usize] as usize - 1];
                    self.exit_tree_edge(p, u, ep, block_root);
                    u = p;
                    continue;
                }

                let e = self.skel.real_children.links[ie as usize];
                let [_, v] = self.skel.edges[e as usize];
                if self.skel.parent[v as usize] == u {
                    self.enter_tree_edge(u, v, e);
                    u = v;
                } else {
                    self.enter_back_edge(u, v, e);
                }
            }
        }

        fn enter_tree_edge(&mut self, u: u32, v: u32, e: u32) {
            self.unvisited_tree_edges[u as usize] -= 1;

            if self.is_path_top[e as usize] {
                let mut last = TSTACK_SEP;
                let mut max = 0;
                loop {
                    let &[h, a, b] = self.tstack.last().unwrap();
                    if !(h != UNSET && self.dfs.t_in[a as usize] > self.dfs.low1[v as usize]) {
                        break;
                    }

                    last = [h, a, b];
                    max = max.max(self.dfs.t_in[h as usize]);
                    self.tstack.pop();
                }

                if last == TSTACK_SEP {
                    self.tstack.push([
                        self.dfs.tour
                            [(self.dfs.t_in[v as usize] + self.dfs.size[v as usize] - 1) as usize],
                        self.dfs.tour[self.dfs.low1[v as usize] as usize],
                        u,
                    ]);
                } else {
                    self.tstack.push([
                        self.dfs.tour[max as usize],
                        self.dfs.tour[self.dfs.low1[v as usize] as usize],
                        last[2],
                    ]);
                }
                self.tstack.push(TSTACK_SEP);
            }
        }

        fn exit_tree_edge(&mut self, u: u32, v: u32, e: u32, block_root: u32) {
            self.estack.push(self.active_parent_edge[v as usize]);

            self.check_type2_split_pairs(u, v, block_root);
            self.check_type1_split_pair(u, v, block_root);

            if self.is_path_top[e as usize] {
                loop {
                    let t = self.tstack.pop().unwrap();
                    if t == TSTACK_SEP {
                        break;
                    }
                }
            }

            loop {
                let &[h, _a, b] = self.tstack.last().unwrap();
                if !(h != UNSET && b != u && self.high(u) > self.dfs.t_in[h as usize]) {
                    break;
                }
                self.tstack.pop();
            }
        }

        fn enter_back_edge(&mut self, u: u32, v: u32, e: u32) {
            self.estack.push(e);

            if self.is_path_top[e as usize] {
                let mut last = TSTACK_SEP;
                let mut max = 0;
                loop {
                    let &[h, a, b] = self.tstack.last().unwrap();
                    if !(h != UNSET && self.dfs.t_in[a as usize] > self.dfs.t_in[v as usize]) {
                        break;
                    }

                    last = [h, a, b];
                    max = max.max(self.dfs.t_in[h as usize]);
                    self.tstack.pop();
                }

                if last == TSTACK_SEP {
                    self.tstack.push([u, v, u]);
                } else {
                    self.tstack.push([self.dfs.tour[max as usize], v, last[2]]);
                }
            }
        }

        fn check_type2_split_pairs(&mut self, u: u32, mut v: u32, block_root: u32) {
            let shift = self.n_bc_nodes() as u32;

            loop {
                if u == block_root {
                    break;
                }

                let &[h, a, b] = self.tstack.last().unwrap();
                let case1 = a == u;
                let case2 = self.degree[v as usize] == 2
                    && self.dfs.t_in[self.skel.edges
                        [(self.active_children.next(v) - shift) as usize][1]
                        as usize]
                        > self.dfs.t_in[v as usize];
                if !(case1 || case2) {
                    break;
                }

                if case1 && self.active_parent[b as usize] == a {
                    self.tstack.pop();
                    continue;
                }

                let mut e_ab = UNSET;
                let mut e_virtual;
                if case2 {
                    let e_uv = self.estack.pop().unwrap();
                    let e_vx = self.estack.pop().unwrap();
                    let [_, x] = self.skel.edges[e_vx as usize];
                    e_virtual = self.add_comp([u, x], NodeType::S); // Triangle
                    self.move_edge_to_last_comp(e_uv);
                    self.move_edge_to_last_comp(e_vx);

                    if let Some(&e_xu_cand) = self.estack.last() {
                        if self.skel.edges[e_xu_cand as usize] == [x, u] {
                            self.estack.pop();
                            e_ab = e_xu_cand;
                        }
                    }

                    v = x;
                } else {
                    debug_assert!(case1);
                    self.tstack.pop();
                    e_virtual = self.add_comp([u, b], NodeType::R);
                    while let Some(&e_xy) = self.estack.last() {
                        let [x, y] = self.skel.edges[e_xy as usize];

                        let bound = self.dfs.t_in[a as usize]..=self.dfs.t_in[h as usize];
                        if !(bound.contains(&self.dfs.t_in[x as usize])
                            && bound.contains(&self.dfs.t_in[y as usize]))
                        {
                            break;
                        }

                        self.estack.pop();
                        if [x, y] == [a, b] || [x, y] == [b, a] {
                            e_ab = e_xy;
                            self.high_list.isolate(e_xy + shift);
                        } else {
                            self.move_edge_to_last_comp(e_xy);
                        }
                    }

                    v = b;
                }

                if e_ab != UNSET {
                    let e_prev = e_virtual;
                    e_virtual = self.add_comp([u, v], NodeType::P);
                    self.move_edge_to_last_comp(e_ab);
                    self.move_edge_to_last_comp(e_prev);
                }
                self.estack.push(e_virtual);
                self.mark_tree_edge(e_virtual);
            }
        }

        fn check_type1_split_pair(&mut self, u: u32, v: u32, block_root: u32) {
            if !((self.dfs.low1[v as usize] + 1..=self.dfs.low2[v as usize])
                .contains(&self.dfs.t_in[u as usize])
                && (self.active_parent[u as usize] != block_root
                    || self.unvisited_tree_edges[u as usize] >= 1))
            {
                return;
            }

            let z = self.dfs.tour[self.dfs.low1[v as usize] as usize];
            let mut e_virtual_back = self.add_comp([u, z], NodeType::R);
            let bound =
                self.dfs.t_in[v as usize]..self.dfs.t_in[v as usize] + self.dfs.size[v as usize];
            while let Some(&e_xy) = self.estack.last() {
                let [x, y] = self.skel.edges[e_xy as usize];
                if !(bound.contains(&self.dfs.t_in[x as usize])
                    || bound.contains(&self.dfs.t_in[y as usize]))
                {
                    break;
                }

                self.estack.pop();
                self.move_edge_to_last_comp(e_xy);
            }

            if let Some(&e_uz) = self.estack.last() {
                if self.skel.edges[e_uz as usize] == [u, z] {
                    self.estack.pop();
                    let e_prev = e_virtual_back;
                    e_virtual_back = self.add_comp([u, z], NodeType::P);
                    self.move_edge_to_last_comp(e_uz);
                    self.move_edge_to_last_comp(e_prev);
                }
            }

            if z == self.active_parent[u as usize] {
                let e_virtual_tree = self.add_comp([z, u], NodeType::P);
                self.move_edge_to_last_comp(self.active_parent_edge[u as usize]);
                self.move_edge_to_last_comp(e_virtual_back);
                self.mark_tree_edge(e_virtual_tree);
            } else {
                self.estack.push(e_virtual_back);
                self.mark_back_edge(e_virtual_back);
            }
        }

        pub fn n_bc_nodes(&self) -> usize {
            self.active_parent.len()
        }

        pub fn is_root_of_cut_edge(&self, b: u32) -> bool {
            let e = self.skel.real_children[b as usize][0];
            let [_b, u] = self.skel.edges[e as usize];
            self.dfs.low1[u as usize] > self.dfs.t_in[b as usize]
        }

        // Create a split component with a virtual edge.
        fn add_comp(&mut self, e_virtual_ends: [u32; 2], ty: NodeType) -> u32 {
            let shift = self.n_bc_nodes() as u32;

            self.split_comps.push();
            self.ty.push(ty);

            let e = self.skel.edges.len() as u32;
            self.skel.edges.push(e_virtual_ends);
            self.is_path_top.push(false);

            let [u, v] = e_virtual_ends;
            self.active_children.add_node();
            self.high_list.add_node();
            if u != UNSET {
                self.active_children.insert_left(u, e + shift);

                self.degree[u as usize] += 1;
                self.degree[v as usize] += 1;
            }
            e
        }

        fn move_edge_to_last_comp(&mut self, e: u32) {
            let shift = self.n_bc_nodes() as u32;

            self.active_children.isolate(e + shift);
            self.high_list.isolate(e + shift);
            self.split_comps.push_to_last_row(e);

            let [u, v] = self.skel.edges[e as usize];
            self.degree[u as usize] -= 1;
            self.degree[v as usize] -= 1;
        }

        fn mark_tree_edge(&mut self, e_virtual: u32) {
            let [u, v] = self.skel.edges[e_virtual as usize];
            self.active_parent[v as usize] = u;
            self.active_parent_edge[v as usize] = e_virtual;
        }

        fn mark_back_edge(&mut self, e_virtual: u32) {
            let [u, v] = self.skel.edges[e_virtual as usize];
            if self.dfs.t_in[u as usize] > self.high(v) {
                let shift = self.n_bc_nodes() as u32;
                self.high_list.insert_right(v, e_virtual + shift);
            }
        }

        fn high(&self, u: u32) -> u32 {
            if self.high_list.is_isolated(u) {
                0
            } else {
                let shift = self.n_bc_nodes() as u32;
                let e = self.high_list.next(u) - shift;
                let [v, _] = self.skel.edges[e as usize];
                self.dfs.t_in[v as usize]
            }
        }
    }

    fn merge_split_components(
        n_real_edges: usize,
        skel: &mut SkeletonGraph,
        split_comps: &mut CSR<u32>,
        ty: &mut Vec<NodeType>,
    ) {
        skel.color = vec![UNSET; skel.edges.len()];
        let mut trans = vec![UNSET; split_comps.len()];
        let mut n_colors = 0u32;

        // Note: `split_comps` are topologically sorted
        let mut ty_new = vec![];
        for s in (0..split_comps.len() as u32).rev() {
            let mut s_new = s;

            let v = s + n_real_edges as u32;
            let p = skel.color[v as usize];
            debug_assert!((skel.edges[v as usize][0] != UNSET) == (p != UNSET));
            if p != UNSET
                && ty[s as usize] == ty[p as usize]
                && matches!(ty[s as usize], NodeType::P | NodeType::S)
            {
                trans[s as usize] = trans[p as usize];
                s_new = p;
                skel.color[v as usize] = INACTIVE;
            } else {
                trans[s as usize] = n_colors as u32;
                n_colors += 1;
                ty_new.push(ty[s as usize]);
            }

            for &e in &split_comps[s as usize] {
                skel.color[e as usize] = s_new;
            }
        }
        ty_new.reverse();
        *ty = ty_new;

        for dest in &mut trans {
            *dest = n_colors - 1 - *dest;
        }

        let mut i = 0;
        for e in 0..skel.color.len() {
            if skel.color[e as usize] == INACTIVE {
                continue;
            }
            skel.edges[i] = skel.edges[e];
            skel.color[i] = if skel.color[e] == UNSET {
                UNSET
            } else {
                trans[skel.color[e] as usize]
            };
            i += 1;
        }
        skel.edges.truncate(i);
        skel.color.truncate(i);

        *split_comps = CSR::from_pairs_rstable(
            n_colors as usize,
            skel.color
                .iter()
                .enumerate()
                .rev()
                .filter(|&(_, &c)| c != UNSET)
                .map(|(e, &c)| (c, e as u32)),
        );
    }

    impl SPQRForest {
        pub fn from_edges(n_verts: usize, edges: impl Iterator<Item = [u32; 2]> + Clone) -> Self {
            let mut dfs = DFSForest {
                parent: (0..n_verts as u32).collect(),
                t_in: vec![INF; n_verts],
                low1: vec![INF; n_verts],
                low2: vec![INF; n_verts],
                size: vec![1; n_verts],
                tour: vec![],
            };
            let mut bc = BCForest {
                parent: (0..n_verts as u32).collect(),
                degree: vec![],
            };
            let mut skel = SkeletonGraph {
                edges: vec![],
                color: vec![],

                parent: vec![],
                real_children: CSR::default(),
            };

            // Step 0. Find biconnected components & build a BC forest.
            let neighbors =
                CSR::from_pairs_rstable(n_verts, edges.flat_map(|[u, v]| [(u, v), (v, u)]));
            let cx0 = BicompCx::new(n_verts, &neighbors, &mut dfs, &mut bc, &mut skel);
            cx0.run();
            let n_real_edges = skel.edges.len();

            // Step 1. Perform reindexing and build an ear decomposition.
            let n_bc_nodes = bc.parent.len();
            skel.reorder_edges(n_bc_nodes, &dfs);

            let mut cx1 = EarDecompCx::new(n_verts, &mut dfs, &mut skel);
            cx1.run();

            // Step 2. Partition each block into its split components.
            let mut cx2 = TutteDecompCx::new(cx1);
            cx2.run();

            let mut split_comps = cx2.split_comps;
            let mut ty = cx2.ty;

            // Step 3. merge adjacent nodes of the same type that is S or P.
            merge_split_components(n_real_edges, &mut skel, &mut split_comps, &mut ty);

            SPQRForest {
                n_verts,
                n_real_edges,

                dfs,
                bc,
                skel,

                split_comps,
                ty,
            }
        }

        pub fn n_verts(&self) -> usize {
            self.n_verts
        }

        pub fn n_bc_nodes(&self) -> usize {
            self.bc.parent.len()
        }

        pub fn block_node_range(&self) -> std::ops::Range<u32> {
            self.n_verts() as u32..self.n_bc_nodes() as u32
        }

        pub fn parent_virtual_edge(&self, s: u32) -> u32 {
            s + self.n_real_edges as u32
        }

        pub fn virtual_edge_range(&self) -> std::ops::Range<u32> {
            self.n_real_edges as u32..self.skel.edges.len() as u32
        }

        pub fn is_cut_vert(&self, u: u32) -> bool {
            self.bc.degree[u as usize] >= 2
        }

        pub fn is_bridge(&self, u: u32, v: u32) -> bool {
            let b = |w| self.bc.parent[w as usize];
            self.bc.degree[b(u) as usize] == 2 && b(b(u)) == v
                || self.bc.degree[b(v) as usize] == 2 && b(b(v)) == u
        }

        pub fn verify_ty(&self) {
            use std::collections::HashMap;
            for s in 0..self.split_comps.len() {
                let mut edges = vec![];
                let mut degree = HashMap::<u32, u32>::new();
                for e in self.split_comps[s as usize]
                    .iter()
                    .copied()
                    .chain(std::iter::once((s + self.n_real_edges) as u32))
                {
                    let [mut u, mut v] = self.skel.edges[e as usize];
                    if u == UNSET {
                        continue;
                    }
                    if u > v {
                        std::mem::swap(&mut u, &mut v);
                    }
                    edges.push([u, v]);
                    *degree.entry(u).or_default() += 1;
                    *degree.entry(v).or_default() += 1;
                }

                let cond_s = degree.len() >= 3
                    && degree.len() == edges.len()
                    && degree.values().all(|&d| d == 2);
                let cond_p = degree.len() == 2 && edges.len() >= 3;
                let cond_r_weak = !cond_s && !cond_p && degree.len() >= 4;
                match self.ty[s] {
                    NodeType::S => assert!(cond_s),
                    NodeType::P => assert!(cond_p),
                    NodeType::R => assert!(cond_r_weak),
                }
            }
        }
    }
}
