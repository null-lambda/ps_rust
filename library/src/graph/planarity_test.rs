pub mod boyer_myrvold {
    pub const UNSET: u32 = u32::MAX;
    pub const INF: u32 = u32::MAX;

    use crate::jagged::CSR;
    use linked_list::{CyclicListPool, UndirectedCyclicListPool};

    // Boyer-myrvold planarity test, O(N)
    //
    // ## Reference
    // - J. Boyer and W. Myrvold, "Stop minding your p's and q's: a simplified O(n) planar embedding algorithm", 1999.
    // - J. Boyer and W. Myrvold,  "On the Cutting Edge: Simplified O(n) Planarity by Edge Addition", 2004.

    // Sort indices 0..n using a key in 0..key_bound
    pub fn bucket_sort_iota(
        n: u32,
        key_bound: u32,
        mut key: impl FnMut(&u32) -> u32,
    ) -> impl Iterator<Item = u32> {
        const UNSET: u32 = u32::MAX;

        // Forward-star
        let mut head = vec![UNSET; key_bound as usize];
        let mut link = vec![UNSET; n as usize];

        for x in (0..n).rev() {
            let k = key(&x);
            debug_assert!(k < key_bound);

            link[x as usize] = head[k as usize];
            head[k as usize] = x;
        }

        let mut k = 0;
        let mut x = UNSET;
        std::iter::from_fn(move || {
            while x == UNSET {
                if k == key_bound {
                    return None;
                }

                x = head[k as usize];
                k += 1;
            }

            let res = x;
            x = link[x as usize];
            Some(res)
        })
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

        // Cyclic linked lists, where links[u][0] and links[u][1] may be swapped arbitarily.
        // (Undirected permutation graph)
        #[derive(Clone, Debug, Default)]
        pub struct UndirectedCyclicListPool {
            links: Vec<[u32; 2]>,
        }

        impl UndirectedCyclicListPool {
            pub fn with_size(n_nodes: usize) -> Self {
                Self {
                    links: (0..n_nodes as u32).map(|u| [u, u]).collect(),
                }
            }

            pub fn is_isolated(&self, u: u32) -> bool {
                self.links[u as usize][0] == u
            }

            pub fn xor(&self, u: u32) -> u32 {
                let [a, b] = self.links[u as usize];
                a ^ b
            }

            pub fn any_side(&self, u: u32) -> u32 {
                self.links[u as usize][0]
            }

            pub fn links(&self, u: u32) -> [u32; 2] {
                self.links[u as usize]
            }

            #[inline]
            fn update_link(&mut self, u: u32, old: u32, new: u32) {
                debug_assert!(self.links[u as usize].contains(&old));
                let b = (self.links[u as usize][1] == old) as usize;
                self.links[u as usize][b] = new;
            }

            pub fn isolate(&mut self, u: u32) {
                let [x, y] = self.links[u as usize];
                if x == u {
                    return;
                }

                self.update_link(x, u, y);
                self.update_link(y, u, x);
                self.links[u as usize] = [u, u];
            }

            pub fn step(&self, u: &mut u32, prev: &mut u32) {
                *u = self.xor(*u) ^ std::mem::replace(prev, *u);
            }

            fn is_connected(&self, u: u32, v: u32) -> bool {
                let mut prev = self.links[u as usize][0];
                let mut c = u;
                loop {
                    if c == v {
                        return true;
                    }

                    self.step(&mut c, &mut prev);
                    if c == u {
                        return false;
                    }
                }
            }

            pub fn insert_slice_between(&mut self, pu: u32, u: u32, v: u32, nv: u32) {
                // debug_assert!(self.is_connected(pu, nv));
                // debug_assert!(self.is_connected(u, v));
                debug_assert!(self.links[u as usize].contains(&v));
                debug_assert!(self.links[pu as usize].contains(&nv));
                debug_assert!(pu != u);
                debug_assert!(nv != v);

                self.update_link(pu, nv, u);
                self.update_link(u, v, pu);
                self.update_link(v, u, nv);
                self.update_link(nv, pu, v);
            }

            pub fn insert_between(&mut self, u: u32, v: u32, a: u32) {
                debug_assert!(self.links[u as usize].contains(&v));
                debug_assert!(self.is_isolated(a));
                debug_assert!(u != a);
                debug_assert!(v != a);

                self.update_link(u, v, a);
                self.update_link(v, u, a);
                self.links[a as usize] = [u, v];
            }

            pub fn insert_any_side(&mut self, pivot: u32, u: u32) {
                self.insert_between(pivot, self.any_side(pivot), u);
            }

            pub fn split_slice_out(&mut self, pu: u32, u: u32, v: u32, nv: u32) {
                // debug_assert!(self.is_connected(u, v));
                debug_assert!(self.links[u as usize].contains(&pu));
                debug_assert!(self.links[v as usize].contains(&nv));
                debug_assert!(pu != u);
                debug_assert!(nv != v);

                self.update_link(pu, u, nv);
                self.update_link(u, pu, v);
                self.update_link(v, nv, u);
                self.update_link(nv, v, pu);
            }

            pub fn find1(
                &self,
                entry: u32,
                next: u32,
                mut pred: impl FnMut(u32) -> bool,
            ) -> Option<(u32, u32)> {
                let mut prev = entry;
                let mut c = next;
                while c != entry {
                    if pred(c) {
                        return Some((c, prev));
                    }

                    self.step(&mut c, &mut prev);
                }

                None
            }

            pub fn bidirectional_search<T>(
                &self,
                entry: u32,
                mut visitor: impl FnMut(u32) -> Result<(), T>,
            ) -> T {
                let mut prev = self.links[entry as usize];
                let mut c = [entry; 2];
                loop {
                    if let Err(e) = visitor(c[0]) {
                        return e;
                    }
                    self.step(&mut c[0], &mut prev[0]);

                    self.step(&mut c[1], &mut prev[1]);
                    if let Err(e) = visitor(c[1]) {
                        return e;
                    }
                }
            }

            pub fn for_each(&mut self, entry: u32, next: u32, mut visitor: impl FnMut(u32)) {
                self.find1(entry, next, |x| {
                    visitor(x);
                    false
                });
            }
        }
    }

    #[derive(Debug)]
    pub struct DFSForest {
        parent: Vec<u32>,
        parent_edge: Vec<u32>,
        t_in: Vec<u32>,
        tour: Vec<u32>,

        low: Vec<u32>,            // Lowest euler index on a subtree's back edge
        least_ancestor: Vec<u32>, // Lowest euler index through a back edge

        order_by_low: Vec<u32>,
    }

    impl DFSForest {
        fn from_assoc_list(neighbors: &CSR<u32>) -> Self {
            let n = neighbors.len();

            let mut parent: Vec<_> = (0..n as u32).collect();
            let mut parent_edge: Vec<_> = vec![UNSET; n];
            let mut t_in = vec![UNSET; n];
            let mut tour = vec![];

            let mut low = vec![UNSET; n];
            let mut least_ancestor = vec![INF; n];

            let mut current_edge = neighbors.head[..n].to_vec();
            let mut timer = 0u32;
            for root in 0..n as u32 {
                if t_in[root as usize] != UNSET {
                    continue;
                }

                // Init root node
                parent[root as usize] = root;
                least_ancestor[root as usize] = timer;

                let mut u = root as u32;
                loop {
                    let p = parent[u as usize];
                    let e = current_edge[u as usize];
                    current_edge[u as usize] += 1;
                    if e == neighbors.head[u as usize] {
                        // On enter
                        t_in[u as usize] = timer;
                        low[u as usize] = timer;
                        tour.push(u);
                        timer += 1;
                    }
                    if e == neighbors.head[u as usize + 1] {
                        // On exit
                        if p == u {
                            break;
                        }

                        low[p as usize] = low[p as usize].min(low[u as usize]);

                        u = p;
                        continue;
                    }

                    let v = neighbors.link[e as usize];
                    if v == p {
                        continue;
                    }
                    assert!(u != v, "Self-loops not handled");

                    if t_in[v as usize] == UNSET {
                        // Forward edge (a part of DFS spanning tree)
                        parent[v as usize] = u;
                        parent_edge[v as usize] = e;
                        least_ancestor[v as usize] = t_in[u as usize];

                        u = v;
                    } else if t_in[v as usize] < t_in[u as usize] {
                        // Back edge
                        low[u as usize] = low[u as usize].min(t_in[v as usize]);
                        least_ancestor[u as usize] =
                            least_ancestor[u as usize].min(t_in[v as usize]);
                    }
                }
            }

            let order_by_low = bucket_sort_iota(n as u32, n as u32, |&u| low[u as usize]).collect();

            Self {
                parent,
                parent_edge,
                t_in,
                tour,

                low,
                least_ancestor,
                order_by_low,
            }
        }
    }

    pub struct BoyerMyrvold {
        neighbors: CSR<u32>,
        dfs: DFSForest,

        // Tracks externally active vertices.
        // Indexing:
        //     node:     n..2n
        //     children: 0..n
        sep_children: CyclicListPool,

        // Boundary edges of a biconnected component.
        // Indexing:
        //     vertex:   0..n
        //     bcc root: n..2n (accessed via the unique DFS child in bcc)
        boundary: UndirectedCyclicListPool,

        // Tracks pertinent / active vertices
        // Indexing: same as `boundary`.
        pertinent_roots: CyclicListPool,

        // Markers for `walk_up` and `walk_down`.
        visited_from: Vec<u32>,
        back_edge_flag: Vec<u32>,

        // Buffer for `walk_down`.
        merge_stack: Vec<(u32, u32, u32)>,
    }

    impl BoyerMyrvold {
        fn from_assoc_list(neighbors: CSR<u32>) -> Self {
            let n = neighbors.len();
            let dfs = DFSForest::from_assoc_list(&neighbors);

            let mut this = BoyerMyrvold {
                neighbors,
                dfs,

                sep_children: CyclicListPool::with_size(n + n),

                boundary: UndirectedCyclicListPool::with_size(n + n),

                pertinent_roots: CyclicListPool::with_size(n + n),
                visited_from: vec![UNSET; n + n],
                back_edge_flag: vec![UNSET; n],

                merge_stack: vec![],
            };

            for &u in &this.dfs.order_by_low {
                let p = this.dfs.parent[u as usize];
                if u == p {
                    continue;
                }

                this.sep_children.insert_left(n as u32 + p, u);
                this.boundary.insert_any_side(n as u32 + u, u);
            }

            this
        }

        fn n_verts(&self) -> usize {
            self.neighbors.len()
        }

        fn is_pertinent(&self, u: u32, w: u32) -> bool {
            self.back_edge_flag[w as usize] == u || !self.pertinent_roots.is_isolated(w)
        }

        fn is_externally_active(&self, u: u32, w: u32) -> bool {
            let n = self.n_verts();
            self.dfs.least_ancestor[w as usize] < self.dfs.t_in[u as usize]
                || !self.sep_children.is_isolated(n as u32 + w) && {
                    let c0 = self.sep_children.next(n as u32 + w);
                    self.dfs.low[c0 as usize] < self.dfs.t_in[u as usize]
                }
        }

        fn is_internally_active(&self, u: u32, w: u32) -> bool {
            self.is_pertinent(u, w) && !self.is_externally_active(u, w)
        }

        fn is_active(&self, u: u32, w: u32) -> bool {
            self.is_pertinent(u, w) || self.is_externally_active(u, w)
        }

        fn walk_up(&mut self, u: u32, mut w: u32) {
            let n = self.n_verts();

            self.back_edge_flag[w as usize] = u;

            loop {
                let Some(bcc_root) = self.boundary.bidirectional_search(w, |g| {
                    if self.visited_from[g as usize] == u {
                        return Err(None);
                    }
                    self.visited_from[g as usize] = u;

                    if g >= n as u32 {
                        Err(Some(g))
                    } else {
                        Ok(())
                    }
                }) else {
                    return;
                };

                let c = bcc_root - n as u32; // A unique DFS child of a bcc root in bcc.
                let r = self.dfs.parent[c as usize];
                if self.dfs.low[c as usize].min(self.dfs.least_ancestor[c as usize])
                    < self.dfs.t_in[r as usize]
                {
                    // Append externally active bcc
                    self.pertinent_roots.insert_left(r, bcc_root);
                } else {
                    // Prepend internally active bcc
                    self.pertinent_roots.insert_right(r, bcc_root);
                }

                if r == u || self.visited_from[r as usize] == u {
                    break;
                }
                w = r;
            }
        }

        fn walk_down(&mut self, u: u32, bcc_root: u32) -> bool {
            let n = self.n_verts();

            self.merge_stack.clear();
            let mut a = bcc_root;
            loop {
                let [la, ra] = self.boundary.links(a);
                let Some((x, px)) = self.boundary.find1(a, la, |w| self.is_active(u, w)) else {
                    break;
                };
                let Some((y, py)) = self.boundary.find1(a, ra, |w| self.is_active(u, w)) else {
                    break;
                };

                let z = if self.is_internally_active(u, x) {
                    x
                } else if self.is_internally_active(u, y) {
                    y
                } else if self.is_pertinent(u, x) {
                    x
                } else if self.is_pertinent(u, y) {
                    y
                } else {
                    debug_assert!(self.is_externally_active(u, x));
                    debug_assert!(self.is_externally_active(u, y));
                    if x != y {
                        // Check whether there is a pertinent vertex in the path `x~y`.
                        // If so, the path from that vertex to the `bcc_root` must pass through `x` or `y`,
                        // thereby intersecting with external path from `x` or `y`.
                        //
                        // Traverse only nodes visited by `walk_up` to bound the total cost.
                        let (mut c, mut pc) = (x, px);
                        loop {
                            self.boundary.step(&mut c, &mut pc);
                            if c == y || self.visited_from[c as usize] != u {
                                break;
                            }
                            if self.is_pertinent(u, c) {
                                return false;
                            }
                        }

                        if c != y {
                            let (mut d, mut pd) = (y, py);
                            loop {
                                self.boundary.step(&mut d, &mut pd);
                                if self.visited_from[d as usize] != u {
                                    break;
                                }
                                if self.is_pertinent(u, d) {
                                    return false;
                                }
                            }
                        }
                    }

                    // Add a short-circuit edge
                    self.boundary.split_slice_out(px, x, y, py);
                    self.boundary.isolate(bcc_root);
                    self.boundary.insert_between(x, y, bcc_root);

                    break;
                };

                let mut pz = if z == x { px } else { py };
                let mut fz = if z == x { ra } else { la };
                if self.back_edge_flag[z as usize] != u {
                    assert!(!self.pertinent_roots.is_isolated(z));
                    self.merge_stack.push((z, pz, fz));
                    a = self.pertinent_roots.next(z);
                    continue;
                }

                // Found a back edge.
                // Merge all bcc's on the root path, and extract external boundary.
                self.back_edge_flag[z as usize] = UNSET;
                while let Some((s, ps, fs)) = self.merge_stack.pop() {
                    a = self.pertinent_roots.pop_right(s).unwrap();

                    let c = a - n as u32;
                    self.sep_children.isolate(c);

                    // ## Structure of bcc boundary:
                    //     Lower cycle      pz-z-...-fz-a-...-pz
                    //     Upper cycle      ps-s-...-fs-...-ps
                    //     After merging    ps-z-...-fz-s-...-fs-...-ps
                    self.boundary.split_slice_out(pz, z, fz, a);
                    self.boundary.insert_slice_between(ps, z, fz, s);

                    pz = ps;
                    fz = fs;
                }

                a = bcc_root;
                let ez = self.boundary.xor(bcc_root) ^ fz;
                self.boundary.split_slice_out(pz, z, bcc_root, ez);
            }

            true
        }

        fn run(&mut self) -> bool {
            for t in (0..self.n_verts() as u32).rev() {
                let u = self.dfs.tour[t as usize];

                for e in self.neighbors.edge_range(u as usize) {
                    let w = self.neighbors.link[e];
                    if self.dfs.t_in[u as usize] < self.dfs.t_in[w as usize]
                        && self.dfs.parent_edge[w as usize] != e as u32
                    {
                        // Back edge
                        self.walk_up(u, w);
                    }
                }

                while let Some(r) = self.pertinent_roots.pop_right(u) {
                    if !self.walk_down(u, r) {
                        return false;
                    }
                }
            }

            true
        }
    }

    pub fn is_planar(n_verts: usize, edges: &[[u32; 2]]) -> bool {
        debug_assert!(
            edges
                .iter()
                .flat_map(|&[u, v]| [[u, v], [v, u]])
                .collect::<std::collections::HashSet<_>>()
                .len()
                == edges.len() * 2,
            "The input graph is a multigraph"
        );

        let neighbors = CSR::from_pairs(
            n_verts,
            edges
                .iter()
                .filter(|&[u, v]| u != v)
                .flat_map(|&[u, v]| [(u, v), (v, u)]),
        );
        let mut cx = BoyerMyrvold::from_assoc_list(neighbors);
        let res = cx.run();

        res
    }
}
