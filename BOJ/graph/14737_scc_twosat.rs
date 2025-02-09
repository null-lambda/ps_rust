use std::io::Write;

use jagged::Jagged;

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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> &'a [T];
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> &'a [T] {
            &self.as_ref()[u]
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len())
                .map(|i| self.get(i).iter().collect())
                .collect();
            v.fmt(f)
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
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> &'a [T] {
            &self.data[self.head[u] as usize..self.head[u + 1] as usize]
        }
    }
}

const UNSET: u32 = u32::MAX;
fn gen_scc<'a>(neighbors: &'a impl jagged::Jagged<'a, u32>) -> (usize, Vec<u32>) {
    // Tarjan algorithm, iterative
    let n = neighbors.len();

    let mut scc_index: Vec<u32> = vec![UNSET; n];
    let mut scc_count = 0;

    let mut path_stack = vec![];
    let mut dfs_stack = vec![];
    let mut order_count: u32 = 1;
    let mut order: Vec<u32> = vec![0; n];
    let mut low_link: Vec<u32> = vec![UNSET; n];

    for u in 0..n {
        if order[u] > 0 {
            continue;
        }

        const UPDATE_LOW_LINK: u32 = 1 << 31;

        dfs_stack.push((u as u32, 0));
        while let Some((u, iv)) = dfs_stack.pop() {
            if iv & UPDATE_LOW_LINK != 0 {
                let v = iv ^ UPDATE_LOW_LINK;
                low_link[u as usize] = low_link[u as usize].min(low_link[v as usize]);
                continue;
            }

            if iv == 0 {
                // Enter node
                order[u as usize] = order_count;
                low_link[u as usize] = order_count;
                order_count += 1;
                path_stack.push(u);
            }

            if iv < neighbors.get(u as usize).len() as u32 {
                // Iterate neighbors
                dfs_stack.push((u, iv + 1));

                let v = neighbors.get(u as usize)[iv as usize];
                if order[v as usize] == 0 {
                    dfs_stack.push((u, v | UPDATE_LOW_LINK));
                    dfs_stack.push((v, 0));
                } else if scc_index[v as usize] == UNSET {
                    low_link[u as usize] = low_link[u as usize].min(order[v as usize]);
                }
            } else {
                // Exit node
                if low_link[u as usize] == order[u as usize] {
                    // Found a strongly connected component
                    loop {
                        let v = path_stack.pop().unwrap();
                        scc_index[v as usize] = scc_count;
                        if v == u {
                            break;
                        }
                    }
                    scc_count += 1;
                }
            }
        }
    }
    (scc_count as usize, scc_index)
}

pub struct TwoSat {
    n_props: usize,
    edges: Vec<(u32, u32)>,
}

impl TwoSat {
    pub fn new(n_props: usize) -> Self {
        Self {
            n_props,
            edges: vec![],
        }
    }

    pub fn add_disj(&mut self, (p, bp): (u32, bool), (q, bq): (u32, bool)) {
        self.edges
            .push((self.prop_to_node((p, !bp)), self.prop_to_node((q, bq))));
        self.edges
            .push((self.prop_to_node((q, !bq)), self.prop_to_node((p, bp))));
    }

    fn prop_to_node(&self, (p, bp): (u32, bool)) -> u32 {
        debug_assert!(p < self.n_props as u32);
        if bp {
            p
        } else {
            self.n_props as u32 + p
        }
    }

    fn node_to_prop(&self, node: u32) -> (u32, bool) {
        if node < self.n_props as u32 {
            (node, true)
        } else {
            (node - self.n_props as u32, false)
        }
    }

    pub fn solve(&self) -> Option<Vec<bool>> {
        let (scc_count, scc_index) =
            gen_scc(&jagged::CSR::from_assoc_list(self.n_props * 2, &self.edges));

        let mut scc = vec![vec![]; scc_count];
        for (i, &scc_idx) in scc_index.iter().enumerate() {
            scc[scc_idx as usize].push(i as u32);
        }

        let satisfiable = (0..self.n_props as u32).all(|p| {
            scc_index[self.prop_to_node((p, true)) as usize]
                != scc_index[self.prop_to_node((p, false)) as usize]
        });
        if !satisfiable {
            return None;
        }

        let mut interpretation = vec![None; self.n_props];
        for component in &scc {
            for &i in component.iter() {
                let (p, p_value) = self.node_to_prop(i);
                if interpretation[p as usize].is_some() {
                    break;
                }
                interpretation[p as usize] = Some(p_value);
            }
        }
        Some(interpretation.into_iter().map(|x| x.unwrap()).collect())
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let h: usize = input.value();
    let w: usize = input.value();

    let mut grid = vec![];
    for _ in 0..h {
        let row: Vec<u8> = input.token().bytes().collect();
        grid.extend(row[..w].iter().copied());
    }

    let row = |i: usize| i * w..(i + 1) * w;
    let mut cell_to_nodes = vec![vec![]; h * w];
    let mut node_ends = vec![];
    let wall = |i, j| grid[row(i)][j] == b'#';

    for i in 0..h {
        let mut j0 = 0;
        while j0 < w {
            if wall(i, j0) {
                j0 += 1;
                continue;
            }

            let mut j1 = j0;
            while j1 + 1 < w && !wall(i, j1 + 1) {
                j1 += 1;
            }

            let node_idx = node_ends.len();
            for j in j0..=j1 {
                cell_to_nodes[row(i)][j].push(node_idx);
            }
            node_ends.push(vec![row(i).start + j0, row(i).start + j1]);

            j0 = j1 + 1;
        }
    }
    let row_node_range = 0..node_ends.len();

    for j in 0..w {
        let mut i0 = 0;
        while i0 < h {
            if wall(i0, j) {
                i0 += 1;
                continue;
            }

            let mut i1 = i0;
            while i1 + 1 < h && !wall(i1 + 1, j) {
                i1 += 1;
            }

            let node_idx = node_ends.len();
            for i in i0..=i1 {
                cell_to_nodes[row(i)][j].push(node_idx);
            }
            node_ends.push(vec![row(i0).start + j, row(i1).start + j]);

            i0 = i1 + 1
        }
    }
    let col_node_range = row_node_range.end..node_ends.len();

    let n_nodes = node_ends.len() + 1;
    let root = n_nodes - 1;

    let mut edges = vec![];
    let z_init = (0..h * w).find(|&u| grid[u] == b'O').unwrap();
    edges.push((root as u32, (cell_to_nodes[z_init][0] as u32)));
    edges.push((root as u32, (cell_to_nodes[z_init][1] as u32)));

    for u in row_node_range.clone() {
        let zs = &node_ends[u];
        edges.push((u as u32, cell_to_nodes[zs[0]][1] as u32));
        if zs[0] != zs[1] {
            edges.push((u as u32, cell_to_nodes[zs[1]][1] as u32));
        }
    }
    for u in col_node_range.clone() {
        let zs = &node_ends[u];
        edges.push((u as u32, cell_to_nodes[zs[0]][0] as u32));
        if zs[0] != zs[1] {
            edges.push((u as u32, cell_to_nodes[zs[1]][0] as u32));
        }
    }
    let neighbors = jagged::CSR::from_assoc_list(n_nodes, &edges);

    let (scc_count, scc_index) = gen_scc(&neighbors);
    let mut sccs = vec![vec![]; scc_count];
    for (i, &scc_idx) in scc_index.iter().enumerate() {
        sccs[scc_idx as usize].push(i as u32);
    }

    let mut scc_edges = vec![];
    for &(u, v) in &edges {
        let scc_u = scc_index[u as usize];
        let scc_v = scc_index[v as usize];
        if scc_u != scc_v {
            scc_edges.push((scc_u, scc_v));
        }
    }
    scc_edges.sort_unstable();
    scc_edges.dedup();
    let scc_neighbors = jagged::CSR::from_assoc_list(scc_count, &scc_edges);

    let mut reachable = vec![vec![false; scc_count]; scc_count];
    for init in 0..scc_count {
        let mut stack = vec![init as u32];
        reachable[init][init] = true;
        while let Some(u) = stack.pop() {
            for &v in scc_neighbors.get(u as usize) {
                if !reachable[init][v as usize] {
                    reachable[init][v as usize] = true;
                    stack.push(v);
                }
            }
        }
    }

    let mut scc_goals = vec![];
    for i in 0..h {
        for j in 0..w {
            if grid[row(i)][j] == b'*' {
                let zs = &cell_to_nodes[row(i)][j];
                scc_goals.push((scc_index[zs[0] as usize], scc_index[zs[1] as usize]));
            }
        }
    }
    scc_goals.sort_unstable();
    scc_goals.dedup();

    let mut twosat = TwoSat::new(scc_count);
    let mut add_tauto = |prop| twosat.add_disj(prop, prop);
    add_tauto((scc_index[root], true));
    for c in 0..scc_count {
        if !reachable[scc_index[root] as usize][c] {
            add_tauto((c as u32, false));
        }
    }

    for cs in scc_goals {
        twosat.add_disj((cs.0 as u32, true), (cs.1 as u32, true));
    }

    for c in 0..scc_count {
        for d in 0..scc_count {
            if !reachable[c][d] && !reachable[d][c] {
                twosat.add_disj((c as u32, false), (d as u32, false));
            }
        }
    }

    let ans = twosat.solve().is_some();
    writeln!(output, "{}", if ans { "YES" } else { "NO" }).unwrap();

    {
        // let mut grid = grid.clone();
        // for (c, scc) in sccs.iter().enumerate().rev() {
        //     if !reachable[scc_index[root] as usize][c] {
        //         continue;
        //     }
        //     for &u in scc {
        //         if u == root as u32 {
        //             continue;
        //         }
        //         let zs = node_ends[u as usize].clone();
        //         if row_node_range.contains(&(u as usize)) {
        //             for z in zs[0]..=zs[1] {
        //                 grid[z] = (c % 10) as u8 + b'0';
        //             }
        //         } else {
        //             for z in (zs[0]..=zs[1]).step_by(w) {
        //                 grid[z] = (c % 10) as u8 + b'0';
        //             }
        //         }
        //     }

        //     for i in 0..h {
        //         println!("{}", std::str::from_utf8(&grid[row(i)]).unwrap());
        //     }
        //     println!();
        // }
    }
}
