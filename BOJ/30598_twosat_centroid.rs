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

pub mod jagged {
    use std::fmt::Debug;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<T>: IndexMut<usize, Output = [T]> {
        fn len(&self) -> usize;
    }

    impl<T, C> Jagged<T> for C
    where
        C: AsRef<[Vec<T>]> + IndexMut<usize, Output = [T]>,
    {
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
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
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
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
        pub fn from_pairs(n: usize, pairs: impl Iterator<Item = (u32, T)> + Clone) -> Self {
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

            CSR { data, head }
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> Jagged<T> for CSR<T> {
        fn len(&self) -> usize {
            self.head.len() - 1
        }
    }
}

fn gen_scc(neighbors: &impl jagged::Jagged<u32>) -> (usize, Vec<u32>) {
    // Tarjan algorithm, iterative
    let n = neighbors.len();

    const UNSET: u32 = u32::MAX;
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

            if iv < neighbors[u as usize].len() as u32 {
                // Iterate neighbors
                dfs_stack.push((u, iv + 1));

                let v = neighbors[u as usize][iv as usize];
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
        let (scc_count, scc_index) = gen_scc(&jagged::CSR::from_pairs(
            self.n_props * 2,
            self.edges.iter().copied(),
        ));

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

const UNSET: u32 = !0;

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
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    'outer: for _ in 0..input.value() {
        let n: usize = input.value();
        let mut groups = vec![[UNSET; 2]; n];
        for u in 0..n * 2 {
            let c = input.value::<usize>() - 1;
            let i = (groups[c][0] != UNSET) as usize;
            groups[c][i] = u as u32;
        }
        let n_verts = 2 * n;

        let mut degree = vec![0u32; n_verts];
        let mut xor_neighbors = vec![0u32; n_verts];
        for _ in 0..n_verts - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            degree[u as usize] += 1;
            degree[v as usize] += 1;
            xor_neighbors[u as usize] ^= v;
            xor_neighbors[v as usize] ^= u;
        }

        let (toposort, parent) = xor_traversal(degree.clone(), xor_neighbors.clone(), 0);
        let mut size = vec![1u32; n_verts];
        for &u in &toposort[..n_verts - 1] {
            size[parent[u as usize] as usize] += size[u as usize];
        }

        let mut centroids = vec![];
        {
            // Reroot down to the lowest centroid
            let mut c = 0;
            let threshold = (n_verts as u32 + 1) / 2;
            for u in toposort.into_iter().rev() {
                let p = parent[u as usize] as usize;
                if p == c && size[u as usize] >= threshold {
                    size[p as usize] -= size[u as usize];
                    size[u as usize] += size[p as usize];
                    c = u as usize;
                }
            }
            centroids.push(c);

            let p = parent[c] as usize;
            if p != c && size[p] >= threshold {
                centroids.push(p);
            }
        }

        for c in centroids {
            let (_toposort, parent) =
                xor_traversal(degree.clone(), xor_neighbors.clone(), c as u32);

            let mut formula = TwoSat::new(2 * n);
            for &[u, v] in &groups {
                formula.add_disj((u, true), (v, true));
                formula.add_disj((u, false), (v, false));
            }

            for u in 0..n_verts as u32 {
                let p = parent[u as usize];
                if u == p {
                    continue;
                }
                formula.add_disj((u, false), (p, true));
            }

            if let Some(interpretation) = formula.solve() {
                for u in 0..n_verts {
                    if interpretation[u] {
                        write!(output, "{} ", u + 1).unwrap();
                    }
                }
                writeln!(output).unwrap();
                continue 'outer;
            }
        }
        writeln!(output, "-1").unwrap();
    }
}
