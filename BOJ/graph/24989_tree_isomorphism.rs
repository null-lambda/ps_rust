use std::{collections::HashMap, io::Write};

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

pub mod tree_iso {
    // AHU algorithm for classifying (rooted) trees up to isomorphism
    // Time complexity: O(N log N)    (O(N) with radix sort))
    pub type Code<E> = Box<[(u32, E)]>;

    #[derive(Debug, Clone)]
    pub struct AHULabelCompression<E> {
        // Extracted classification codes
        pub forest_roots: (Vec<usize>, E),
        pub levels: Vec<Vec<Code<E>>>,

        // Auxiliary information for isomorphism construction
        pub ordered_children: Vec<Vec<(u32, E)>>,
        pub depth: Vec<u32>,
        pub code_in_parent: Vec<u32>,
        pub bfs_tour: Vec<u32>,
    }

    impl<E: Clone + Ord + Default> AHULabelCompression<E> {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = (u32, u32, E)>,
            root: Option<usize>, // None for unrooted tree isomorphism
        ) -> Self {
            assert!(n_verts > 0);

            // Build the tree structure
            let base_root = root.unwrap_or(0) as usize;
            let mut degree = vec![0u32; n_verts];
            let mut xor_neighbors = vec![0u32; n_verts];
            let mut neighbors = vec![vec![]; n_verts];
            for (u, v, w) in edges {
                debug_assert!(u < n_verts as u32);
                debug_assert!(v < n_verts as u32);
                degree[u as usize] += 1;
                degree[v as usize] += 1;
                xor_neighbors[u as usize] ^= v;
                xor_neighbors[v as usize] ^= u;
                neighbors[u as usize].push((v, w.clone()));
                neighbors[v as usize].push((u, w));
            }
            degree[base_root] += 2;

            // Upward propagation
            let mut size = vec![1u32; n_verts];
            let mut topological_order = vec![];
            for mut u in 0..n_verts as u32 {
                while degree[u as usize] == 1 {
                    let p = xor_neighbors[u as usize];
                    xor_neighbors[p as usize] ^= u;
                    degree[u as usize] -= 1;
                    degree[p as usize] -= 1;
                    topological_order.push(u);

                    size[p as usize] += size[u as usize];

                    u = p;
                }
            }
            assert!(
                topological_order.len() == n_verts - 1,
                "Invalid tree structure"
            );
            let parent = xor_neighbors;

            let mut forest_roots = if let Some(root) = root {
                (vec![root], E::default())
            } else {
                // For unrooted tree classification, assign some distinguished roots.

                // Reroot down to the lowest centroid
                let mut root = base_root;
                let threshold = (n_verts as u32 + 1) / 2;
                for u in topological_order.into_iter().rev() {
                    let p = parent[u as usize] as usize;
                    if p == root && size[u as usize] >= threshold {
                        size[p as usize] -= size[u as usize];
                        size[u as usize] += size[p as usize];
                        root = u as usize;
                    }
                }

                // Check the direct parent for another centroid
                let mut aux_root = None;
                let p = parent[root];
                if p != root as u32 && size[p as usize] >= threshold {
                    aux_root = Some(p as usize);
                }

                if let Some(aux_root) = aux_root {
                    // Split the double-centroid tree into a two-component forest
                    let i = neighbors[root]
                        .iter()
                        .position(|&(v, _)| v as usize == aux_root)
                        .unwrap();
                    let (_, w) = neighbors[root].swap_remove(i);
                    neighbors[root].retain(|&(v, _)| v != aux_root as u32);
                    neighbors[aux_root].retain(|&(v, _)| v != root as u32);
                    size[root] -= size[aux_root];

                    (vec![root, aux_root], w)
                } else {
                    (vec![root], E::default())
                }
            };

            // Downward propagation
            let mut depth = vec![0u32; n_verts];
            let mut bfs_tour: Vec<_> = forest_roots.0.iter().map(|&u| u as u32).collect();
            let mut timer = 0;
            while let Some(&u) = bfs_tour.get(timer) {
                timer += 1;
                neighbors[u as usize]
                    .iter()
                    .position(|&(v, _)| size[v as usize] > size[u as usize])
                    .map(|ip| neighbors[u as usize].swap_remove(ip));
                for &(v, _) in &neighbors[u as usize] {
                    depth[v as usize] = depth[u as usize] + 1;
                    bfs_tour.push(v);
                }
            }

            // Encode the forest starting from the deepest level
            let mut code_in_parent: Vec<u32> = vec![Default::default(); n_verts];
            let mut levels: Vec<_> = vec![];
            let (mut start, mut end) = (n_verts, n_verts);
            for d in (0..n_verts as u32).rev() {
                while start > 0 && depth[bfs_tour[start - 1] as usize] >= d {
                    start -= 1;
                }
                let level_nodes = &bfs_tour[start..end];
                end = start;

                let mut codes_d: Vec<(u32, Code<E>)> = level_nodes
                    .iter()
                    .map(|&u| {
                        neighbors[u as usize].sort_unstable_by(|(v1, w1), (v2, w2)| {
                            code_in_parent[*v1 as usize]
                                .cmp(&code_in_parent[*v2 as usize])
                                .then_with(|| w1.cmp(w2))
                        });
                        let code: Code<E> = neighbors[u as usize]
                            .iter()
                            .map(|(v, w)| (code_in_parent[*v as usize], w.clone()))
                            .collect();
                        (u, code)
                    })
                    .collect();
                codes_d.sort_unstable_by(|(_, code_a), (_, code_b)| code_a.cmp(code_b));

                let mut c = 0u32;
                let mut prev_code = None;
                for (u, code) in &codes_d {
                    if prev_code.is_some() && prev_code != Some(code) {
                        c += 1;
                    }
                    code_in_parent[*u as usize] = c;
                    prev_code = Some(code);
                }
                levels.push(codes_d.into_iter().map(|(_, code)| code).collect());
            }
            let ordered_children = neighbors;
            forest_roots.0.sort_unstable_by_key(|&u| code_in_parent[u]);

            Self {
                forest_roots,
                levels,

                ordered_children,
                depth,
                code_in_parent,
                bfs_tour,
            }
        }

        pub fn is_iso_to(&self, other: &Self) -> bool {
            self.forest_roots.0.len() == other.forest_roots.0.len()
                && self.forest_roots.1 == other.forest_roots.1
                && self.levels == other.levels
        }

        pub fn cmp_by_labels(&self, other: &Self) -> std::cmp::Ordering {
            (self.forest_roots.0.len().cmp(&other.forest_roots.0.len()))
                .then_with(|| self.forest_roots.1.cmp(&other.forest_roots.1))
                .then_with(|| self.levels.cmp(&other.levels))
        }

        pub fn get_mapping(&self, other: &Self) -> Option<Vec<u32>> {
            if !self.is_iso_to(other) {
                return None;
            }

            const UNSET: u32 = !0;
            let mut mapping = vec![UNSET; self.ordered_children.len()];

            let mut queue = vec![];
            for (&u1, &u2) in self.forest_roots.0.iter().zip(&other.forest_roots.0) {
                mapping[u1] = u2 as u32;
                queue.push(u1 as u32);
            }

            let mut timer = 0;
            while let Some(&u1) = queue.get(timer) {
                timer += 1;
                let u2 = mapping[u1 as usize];

                for (&(v1, _), &(v2, _)) in self.ordered_children[u1 as usize]
                    .iter()
                    .zip(&other.ordered_children[u2 as usize])
                {
                    mapping[v1 as usize] = v2;
                    queue.push(v1);
                }
            }
            debug_assert!(mapping.iter().all(|&v| v != UNSET));

            Some(mapping)
        }
    }
}

const UNSET: u32 = !0;

pub fn tree_divisor<'a>(
    n: usize,
    bfs_order: &[(u32, u32)],
    children: &'a impl jagged::Jagged<'a, u32>,
    subtree_size: usize,
) -> Option<tree_iso::AHULabelCompression<()>> {
    if subtree_size > n || n % subtree_size != 0 {
        return None;
    }

    let subtree_size = subtree_size as u32;
    let mut residual_size = vec![1u32; n];
    let mut subtree0 = None;
    for &(u, p) in bfs_order.iter().rev() {
        if residual_size[u as usize] > subtree_size {
            return None;
        } else if residual_size[u as usize] == subtree_size {
            residual_size[u as usize] = 0;

            // Dfs through residual size > 0
            let mut induced_edges = vec![];
            let mut stack = vec![u];
            while let Some(u) = stack.pop() {
                for &v in children.get(u as usize) {
                    if residual_size[v as usize] > 0 {
                        induced_edges.push((u, v));
                        stack.push(v);
                    }
                }
            }

            let mut index_map: HashMap<u32, u32> = Default::default();
            let mut relabel = |u: u32| -> u32 {
                let idx = index_map.len() as u32;
                *index_map.entry(u).or_insert_with(|| idx)
            };
            relabel(u);
            for u in induced_edges.iter_mut().flat_map(|(u, v)| [u, v]) {
                *u = relabel(*u);
            }

            let comp = tree_iso::AHULabelCompression::from_edges(
                subtree_size as usize,
                induced_edges.into_iter().map(|(u, v)| (u, v, ())),
                None,
            );
            if subtree0.is_none() {
                subtree0 = Some(comp);
            } else if !subtree0.as_ref().unwrap().is_iso_to(&comp) {
                return None;
            }
        }

        if p != UNSET {
            residual_size[p as usize] += residual_size[u as usize];
        }
    }

    subtree0
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let t: usize = input.value();
    let mut trees = vec![];
    let n_max = 400_002;
    let mut cs = vec![vec![]; n_max + 1];
    for _ in 0..t {
        let n: usize = input.value();
        let mut neighbors = vec![vec![]; n];
        for _ in 0..n - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            neighbors[u as usize].push(v);
            neighbors[v as usize].push(u);
        }

        let mut bfs_order = vec![(0, UNSET)];
        let mut timer = 0;
        while let Some(&(u, p)) = bfs_order.get(timer) {
            timer += 1;
            neighbors[u as usize]
                .iter()
                .position(|&v| v == p)
                .map(|i| neighbors[u as usize].swap_remove(i));
            for &v in &neighbors[u as usize] {
                bfs_order.push((v, u));
            }
        }
        let children = jagged::CSR::from_iter(neighbors);

        let comp = tree_iso::AHULabelCompression::from_edges(
            n,
            bfs_order[1..].iter().map(|&(u, p)| (u, p, ())),
            None,
        );
        trees.push((n, bfs_order, children));
        cs[n].push((comp, 1));
    }

    for n in 1..=n_max {
        cs[n].sort_unstable_by(|(c1, _), (c2, _)| c1.cmp_by_labels(c2));
        cs[n].dedup_by(|(c1, mult1), (c2, mult2)| {
            if c1.is_iso_to(c2) {
                *mult2 += *mult1;
                true
            } else {
                false
            }
        });
    }

    for (n, bfs_order, children) in trees {
        let mut ans = 0u32;
        for d in 1..=n {
            if n % d != 0 || cs[d].is_empty() {
                continue;
            }

            let Some(subtree0) = tree_divisor(n, &bfs_order, &children, d) else {
                continue;
            };

            // binary search for isomorphic element
            let Ok(i) = cs[d].binary_search_by(|(c, _)| c.cmp_by_labels(&subtree0)) else {
                continue;
            };
            let (_, mult) = &mut cs[d][i];
            ans += *mult;
        }
        ans -= 1;
        write!(output, "{} ", ans).unwrap();
    }
    writeln!(output).unwrap();
}
