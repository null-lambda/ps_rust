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

pub mod hld {
    // Heavy-Light Decomposition
    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        #[cold]
        #[inline(always)]
        pub fn cold() {}

        if !b {
            cold();
        }
        b
    }

    const UNSET: u32 = u32::MAX;

    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub segmented_idx: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
            use_dfs_ordering: bool,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                xor_neighbors[u as usize] ^= v;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = Vec::with_capacity(n);
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Topological sort
                    let p = xor_neighbors[u];
                    topological_order.push(u as u32);
                    degree[u] = 0;
                    degree[p as usize] -= 1;
                    xor_neighbors[p as usize] ^= u as u32;

                    // Upward propagation
                    size[p as usize] += size[u as usize];
                    let h = &mut heavy_child[p as usize];
                    if *h == UNSET || size[*h as usize] < size[u as usize] {
                        *h = u as u32;
                    }

                    assert!(u != p as usize);
                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            assert!(topological_order.len() == n, "Invalid tree structure");

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut chain_top = vec![root as u32; n];
            let mut segmented_idx = vec![UNSET; n];
            if !use_dfs_ordering {
                // A rearranged topological index continuous in a chain, for path queries
                let mut timer = 0;
                for mut u in topological_order.into_iter().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }
                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        segmented_idx[u as usize] = timer;
                        timer += 1;
                        u = heavy_child[u as usize];
                        if u == UNSET {
                            break;
                        }
                    }
                }
            } else {
                // DFS ordering for path & subtree queries
                let mut offset = vec![0; n];
                for mut u in topological_order.into_iter().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }

                    let mut p = parent[u as usize];
                    let mut timer = 0;
                    if likely(p != UNSET) {
                        timer = offset[p as usize] + 1;
                        offset[p as usize] += size[u as usize] as u32;
                    }

                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        offset[u as usize] = timer;
                        segmented_idx[u as usize] = timer;
                        timer += 1;

                        p = u as u32;
                        u = heavy_child[p as usize];
                        unsafe { assert_unchecked(u != p) };
                        if u == UNSET {
                            break;
                        }
                        offset[p as usize] += size[u as usize] as u32;
                    }
                }
            }

            Self {
                size,
                parent,
                heavy_child,
                chain_top,
                segmented_idx,
            }
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    > self.segmented_idx[self.chain_top[v] as usize]
                {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
    }
}

pub mod trie {
    use std::collections::HashMap;
    use std::hash::Hash;

    pub const UNSET: u32 = !0;
    pub const UNSET_KEY: u32 = !0 - 2;
    pub type NodeRef = u32;

    // The most generic one
    impl<K> TransitionMap for HashMap<K, NodeRef>
    where
        K: Eq + Hash,
    {
        type Key = K;

        fn empty() -> Self {
            Default::default()
        }

        fn get(&self, key: &Self::Key) -> NodeRef {
            self.get(key).copied().unwrap_or(UNSET)
        }

        fn insert(&mut self, key: K, value: NodeRef) {
            self.insert(key, value);
        }
    }

    // Fixed-size array map
    impl<const N_ALPHABETS: usize> TransitionMap for [NodeRef; N_ALPHABETS] {
        type Key = usize;

        fn empty() -> Self {
            std::array::from_fn(|_| Default::default())
        }

        fn get(&self, key: &Self::Key) -> NodeRef {
            self[*key]
        }

        fn insert(&mut self, key: usize, value: NodeRef) {
            self[key] = value;
        }
    }

    // Adaptive array map, based on the fact that most nodes are slim.
    pub enum AdaptiveArrayMap<K, const N_ALPHABETS: usize, const STACK_CAP: usize> {
        Small([(K, NodeRef); STACK_CAP]),
        Large(Box<[NodeRef; N_ALPHABETS]>),
    }

    impl<
            K: Into<usize> + Copy + Default + Eq,
            const N_ALPHABETS: usize,
            const STACK_CAP: usize,
        > TransitionMap for AdaptiveArrayMap<K, N_ALPHABETS, STACK_CAP>
    {
        type Key = K;

        fn empty() -> Self {
            assert!(1 <= STACK_CAP && STACK_CAP <= N_ALPHABETS);
            Self::Small(std::array::from_fn(|_| (Default::default(), UNSET)))
        }

        fn get(&self, key: &Self::Key) -> NodeRef {
            match self {
                Self::Small(assoc_list) => assoc_list
                    .iter()
                    .find_map(|(k, v)| (*k == *key).then(|| *v))
                    .unwrap_or(UNSET),
                Self::Large(array_map) => array_map[(*key).into()],
            }
        }

        fn insert(&mut self, key: Self::Key, value: NodeRef) {
            match self {
                Self::Small(assoc_list) => {
                    for (k, v) in assoc_list.iter_mut() {
                        if *k == key {
                            *v = value;
                            return;
                        } else if *v == UNSET_KEY {
                            *k = key;
                            *v = value;
                            return;
                        }
                    }

                    let mut array_map = Box::new([UNSET; N_ALPHABETS]);
                    for (k, v) in assoc_list {
                        array_map[(*k).into()] = *v;
                    }
                    array_map[key.into()] = value;
                    *self = Self::Large(array_map);
                }
                Self::Large(array_map) => {
                    array_map[key.into()] = value;
                }
            }
        }
    }

    impl<K: Copy + Default + Eq + From<usize>, const N: usize, const STACK_CAP: usize>
        AdaptiveArrayMap<K, N, STACK_CAP>
    {
        pub fn for_each(&self, mut f: impl FnMut(&K, NodeRef)) {
            match self {
                Self::Small(assoc_list) => {
                    for (k, v) in assoc_list {
                        if *v != UNSET {
                            f(k, *v);
                        }
                    }
                }
                Self::Large(array_map) => {
                    for (i, &v) in array_map.iter().enumerate() {
                        if v != UNSET {
                            f(&i.into(), v);
                        }
                    }
                }
            }
        }
    }

    // Generic interface for different associative containers
    pub trait TransitionMap {
        type Key;
        fn empty() -> Self;
        fn get(&self, key: &Self::Key) -> NodeRef;
        fn insert(&mut self, key: Self::Key, value: NodeRef);
    }

    #[derive(Debug, Clone)]
    pub struct Node<M> {
        pub children: M,
    }

    #[derive(Debug)]
    pub struct Trie<M> {
        pub pool: Vec<Node<M>>,
    }

    impl<M: TransitionMap> Trie<M> {
        pub fn new() -> Self {
            let root = Node {
                children: M::empty(),
            };
            Self { pool: vec![root] }
        }

        fn alloc(&mut self) -> NodeRef {
            let idx = self.pool.len() as u32;
            self.pool.push(Node {
                children: M::empty(),
            });
            idx
        }

        pub fn insert(&mut self, path: impl IntoIterator<Item = M::Key>) -> NodeRef {
            let mut u = 0;
            for c in path {
                let next = self.pool[u as usize].children.get(&c);
                if next == UNSET {
                    let new_node = self.alloc();
                    self.pool[u as usize].children.insert(c, new_node);
                    u = new_node;
                } else {
                    u = next;
                }
            }
            u
        }

        pub fn find(&mut self, path: impl IntoIterator<Item = M::Key>) -> Option<NodeRef> {
            let mut u = 0;
            for c in path {
                let next = self.pool[u as usize].children.get(&c);
                if next == UNSET {
                    return None;
                }
                u = next;
            }
            Some(u)
        }
    }
}

type AlphabetMap = trie::AdaptiveArrayMap<usize, 26, 1>;

fn parse_byte(c: u8) -> usize {
    (c - b'a') as usize
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut trie = trie::Trie::<AlphabetMap>::new();
    let mut node_idx = vec![];
    for _ in 0..n {
        let word = input.token().bytes().map(parse_byte);
        let u = trie.insert(word);
        node_idx.push(u);
    }
    let n_nodes = trie.pool.len();
    let mut parent = vec![0; n_nodes];

    let mut edges = vec![];
    let mut depth = vec![0; n_nodes];
    for u in 0..n_nodes {
        trie.pool[u].children.for_each(|&_, v| {
            edges.push((u as u32, v));
            parent[v as usize] = u as u32;
            depth[v as usize] = depth[u as usize] + 1;
        });
    }

    let hld = hld::HLD::from_edges(n_nodes, edges.iter().copied(), 0, true);
    let dist = |u, v| {
        let lca = hld.lca(u, v);
        depth[u] + depth[v] - 2 * depth[lca]
    };
    let euler_in = |u| hld.segmented_idx[u] as usize;

    let mut count = vec![0; n_nodes];
    for _ in 0..q {
        let k: usize = input.value();
        let l: u32 = input.value();

        let mut relevant: Vec<u32> = (0..k)
            .map(|_| node_idx[input.value::<usize>() - 1])
            .collect();
        relevant.push(0);
        for &u in &relevant {
            count[u as usize] += 1;
        }

        relevant.sort_unstable_by_key(|&u| euler_in(u as usize));
        for i in 1..relevant.len() {
            relevant.push(hld.lca(relevant[i - 1] as usize, relevant[i] as usize) as u32);
        }
        relevant.sort_unstable_by_key(|&u| euler_in(u as usize));
        relevant.dedup();

        let mut ans = 0;
        for i in (1..relevant.len()).rev() {
            let u = relevant[i] as usize;

            let p = hld.lca(relevant[i - 1] as usize, u) as usize;
            let w = dist(u, p);

            if count[u] == l {
                ans += w;
            }

            count[p] += count[u];
        }
        writeln!(output, "{}", ans).unwrap();

        for &u in &relevant {
            count[u as usize] = 0;
        }
    }
}
