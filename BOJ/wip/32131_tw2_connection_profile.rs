use std::{cmp::Ordering, collections::HashMap, io::Write};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

pub mod map {
    use std::{hash::Hash, mem::MaybeUninit};

    pub enum AdaptiveHashSet<K, const STACK_CAP: usize> {
        Small([MaybeUninit<K>; STACK_CAP], usize),
        Large(std::collections::HashSet<K>),
    }

    impl<K, const STACK_CAP: usize> Drop for AdaptiveHashSet<K, STACK_CAP> {
        fn drop(&mut self) {
            match self {
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe { arr[i].assume_init_drop() }
                    }
                }
                _ => {}
            }
        }
    }

    impl<K: Clone, const STACK_CAP: usize> Clone for AdaptiveHashSet<K, STACK_CAP> {
        fn clone(&self) -> Self {
            match self {
                Self::Small(arr, size) => {
                    let mut cloned = std::array::from_fn(|_| MaybeUninit::uninit());
                    for i in 0..*size {
                        cloned[i] = MaybeUninit::new(unsafe { arr[i].assume_init_ref().clone() });
                    }
                    Self::Small(cloned, *size)
                }
                Self::Large(set) => Self::Large(set.clone()),
            }
        }
    }

    impl<K, const STACK_CAP: usize> Default for AdaptiveHashSet<K, STACK_CAP> {
        fn default() -> Self {
            Self::Small(std::array::from_fn(|_| MaybeUninit::uninit()), 0)
        }
    }

    impl<K: Eq + Hash, const STACK_CAP: usize> AdaptiveHashSet<K, STACK_CAP> {
        pub fn len(&self) -> usize {
            match self {
                Self::Small(_, size) => *size,
                Self::Large(set) => set.len(),
            }
        }

        pub fn contains(&self, key: &K) -> bool {
            match self {
                Self::Small(arr, size) => arr[..*size]
                    .iter()
                    .find(|&x| unsafe { x.assume_init_ref() } == key)
                    .is_some(),
                Self::Large(set) => set.contains(key),
            }
        }

        pub fn insert(&mut self, key: K) -> bool {
            if self.contains(&key) {
                return false;
            }
            match self {
                Self::Small(arr, size) => {
                    if arr[..*size]
                        .iter()
                        .find(|&x| unsafe { x.assume_init_ref() } == &key)
                        .is_some()
                    {
                        return false;
                    }

                    if *size < STACK_CAP {
                        arr[*size] = MaybeUninit::new(key);
                        *size += 1;
                    } else {
                        let arr =
                            std::mem::replace(arr, std::array::from_fn(|_| MaybeUninit::uninit()));
                        *size = 0; // Prevent `drop` call on arr elements
                        *self = Self::Large(
                            arr.into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .chain(Some(key))
                                .collect(),
                        );
                    }
                    true
                }
                Self::Large(set) => set.insert(key),
            }
        }

        pub fn remove(&mut self, key: &K) -> bool {
            match self {
                Self::Small(_, 0) => false,
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe {
                            if arr[i].assume_init_ref() == key {
                                *size -= 1;
                                arr[i].assume_init_drop();
                                arr[i] = std::mem::replace(&mut arr[*size], MaybeUninit::uninit());
                                return true;
                            }
                        }
                    }

                    false
                }
                Self::Large(set) => set.remove(key),
            }
        }

        pub fn for_each(&mut self, mut visitor: impl FnMut(&K)) {
            match self {
                Self::Small(arr, size) => {
                    arr[..*size]
                        .iter()
                        .for_each(|x| visitor(unsafe { x.assume_init_ref() }));
                }
                Self::Large(set) => set.iter().for_each(visitor),
            }
        }
    }
}

pub mod tree_decomp {
    use std::collections::VecDeque;

    // pub type HashSet<T> = std::collections::HashSet<T>;
    pub type HashSet<T> = crate::map::AdaptiveHashSet<T, 5>;

    pub const UNSET: u32 = u32::MAX;

    // Tree decomposition of treewidth 2.
    #[derive(Clone)]
    pub struct TW2 {
        // Perfect elimination ordering in the chordal completion
        pub topological_order: Vec<u32>,
        pub t_in: Vec<u32>,
        pub parents: Vec<[u32; 2]>,
    }

    impl TW2 {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = [u32; 2]>,
        ) -> Option<Self> {
            let mut neighbors = vec![HashSet::default(); n_verts];
            for [u, v] in edges {
                neighbors[u as usize].insert(v);
                neighbors[v as usize].insert(u);
            }

            let mut visited = vec![false; n_verts];
            let mut parents = vec![[UNSET; 2]; n_verts];

            let mut topological_order = vec![];
            let mut t_in = vec![UNSET; n_verts];
            let mut root = None;

            let mut queue: [_; 3] = std::array::from_fn(|_| VecDeque::new());
            for u in 0..n_verts {
                let d = neighbors[u].len();
                if d <= 2 {
                    visited[u] = true;
                    queue[d].push_back(u as u32);
                }
            }

            while let Some(u) = (0..=2).flat_map(|i| queue[i].pop_front()).next() {
                t_in[u as usize] = topological_order.len() as u32;
                topological_order.push(u);

                match neighbors[u as usize].len() {
                    0 => {
                        if let Some(old_root) = root {
                            parents[old_root as usize][0] = u;
                        }
                        root = Some(u);
                    }
                    1 => {
                        let mut p = UNSET;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| p = v);
                        neighbors[p as usize].remove(&u);

                        parents[u as usize][0] = p;

                        if !visited[p as usize] && neighbors[p as usize].len() <= 2 {
                            visited[p as usize] = true;
                            queue[neighbors[p as usize].len()].push_back(p);
                        }
                    }
                    2 => {
                        let mut ps = [UNSET; 2];
                        let mut i = 0;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| {
                            ps[i] = v;
                            i += 1;
                        });
                        let [p, q] = ps;

                        neighbors[p as usize].remove(&u);
                        neighbors[q as usize].remove(&u);

                        neighbors[p as usize].insert(q);
                        neighbors[q as usize].insert(p);

                        parents[u as usize] = [p, q];

                        for w in [p, q] {
                            if !visited[w as usize] && neighbors[w as usize].len() <= 2 {
                                visited[w as usize] = true;
                                queue[neighbors[w as usize].len()].push_back(w);
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }

            if topological_order.len() != n_verts {
                return None;
            }
            assert_eq!(root, topological_order.iter().last().copied());

            for u in 0..n_verts {
                let ps = &mut parents[u];
                if ps[1] != UNSET && t_in[ps[0] as usize] > t_in[ps[1] as usize] {
                    ps.swap(0, 1);
                }
            }

            Some(Self {
                parents,
                topological_order,
                t_in,
            })
        }
    }
}

use tree_decomp::UNSET;
const TW: usize = 2;
const K: usize = TW + 1;
type Bag = [u32; K];

const INF: u64 = 1 << 60;

fn bag_len(bag: &Bag) -> usize {
    (0..K).find(|&i| bag[i] == UNSET).unwrap_or(K)
}

enum MergeType<T> {
    Left(T),
    Right(T),
    Equal(T, T),
}

fn iter_merge_bag(lhs: &Bag, rhs: &Bag, mut f: impl FnMut(MergeType<usize>, usize)) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < K && j < K {
        match lhs[i].cmp(&rhs[j]) {
            Ordering::Less => {
                f(MergeType::Left(i), k);
                i += 1;
            }
            Ordering::Greater => {
                f(MergeType::Right(j), k);
                j += 1;
            }
            Ordering::Equal => {
                f(MergeType::Equal(i, j), k);
                i += 1;
                j += 1;
                k += 1;
            }
        }
    }
    k
}

fn iter_inter_bag(lhs: &Bag, rhs: &Bag, mut f: impl FnMut(usize, usize, usize)) -> usize {
    iter_merge_bag(lhs, rhs, |ty, k| {
        if let MergeType::Equal(i, j) = ty {
            f(i, j, k);
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
struct ConnState([u8; 3]);

impl ConnState {
    const fn normalize(mut self: ConnState) -> ConnState {
        let mut color_map = [!0; 6];
        color_map[0] = 0;

        let mut n_colors = 1;
        let mut i = 0;
        while i < 3 {
            let c = &mut self.0[i];
            if color_map[*c as usize] == !0 {
                color_map[*c as usize] = n_colors;
                n_colors += 1;
            }
            *c = color_map[*c as usize];

            i += 1;
        }
        self
    }

    fn merge(mut self: ConnState, u: u8, v: u8) -> Option<ConnState> {
        let c_src = self.0[u as usize];
        let c_dest = self.0[v as usize];
        match (c_src, c_dest) {
            (0, 0) => return None,
            (0, _) => self.0[u as usize] = self.0[v as usize],
            (_, 0) => self.0[v as usize] = self.0[u as usize],
            _ if c_src != c_dest => {
                for i in 0..3 {
                    if self.0[i] == c_src {
                        self.0[i] = c_dest;
                    }
                }
            }
            _ => {}
        }
        Some(self.normalize())
    }
}

const N_STATES: usize = 15;

struct Cx {
    states: [ConnState; N_STATES],

    symm_entries: [[u8; 2]; 3],
    edge_masks: [u8; 4],
    trans: Vec<(u8, u8, u8)>,
}

impl Cx {
    fn new() -> Self {
        let mut states = vec![];
        for c0 in 0..4 {
            for c1 in 0..4 {
                for c2 in 0..4 {
                    states.push(ConnState([c0, c1, c2]).normalize());
                }
            }
        }
        states.sort_unstable();
        states.dedup();
        // println!("N_STATES = {}", states.len());
        // println!("{:?}", states);
        assert_eq!(states.len(), N_STATES);
        let states = std::array::from_fn(|i| states[i]);

        let inv_state: HashMap<_, _> = states
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i as u8))
            .collect();

        let symm_entries = [[1, 2], [0, 2], [0, 1]];
        let edge_masks = [0b001, 0b010, 0b100, 0b111];

        let mut trans = vec![];
        let full = inv_state[&ConnState([1, 1, 1])];
        for src in 0..N_STATES as u8 {
            for dest in 0..N_STATES as u8 {
                let mut row = vec![];
                'mask_loop: for k in 0..3 {
                    let [u, v] = symm_entries[k as usize];
                    let Some(s) = states[src as usize].merge(u, v) else {
                        continue 'mask_loop;
                    };
                    if s != states[dest as usize] {
                        continue;
                    }

                    row.push(k);
                }
                if dest == full && row.is_empty() {
                    row.push(3);
                }
                for k in row {
                    trans.push((src, dest, k));
                }
            }
        }

        for &(src, dest, k) in &trans {
            println!(
                "{:?} {:?} {:0b}",
                states[src as usize], states[dest as usize], edge_masks[k as usize]
            );
        }

        Self {
            states,

            symm_entries,
            edge_masks,
            trans,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NodeAgg {
    bag: Bag,
    min_cost: [u64; N_STATES],
    inner_cost: [u64; 3],
}

impl NodeAgg {
    fn pull_up(&mut self, child: Self) {
        //
    }

    fn finalize(&mut self, cx: &Cx) {
        let ic = self.inner_cost;
        let inner_cost = [
            ic[0],
            ic[1],
            ic[2],
            (ic[1] + ic[2]).min(ic[0] + ic[2]).min(ic[0] + ic[1]),
        ];

        for &(src, dest, k) in &cx.trans {
            self.min_cost[dest as usize] = self.min_cost[dest as usize]
                .min(self.min_cost[src as usize] + inner_cost[k as usize]);
        }
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges = HashMap::new();
    let edges_unweighted = (0..m).map(|_| {
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        let w = input.u32();
        edges.insert([u, v], w);
        edges.insert([v, u], w);
        [u, v]
    });

    let td = tree_decomp::TW2::from_edges(n, edges_unweighted).unwrap();

    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];
    for u in 0..n {
        let p = td.parents[u][0] as usize;
        if p == UNSET as usize {
            continue;
        }
        degree[u] += 1;
        degree[p] += 1;
        xor_neighbors[u] ^= p as u32;
        xor_neighbors[p] ^= u as u32;
    }

    let mut included = vec![false; n];
    let mut root = UNSET as usize;
    for _ in 0..input.value() {
        let u = input.u32() - 1;
        included[u as usize] = true;
        root = u as usize;
    }
    degree[root] += 2;

    let cx = Cx::new();
    let mut dp: Vec<_> = (0..n)
        .map(|u| {
            let mut bag = [UNSET; K];
            bag[0] = u as u32;
            bag[1] = td.parents[u][0];
            bag[2] = td.parents[u][1];
            bag.sort_unstable();

            let mut min_cost = [INF; N_STATES];
            if degree[u] == 1 {
                min_cost[0] = 0;
            }
            let inner_cost = [[0, 1], [1, 2], [0, 2]]
                .map(|[i, j]| edges.get(&[bag[i], bag[j]]).copied().map_or(INF, u64::from));
            NodeAgg {
                bag,
                min_cost,
                inner_cost,
            }
        })
        .collect();

    for mut u in 0..n {
        while degree[u] == 1 {
            let p = xor_neighbors[u] as usize;
            degree[u] -= 1;
            degree[p] -= 1;
            xor_neighbors[p] ^= u as u32;
            xor_neighbors[u] ^= p as u32;

            let mut dp_u = dp[u];
            dp_u.finalize(&cx);
            dp[p].pull_up(dp_u);

            println!("{:?}", dp_u);

            u = p;
        }
    }
    dp[root].finalize(&cx);

    // let unique_mask = (0..

    println!("{:?}", dp[root]);
}
