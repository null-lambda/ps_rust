use std::{io::Write, ops::Range};

use trie::TransitionMap;

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

pub mod trie {
    use std::collections::HashMap;
    use std::hash::Hash;

    pub const UNSET: u32 = !0;
    pub type NodeRef = u32;

    // An interface for different associative containers
    pub trait TransitionMap {
        type Key;
        fn empty() -> Self;
        fn get(&self, key: &Self::Key) -> NodeRef;
        fn insert(&mut self, key: Self::Key, value: NodeRef);
        fn for_each(&self, f: impl FnMut(&Self::Key, &NodeRef));
    }

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

        fn for_each(&self, mut f: impl FnMut(&Self::Key, &NodeRef)) {
            for (k, v) in self {
                f(k, v);
            }
        }
    }

    // Fixed-size array map
    impl<const N_ALPHABETS: usize> TransitionMap for [NodeRef; N_ALPHABETS] {
        type Key = usize;

        fn empty() -> Self {
            std::array::from_fn(|_| UNSET)
        }

        fn get(&self, key: &Self::Key) -> NodeRef {
            self[*key]
        }

        fn insert(&mut self, key: usize, value: NodeRef) {
            self[key] = value;
        }

        fn for_each(&self, mut f: impl FnMut(&Self::Key, &NodeRef)) {
            for (i, v) in self.iter().enumerate() {
                if *v != UNSET {
                    f(&i, v);
                }
            }
        }
    }

    // Adaptive array map, based on the fact that most nodes are slim.
    pub enum AdaptiveArrayMap<K, const N_ALPHABETS: usize, const STACK_CAP: usize> {
        Small([(K, NodeRef); STACK_CAP]),
        Large(Box<[NodeRef; N_ALPHABETS]>),
    }

    impl<
            K: Into<usize> + TryFrom<usize> + Copy + Default + Eq,
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
                        } else if *v == UNSET {
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

        fn for_each(&self, mut f: impl FnMut(&Self::Key, &NodeRef)) {
            match self {
                Self::Small(assoc_list) => {
                    for (k, v) in assoc_list {
                        if *v != UNSET {
                            f(k, v);
                        }
                    }
                }
                Self::Large(array_map) => {
                    for (i, v) in array_map.iter().enumerate() {
                        if *v != UNSET {
                            f(&(unsafe { i.try_into().unwrap_unchecked() }), v);
                        }
                    }
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Node<M> {
        pub children: M,
        pub tag: u32,
    }

    #[derive(Debug)]
    pub struct Trie<M> {
        pub pool: Vec<Node<M>>,
    }

    impl<M: TransitionMap> Trie<M> {
        pub fn new() -> Self {
            let root = Node {
                children: M::empty(),
                tag: UNSET,
            };
            Self { pool: vec![root] }
        }

        fn alloc(&mut self) -> NodeRef {
            let idx = self.pool.len() as u32;
            self.pool.push(Node {
                children: M::empty(),
                tag: UNSET,
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

pub mod segtree_wide {
    // Cache-friendly segment tree, based on a B-ary tree.
    // https://en.algorithmica.org/hpc/data-structures/segment-trees/#wide-segment-trees

    // const CACHE_LINE_SIZE: usize = 64;

    // const fn adaptive_block_size<T>() -> usize {
    //     assert!(
    //         std::mem::size_of::<T>() > 0,
    //         "Zero-sized types are not supported"
    //     );
    //     let mut res = CACHE_LINE_SIZE / std::mem::size_of::<T>();
    //     if res < 2 {
    //         res = 2;
    //     }
    //     res
    // }

    use std::iter;

    const fn height<const B: usize>(mut node: usize) -> u32 {
        debug_assert!(node > 0);
        let mut res = 1;
        while node > B {
            res += 1;
            node = node.div_ceil(B);
        }
        res
    }

    // yields (h, offset)
    fn offsets<const B: usize>(size: usize) -> impl Iterator<Item = usize> {
        let mut offset = 0;
        let mut n = size;
        iter::once(0).chain((1..).map(move |_| {
            n = n.div_ceil(B);
            offset += n * B;
            offset
        }))
    }

    fn offset<const B: usize>(size: usize, h: u32) -> usize {
        offsets::<B>(size).nth(h as usize).unwrap()
    }

    fn log<const B: usize>() -> u32 {
        usize::BITS - B.leading_zeros() - 1
    }

    fn round<const B: usize>(x: usize) -> usize {
        x & !(B - 1)
    }

    const fn compute_mask<const B: usize>() -> [[X; B]; B] {
        let mut res = [[0; B]; B];
        let mut i = 0;
        while i < B {
            let mut j = 0;
            while j < B {
                res[i][j] = if i < j { !0 } else { 0 };
                j += 1;
            }
            i += 1;
        }
        res
    }

    type X = i32;

    #[derive(Debug, Clone)]
    pub struct SegTree<const B: usize> {
        n: usize,
        sum: Vec<X>,
        mask: [[X; B]; B],
        offsets: Vec<usize>,
    }

    impl<const B: usize> SegTree<B> {
        pub fn with_size(n: usize) -> Self {
            assert!(B >= 2 && B.is_power_of_two());
            let max_height = height::<B>(n);
            Self {
                n,
                sum: vec![0; offset::<B>(n, max_height)],
                mask: compute_mask::<B>(),
                offsets: offsets::<B>(n).take(max_height as usize).collect(),
            }
        }

        #[target_feature(enable = "avx2")] // Required. __mm256 has significant performance benefits over __m128.
        unsafe fn add_avx2(&mut self, mut idx: usize, value: X) {
            debug_assert!(idx < self.n);
            for (_, offset) in self.offsets.iter().enumerate() {
                let block = &mut self.sum[offset + round::<B>(idx)..];
                for (b, m) in block.iter_mut().zip(&self.mask[idx % B]) {
                    *b += value & m;
                }
                idx >>= log::<B>();
            }
        }

        pub fn add(&mut self, idx: usize, value: X) {
            unsafe {
                self.add_avx2(idx, value);
            }
        }

        pub fn sum_prefix(&mut self, idx: usize) -> X {
            debug_assert!(idx <= self.n);
            let mut res = 0;
            for (h, offset) in self.offsets.iter().enumerate() {
                res += self.sum[offset + (idx >> h as u32 * log::<B>())];
            }
            res
        }

        pub fn sum_range(&mut self, range: std::ops::Range<usize>) -> X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let r = self.sum_prefix(range.end);
            let l = self.sum_prefix(range.start);
            r - l
        }
    }
}

// type AlphabetMap = std::collections::HashMap<u8, trie::NodeRef>;
// type AlphabetMap = [trie::NodeRef; 4];
// type AlphabetMap = trie::AdaptiveArrayMap<u8, 10, 2>;
type AlphabetMap = trie::AdaptiveArrayMap<u8, 4, 1>;

fn parse_byte(b: u8) -> u8 {
    match b {
        b'A' => 0,
        b'U' => 1,
        b'G' => 2,
        b'C' => 3,
        _ => panic!(),
    }
}

#[derive(Debug, Clone)]
enum EventTag {
    RangeQuery {
        v_range: Range<u32>,
        inv: bool,
        idx_query: u32,
    },
    AddPoint {
        v: u32,
    },
}

impl EventTag {
    fn discriminant(&self) -> u32 {
        match self {
            EventTag::RangeQuery { .. } => 0,
            EventTag::AddPoint { .. } => 1,
        }
    }
}

use EventTag::*;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut prefixes = trie::Trie::<AlphabetMap>::new();
    let mut rev_prefixes = trie::Trie::<AlphabetMap>::new();

    let mut points = vec![];
    for _ in 0..n {
        let s = input.token();
        let u = prefixes.insert(s.bytes().map(parse_byte));
        let v = rev_prefixes.insert(s.bytes().rev().map(parse_byte));
        points.push((u, v));
    }

    let gen_euler_tour = |trie: &mut trie::Trie<AlphabetMap>| {
        let n_nodes = trie.pool.len();
        let mut size = vec![1u32; n_nodes];
        for u in (0..n_nodes).rev() {
            trie.pool[u].children.for_each(|_, &v| {
                size[u] += size[v as usize];
            });
        }

        let mut euler_in = size.clone();
        let mut euler_out = size;
        for u in 0..n_nodes {
            trie.pool[u].children.for_each(|_, &v| {
                let last_idx = euler_in[u];
                euler_in[u] -= euler_in[v as usize];
                euler_in[v as usize] = last_idx;
            });
            euler_in[u] -= 1;
            euler_out[u] += euler_in[u];
        }

        (n_nodes, euler_in, euler_out)
    };
    let (_n_nodes, euler_in, euler_out) = gen_euler_tour(&mut prefixes);
    let (n_nodes_rev, rev_euler_in, rev_euler_out) = gen_euler_tour(&mut rev_prefixes);

    let mut events = vec![];
    for &(u, v) in &points {
        events.push((
            euler_in[u as usize],
            AddPoint {
                v: rev_euler_in[v as usize],
            },
        ));
    }

    for i in 0..m {
        let p = input.token();
        let q = input.token();

        let (Some(x), Some(y)) = (
            prefixes.find(p.bytes().map(parse_byte)),
            rev_prefixes.find(q.bytes().rev().map(parse_byte)),
        ) else {
            continue;
        };

        let (us, ue) = (euler_in[x as usize], euler_out[x as usize]);
        let (vs, ve) = (rev_euler_in[y as usize], rev_euler_out[y as usize]);

        events.push((
            us,
            RangeQuery {
                v_range: vs..ve,
                idx_query: i as u32,
                inv: true,
            },
        ));
        events.push((
            ue,
            RangeQuery {
                v_range: vs..ve,
                idx_query: i as u32,
                inv: false,
            },
        ));
    }

    events.sort_unstable_by_key(|(u, ty)| (*u, ty.discriminant()));

    let mut ans = vec![0; m];
    let mut counter = segtree_wide::SegTree::<32>::with_size(n_nodes_rev);
    for (_u, ty) in events {
        match ty {
            AddPoint { v } => {
                counter.add(v as usize, 1);
            }
            RangeQuery {
                v_range,
                idx_query,
                inv,
            } => {
                let delta = counter.sum_range(v_range.start as usize..v_range.end as usize);
                ans[idx_query as usize] += if inv { -delta } else { delta };
            }
        }
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
