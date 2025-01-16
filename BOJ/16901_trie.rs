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

// type AlphabetMap = std::collections::HashMap<u8, trie::NodeRef>;
type AlphabetMap = [trie::NodeRef; 2];
// type AlphabetMap = trie::AdaptiveArrayMap<u8, 10, 2>;
// type AlphabetMap = trie::AdaptiveArrayMap<u8, 10, 1>;

fn query_similar(
    trie: &mut trie::Trie<impl trie::TransitionMap<Key = usize>>,
    path: impl IntoIterator<Item = usize>,
) -> u32 {
    let mut u = 0;
    let mut acc = 0;
    for c in path {
        let dual = 1 - c;
        let next = trie.pool[u as usize].children.get(&c);
        if next != trie::UNSET {
            u = next;
            acc = (acc << 1) | c as u32;
        } else {
            u = trie.pool[u as usize].children.get(&dual);
            acc = (acc << 1) | dual as u32;
        }
    }
    acc
}

type IntervalAgg = trie::Trie<AlphabetMap>;

fn bits(x: u32) -> impl Iterator<Item = usize> {
    (0..30).rev().map(move |i| ((x >> i) & 1) as usize)
}

fn build_mst_dnc(xs: &[u32], mst_len: &mut u64, bit_mask: u32) -> Option<IntervalAgg> {
    if xs.is_empty() {
        return None;
    }
    if bit_mask == 0 {
        debug_assert!(xs.len() <= 1);
        return None;
    }

    let i = xs.partition_point(|&x| x & bit_mask == 0);
    let (mut left, mut right) = xs.split_at(i);
    if left.len() < right.len() {
        std::mem::swap(&mut left, &mut right);
    }

    let left_trie = build_mst_dnc(left, mst_len, bit_mask >> 1);
    let right_trie = build_mst_dnc(right, mst_len, bit_mask >> 1);

    if !left.is_empty() && !right.is_empty() {
        let mut selected_trie;
        let counterparts;
        match (left_trie, right_trie) {
            (Some(left_trie), _) => {
                selected_trie = left_trie;
                counterparts = right;
            }
            (None, Some(right_trie)) => {
                selected_trie = right_trie;
                counterparts = left;
            }
            (None, None) => {
                selected_trie = trie::Trie::new();
                for &x in left {
                    selected_trie.insert(bits(x));
                }
                counterparts = right;
            }
        }

        let min_edge = counterparts
            .iter()
            .map(|&y| y ^ query_similar(&mut selected_trie, bits(y)))
            .min()
            .unwrap();
        *mst_len += min_edge as u64;

        for &y in counterparts {
            selected_trie.insert(bits(y));
        }
        return Some(selected_trie);
    }

    None
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut xs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    xs.sort_unstable();
    xs.dedup();

    let mut ans = 0;
    build_mst_dnc(&xs, &mut ans, 1 << 30);
    writeln!(output, "{}", ans).unwrap();
}
