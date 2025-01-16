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
//
fn query_exclusive(
    trie: &mut trie::Trie<impl trie::TransitionMap<Key = usize>>,
    path: impl IntoIterator<Item = usize>,
) -> u32 {
    let mut u = 0;
    let mut acc = 0;
    for c in path {
        let dual = 1 - c;
        let mut next = trie.pool[u].children.get(&dual);
        let mut b = dual as u32;
        if next == trie::UNSET {
            next = trie.pool[u].children.get(&c);
            b = 1 - b;
        }
        u = next as usize;
        acc = (acc << 1) | b;
    }
    acc
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut ts = trie::Trie::<AlphabetMap>::new();
    let mut xs = (0..input.value()).map(|_| {
        let x = input.value::<u32>();
        let bits = move || (0..30).rev().map(move |i| ((x >> i) & 1) as usize);
        (x, bits)
    });
    let mut ans = 0;
    ts.insert(xs.next().unwrap().1());
    for (x, bits) in xs {
        let q = query_exclusive(&mut ts, bits());
        ans = ans.max(x ^ q);
        ts.insert(bits());
    }
    writeln!(output, "{}", ans).unwrap();
}
