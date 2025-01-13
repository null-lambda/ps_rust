use std::{collections::HashSet, io::Write};

use segtree::{persistent::*, Monoid};

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

pub mod segtree {
    pub trait Monoid {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    pub trait Group: Monoid {
        fn sub(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    pub mod persistent {
        use super::*;

        use std::ops::Range;

        const UNSET: u32 = std::u32::MAX;
        pub type NodeRef = u32;
        type Link = NodeRef;

        struct Node<M: Monoid> {
            value: M::X,
            children: [Link; 2],
        }

        pub struct NodePool<M: Monoid> {
            n: usize,
            nodes: Vec<Node<M>>,
            pub monoid: M,
        }

        impl<M: Monoid> NodePool<M> {
            fn add_node(&mut self, value: M::X) -> NodeRef {
                let node = Node {
                    value,
                    children: [UNSET; 2],
                };
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                idx
            }

            pub fn with_size(n: usize, monoid: M) -> (Self, NodeRef) {
                let mut this = Self {
                    n,
                    nodes: vec![],
                    monoid,
                };
                this.with_size_rec(0..n);
                (this, 0)
            }

            fn with_size_rec(&mut self, range: Range<usize>) -> Link {
                debug_assert!(range.start <= range.end);
                let Range { start, end } = range;
                if end - start == 0 {
                    return UNSET;
                }

                let mid = (start + end) / 2;
                let u = self.add_node(self.monoid.id());
                if end - start > 1 {
                    self.nodes[u as usize].children =
                        [self.with_size_rec(start..mid), self.with_size_rec(mid..end)];
                }

                u
            }

            #[must_use]
            pub fn set(&mut self, root: NodeRef, idx: usize, value: M::X) -> NodeRef {
                let mut path = vec![];
                let mut node = root;
                let (mut start, mut end) = (0, self.n);
                loop {
                    if end - start == 1 {
                        break;
                    }

                    let mid = (start + end) / 2;
                    if idx < mid {
                        path.push((node, 0u8));
                        end = mid;
                        node = self.nodes[node as usize].children[0];
                    } else {
                        path.push((node, 1u8));
                        start = mid;
                        node = self.nodes[node as usize].children[1];
                    }
                }

                let mut root = self.add_node(value);
                for (node, branch) in path.into_iter().rev() {
                    let (left, right) = match branch {
                        0 => (root, self.nodes[node as usize].children[1]),
                        1 => (self.nodes[node as usize].children[0], root),
                        _ => unreachable!(),
                    };
                    root = self.add_node(self.monoid.combine(
                        &self.nodes[left as usize].value,
                        &self.nodes[right as usize].value,
                    ));
                    self.nodes[root as usize].children = [left, right];
                }
                root
            }

            pub fn query_range(&self, node: NodeRef, range: Range<usize>) -> M::X {
                self.query_range_rec(node, 0..self.n, range)
            }

            pub fn query_range_rec(
                &self,
                node: NodeRef,
                node_range: Range<usize>,
                query_range: Range<usize>,
            ) -> M::X {
                let Range { start, end } = node_range;
                let Range {
                    start: query_start,
                    end: query_end,
                } = query_range;
                if query_end <= start || end <= query_start {
                    return self.monoid.id();
                }
                if query_start <= start && end <= query_end {
                    return self.nodes[node as usize].value.clone();
                }
                let mid = (start + end) / 2;
                let c = self.nodes[node as usize].children;
                self.monoid.combine(
                    &self.query_range_rec(c[0], start..mid, query_range.clone()),
                    &self.query_range_rec(c[1], mid..end, query_range),
                )
            }
        }
    }
}

pub mod suffix_trie {
    // O(N) suffix array construction with suffix automaton
    // https://cp-algorithms.com/string/suffix-automaton.html#implementation

    type T = u8;
    type Tag = i32;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = !0 - 1; // Prevent overflow during an increment operation

    // // HashMap-based transition table, for generic types
    // // TODO: do an adaptive switch between hasmap and array
    // #[derive(Clone, Debug, Default)]
    // pub struct TransitionTable(std::collections::HashMap<T, NodeRef>);

    // impl TransitionTable {
    //     pub fn get(&self, c: T) -> NodeRef {
    //         self.0.get(&c).copied().unwrap_or(UNSET)
    //     }

    //     pub fn set(&mut self, c: T, u: NodeRef) {
    //         self.0.insert(c, u);
    //     }

    //     pub fn for_each(&self, mut f: impl FnMut(T, NodeRef)) {
    //         self.0.iter().for_each(|(&c, &u)| f(c, u));
    //     }
    // }

    // Array-based transition table, for small set of alphabets.

    // pub const N_ALPHABET: usize = 26;
    pub const N_ALPHABET: usize = 2;

    #[derive(Clone, Debug)]
    pub struct TransitionTable([NodeRef; N_ALPHABET]);

    impl Default for TransitionTable {
        fn default() -> Self {
            Self([UNSET; N_ALPHABET])
        }
    }

    impl TransitionTable {
        pub fn get(&self, key: T) -> NodeRef {
            self.0[key as usize]
        }
        pub fn set(&mut self, key: T, u: NodeRef) {
            self.0[key as usize] = u;
        }

        pub fn for_each(&self, mut f: impl FnMut(T, NodeRef)) {
            for (c, &u) in self.0.iter().enumerate() {
                f(c as T, u);
            }
        }
    }

    // // Since the number of transition is O(N), Most transitions tables are slim.
    // pub const STACK_CAP: usize = 1;

    // #[derive(Clone, Debug)]
    // pub enum TransitionTable {
    //     Stack([(T, NodeRef); STACK_CAP]),
    //     HeapFixed(Box<[NodeRef; N_ALPHABET]>),
    // }

    // impl TransitionTable {
    //     pub fn get(&self, key: T) -> NodeRef {
    //         match self {
    //             Self::Stack(items) => items
    //                 .iter()
    //                 .find_map(|&(c, u)| (c == key).then(|| u))
    //                 .unwrap_or(UNSET),
    //             Self::HeapFixed(vec) => vec[key as usize],
    //         }
    //     }

    //     pub fn set(&mut self, key: T, u: NodeRef) {
    //         match self {
    //             Self::Stack(arr) => {
    //                 for (c, v) in arr.iter_mut() {
    //                     if c == &(N_ALPHABET as u8) {
    //                         *c = key;
    //                     }
    //                     if c == &key {
    //                         *v = u;
    //                         return;
    //                     }
    //                 }

    //                 let mut vec = Box::new([UNSET; N_ALPHABET]);
    //                 for (c, v) in arr.iter() {
    //                     vec[*c as usize] = *v;
    //                 }
    //                 vec[key as usize] = u;
    //                 *self = Self::HeapFixed(vec);
    //             }
    //             Self::HeapFixed(vec) => vec[key as usize] = u,
    //         }
    //     }

    //     pub fn for_each(&self, mut f: impl FnMut(T, NodeRef)) {
    //         match self {
    //             Self::Stack(items) => {
    //                 for &(c, u) in items {
    //                     if c != N_ALPHABET as u8 {
    //                         f(c, u);
    //                     }
    //                 }
    //             }
    //             Self::HeapFixed(vec) => {
    //                 for (c, &u) in vec.iter().enumerate() {
    //                     if u != UNSET {
    //                         f(c as T, u);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // impl Default for TransitionTable {
    //     fn default() -> Self {
    //         Self::Stack([(N_ALPHABET as u8, UNSET); STACK_CAP])
    //     }
    // }

    #[derive(Debug, Default)]
    pub struct Node {
        // DAWG of the string
        pub children: TransitionTable,

        pub tag: Tag,

        // Suffix tree of the reversed string
        pub rev_parent: NodeRef,
        pub rev_depth: u32,

        pub first_endpos: u32,
    }

    impl Node {
        pub fn is_rev_terminal(&self) -> bool {
            self.first_endpos + 1 == self.rev_depth
        }
    }

    pub struct SuffixAutomaton {
        pub nodes: Vec<Node>,
        pub tail: NodeRef,
    }

    impl SuffixAutomaton {
        pub fn new() -> Self {
            let root = Node {
                children: Default::default(),

                tag: Tag::default(),

                rev_parent: UNSET,
                rev_depth: 0,

                first_endpos: UNSET,
            };

            Self {
                nodes: vec![root],
                tail: 0,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let u = self.nodes.len() as u32;
            self.nodes.push(node);
            u
        }

        pub fn push(&mut self, x: T, tag: Tag) {
            let u = self.alloc(Node {
                children: Default::default(),

                tag,

                rev_parent: 0,
                rev_depth: self.nodes[self.tail as usize].rev_depth + 1,

                first_endpos: self.nodes[self.tail as usize].rev_depth,
            });
            let mut p = self.tail;
            self.tail = u;

            while p != UNSET {
                let c = self.nodes[p as usize].children.get(x);
                if c != UNSET {
                    if self.nodes[p as usize].rev_depth + 1 == self.nodes[c as usize].rev_depth {
                        self.nodes[u as usize].rev_parent = c;
                    } else {
                        let c_cloned = self.alloc(Node {
                            children: self.nodes[c as usize].children.clone(),

                            tag: Tag::default(),

                            rev_parent: self.nodes[c as usize].rev_parent,
                            rev_depth: self.nodes[p as usize].rev_depth + 1,

                            first_endpos: self.nodes[c as usize].first_endpos,
                        });

                        self.nodes[u as usize].rev_parent = c_cloned;
                        self.nodes[c as usize].rev_parent = c_cloned;

                        while p != UNSET && self.nodes[p as usize].children.get(x) == c {
                            self.nodes[p as usize].children.set(x, c_cloned);
                            p = self.nodes[p as usize].rev_parent;
                        }
                    }

                    break;
                }

                self.nodes[p as usize].children.set(x, u);
                p = self.nodes[p as usize].rev_parent;
            }
        }

        pub fn push_sep(&mut self) {
            self.tail = 0;
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Additive;

impl Monoid for Additive {
    type X = u32;

    fn id(&self) -> Self::X {
        0
    }

    fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X {
        a + b
    }
}

#[derive(Clone, Default)]
struct NodeData {
    counter: NodeRef,
    items: HashSet<u32>,
}

impl NodeData {
    fn new(root: NodeRef) -> Self {
        Self {
            counter: root,
            items: Default::default(),
        }
    }

    fn insert(&mut self, x: u32, pool: &mut NodePool<Additive>) {
        self.items.insert(x);
        self.counter = pool.set(self.counter, x as usize, 1);
    }

    fn extend(&mut self, mut other: Self, pool: &mut NodePool<Additive>) {
        if self.items.len() < other.items.len() {
            std::mem::swap(self, &mut other);
        }
        for &x in &other.items {
            self.counter = pool.set(self.counter, x as usize, 1);
        }
        self.items.extend(other.items);
    }
}

fn parse_char(b: u8) -> u8 {
    b - b'a'
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut automaton = suffix_trie::SuffixAutomaton::new();

    for u in 1..=n {
        for b in input.token().bytes().map(parse_char) {
            automaton.push(b, u as i32);
        }
        automaton.push_sep();
    }
    let n_nodes = automaton.nodes.len();

    let (mut terminals, root) = NodePool::with_size(n, Additive);
    let mut singleton = vec![NodeData::new(root); n + 1];
    for u in 1..=n {
        singleton[u].insert(u as u32 - 1, &mut terminals);
    }
    let mut dp: Vec<_> = (0..n_nodes)
        .map(|u| singleton[automaton.nodes[u].tag as usize].clone())
        .collect();

    let mut queries = vec![vec![]; n_nodes];
    let mut ans = vec![0u32; q];
    'outer: for i in 0..q {
        let query_range = (input.value::<u32>() - 1, input.value::<u32>() - 1);
        let mut u = 0;
        for b in input.token().bytes().map(parse_char) {
            u = automaton.nodes[u as usize].children.get(b);
            if u == suffix_trie::UNSET {
                continue 'outer;
            }
        }
        queries[u as usize].push((i, query_range));
    }

    let mut indegree = vec![0; n_nodes];
    for u in 1..n_nodes {
        indegree[automaton.nodes[u].rev_parent as usize] += 1;
    }
    let mut topological_order: Vec<_> = (0..n_nodes as u32)
        .filter(|&u| indegree[u as usize] == 0)
        .collect();
    let mut timer = 0;
    while let Some(&u) = topological_order.get(timer) {
        timer += 1;

        for (i, (l, r)) in queries[u as usize].drain(..) {
            ans[i] = terminals.query_range(dp[u as usize].counter, l as usize..r as usize + 1);
        }

        let dp_u = std::mem::take(&mut dp[u as usize]);
        let p = automaton.nodes[u as usize].rev_parent;
        if p != suffix_trie::UNSET {
            dp[p as usize].extend(dp_u, &mut terminals);
        }

        let p = automaton.nodes[u as usize].rev_parent;
        if p != suffix_trie::UNSET {
            indegree[p as usize] -= 1;
            if indegree[p as usize] == 0 {
                topological_order.push(p);
            }
        }
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
