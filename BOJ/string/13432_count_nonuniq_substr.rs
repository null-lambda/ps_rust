use std::{collections::HashSet, io::Write};

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

pub mod suffix_trie {
    // O(N) suffix array construction with suffix automaton
    // https://cp-algorithms.com/string/suffix-automaton.html#implementation

    type T = u8;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = !0 - 1; // Prevent overflow during an increment operation

    // HashMap-based transition table, for generic types
    // #[derive(Clone, Debug, Default)]
    // pub struct TransitionTable(std::collections::HashMap<T, NodeRef>);

    // impl TransitionTable {
    //     pub fn get(&self, c: T) -> NodeRef {
    //         self.0.get(&c).copied().unwrap_or(UNSET)
    //     }

    //     pub fn set(&mut self, c: T, u: NodeRef) {
    //         self.0.insert(c, u);
    //     }
    // }

    // SmallMap-based transition table, for small set of alphabets.

    pub const N_ALPHABET: usize = 26;

    // Since the number of transition is O(N), Most transitions tables are slim.
    pub const STACK_CAP: usize = 1;

    #[derive(Clone, Debug)]
    pub enum TransitionTable {
        Stack([(T, NodeRef); STACK_CAP]),
        HeapFixed(Box<[NodeRef; N_ALPHABET]>),
    }

    impl TransitionTable {
        pub fn get(&self, key: T) -> NodeRef {
            match self {
                Self::Stack(items) => items
                    .iter()
                    .find_map(|&(c, u)| (c == key).then(|| u))
                    .unwrap_or(UNSET),
                Self::HeapFixed(vec) => vec[key as usize],
            }
        }
        pub fn set(&mut self, key: T, u: NodeRef) {
            match self {
                Self::Stack(arr) => {
                    for (c, v) in arr.iter_mut() {
                        if c == &(N_ALPHABET as u8) {
                            *c = key;
                        }
                        if c == &key {
                            *v = u;
                            return;
                        }
                    }

                    let mut vec = Box::new([UNSET; N_ALPHABET]);
                    for (c, v) in arr.iter() {
                        vec[*c as usize] = *v;
                    }
                    vec[key as usize] = u;
                    *self = Self::HeapFixed(vec);
                }
                Self::HeapFixed(vec) => vec[key as usize] = u,
            }
        }
    }

    impl Default for TransitionTable {
        fn default() -> Self {
            Self::Stack([(N_ALPHABET as u8, UNSET); STACK_CAP])
        }
    }

    #[derive(Debug, Default)]
    pub struct Node {
        // DAWG of the string
        pub children: TransitionTable,

        // Suffix tree of the reversed string
        pub rev_depth: u32,
        pub rev_parent: NodeRef,

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

        pub fn push(&mut self, x: T) {
            let u = self.alloc(Node {
                children: Default::default(),
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
    }

    pub fn suffix_array<S>(s: &[S], mut f: impl FnMut(&S) -> T) -> Vec<u32> {
        let mut automaton = SuffixAutomaton::new();
        for b in s.iter().map(|b| f(b)).rev() {
            automaton.push(b);
        }

        let n_nodes = automaton.nodes.len();

        // Construct CSR of the suffix tree of rev(s)
        let mut head = vec![0u32; n_nodes + 1];
        for u in 1..n_nodes {
            head[1 + automaton.nodes[u].rev_parent as usize] += 1;
        }
        for u in 1..n_nodes {
            head[u + 1] += head[u];
        }
        let mut cursor = head[..n_nodes].to_vec();

        let mut rev_children = vec![(T::default(), 0u32); head[n_nodes] as usize];
        for (u, node) in automaton.nodes.iter().enumerate().skip(1) {
            let parent = &automaton.nodes[node.rev_parent as usize];
            let b = &s[s.len() - 1 - (node.first_endpos - parent.rev_depth) as usize];

            let p = node.rev_parent as usize;
            rev_children[cursor[p] as usize] = (f(b), u as u32);
            cursor[p] += 1;
        }

        // DFS
        let mut stack = vec![(0, 0)];
        let mut sa = Vec::with_capacity(s.len());
        while let Some((u, iv)) = stack.pop() {
            if iv == 0 {
                rev_children[head[u] as usize..head[u + 1] as usize].sort_unstable();
                let node = &automaton.nodes[u];
                if node.is_rev_terminal() {
                    sa.push(s.len() as u32 - node.rev_depth);
                }
            }
            if iv < head[u + 1] - head[u] {
                let (_c, v) = rev_children[head[u] as usize..][iv as usize];
                stack.push((u, iv + 1));
                stack.push((v as usize, 0));
            }
        }
        sa
    }
}

fn parse_char(b: u8) -> u8 {
    b - b'a'
}

fn combine_bound(a: (u32, u32), b: (u32, u32)) -> (u32, u32) {
    (a.0.min(b.0), a.1.max(b.1))
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut automaton = suffix_trie::SuffixAutomaton::new();
    for b in input.token().bytes().map(parse_char) {
        automaton.push(b);
    }

    let n_nodes = automaton.nodes.len();
    let mut degree = vec![1; n_nodes];
    for u in 1..n_nodes {
        degree[automaton.nodes[u].rev_parent as usize] += 1;
    }
    degree[0] += 2;

    const INF: u32 = 1 << 30;
    let mut ans = 0u64;
    let mut subtree_depth_bound = vec![(INF, 0); n_nodes];
    for mut u in 0..n_nodes as u32 {
        while degree[u as usize] == 1 {
            let node = &automaton.nodes[u as usize];
            let p = node.rev_parent;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            let l0 = automaton.nodes[p as usize].rev_depth + 1;
            let l1 = node.rev_depth;
            if node.is_rev_terminal() {
                subtree_depth_bound[u as usize] =
                    combine_bound((l1, l1), subtree_depth_bound[u as usize]);
            }
            let delta = subtree_depth_bound[u as usize].1 - subtree_depth_bound[u as usize].0;
            ans += l1.min(delta).saturating_sub(l0 - 1) as u64;

            subtree_depth_bound[p as usize] = combine_bound(
                subtree_depth_bound[p as usize],
                subtree_depth_bound[u as usize],
            );

            u = p;
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
