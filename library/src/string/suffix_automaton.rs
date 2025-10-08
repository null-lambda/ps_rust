pub mod suffix_trie {
    /// O(N) representation of a suffix trie using a suffix automaton.
    /// References: https://cp-algorithms.com/string/suffix-automaton.html

    pub type T = u8;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = 1 << 30;

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

    // array-based transition table, for small set of alphabets.
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

    #[derive(Debug, Default, Clone)]
    pub struct Node {
        /// DAWG of the string
        pub children: TransitionTable,

        /// Suffix tree of the reversed string (equivalent to the compressed trie of the reversed prefixes)
        pub rev_depth: u32,
        pub rev_parent: NodeRef,

        /// An auxilary tag for the substring reconstruction
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
                            rev_depth: self.nodes[p as usize].rev_depth + 1,
                            ..self.nodes[c as usize].clone()
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

        // The range in the original text for the reversed edge u -> rev_parent[u].
        pub fn substr_range(&self, u: NodeRef) -> std::ops::Range<usize> {
            let u = &self.nodes[u as usize];
            let p = &self.nodes[u.rev_parent as usize];
            let s = u.first_endpos - u.rev_depth + 1;
            let e = u.first_endpos - p.rev_depth + 1;
            s as usize..e as usize
        }
    }

    pub fn suffix_array<S>(s: &[S], mut f: impl FnMut(&S) -> T) -> Vec<u32> {
        let mut automaton = SuffixAutomaton::new();
        for b in s.iter().map(|b| f(b)).rev() {
            automaton.push(b);
        }

        let n_nodes = automaton.nodes.len();

        // Construct CSR of the suffix tree
        let mut head = vec![0u32; n_nodes + 1];
        for u in 1..n_nodes {
            head[automaton.nodes[u].rev_parent as usize] += 1;
        }
        for u in 0..n_nodes {
            head[u + 1] += head[u];
        }

        let mut rev_children = vec![(T::default(), 0u32); head[n_nodes] as usize];
        for (u, node_u) in automaton.nodes.iter().enumerate().skip(1) {
            let p = node_u.rev_parent as usize;
            let b = &s[s.len() - automaton.substr_range(u as u32).end as usize];

            head[p] -= 1;
            rev_children[head[p] as usize] = (f(b), u as u32);
        }
        for u in 0..n_nodes {
            rev_children[head[u as usize] as usize..head[u as usize + 1] as usize].sort_unstable();
        }

        // Preorder traversal
        let mut sa = Vec::with_capacity(s.len());
        let mut current_edge = (0..n_nodes).map(|u| head[u]).collect::<Vec<_>>();
        let mut u = 0;
        loop {
            let p = automaton.nodes[u as usize].rev_parent;
            let e = current_edge[u as usize];
            current_edge[u as usize] += 1;

            if e == head[u as usize] {
                let node = &automaton.nodes[u as usize];
                if node.is_rev_terminal() {
                    sa.push(s.len() as u32 - node.rev_depth);
                }
            }
            if e == head[u as usize + 1] {
                if p == UNSET {
                    break;
                }
                u = p;
                continue;
            }

            let (_, v) = rev_children[e as usize];
            u = v;
        }

        sa
    }

    pub fn lcp_array<S>(s: &[S], mut f: impl FnMut(&S) -> T, suffix_array: &[u32]) -> Vec<u32> {
        let n = s.len();
        let mut rank = vec![0u32; n];
        let mut lcp_len = vec![0u32; n];
        for i in 0..n as u32 {
            rank[suffix_array[i as usize] as usize] = i;
        }

        let mut k = 0;
        for i in 0..n as u32 {
            if rank[i as usize] == 0 {
                continue;
            }
            let j = suffix_array[(rank[i as usize] - 1) as usize];
            while k < n as u32 - i.max(j) && f(&s[(i + k) as usize]) == f(&s[(j + k) as usize]) {
                k += 1;
            }
            lcp_len[rank[i as usize] as usize] = k;
            k = k.saturating_sub(1);
        }
        lcp_len
    }
}
