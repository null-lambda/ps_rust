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
    }
}

pub fn parse_char(b: u8) -> u8 {
    b - b'a'
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut automaton = suffix_trie::SuffixAutomaton::new();
    for _ in 0..n {
        for b in input.token().bytes().map(parse_char) {
            automaton.push(b);
        }
        automaton.push_sep();
    }

    'outer: for _ in 0..q {
        let mut u = 0;
        for b in input.token().bytes().map(parse_char) {
            u = automaton.nodes[u as usize].children.get(b);
            if u == suffix_trie::UNSET {
                writeln!(output, "0").unwrap();
                continue 'outer;
            }
        }

        let p = automaton.nodes[u as usize].rev_parent;
        let len = automaton.nodes[u as usize].rev_depth - automaton.nodes[p as usize].rev_depth;
        writeln!(output, "{}", len).unwrap();
    }
}
