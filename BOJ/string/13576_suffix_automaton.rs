use std::io::Write;

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
    pub const N_ALPHABET: usize = 94;

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

        pub fn push(&mut self, x: T) -> NodeRef {
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
            u
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

        // Preorder traversal
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

fn parse_byte(b: u8) -> u8 {
    b - 33
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let s = input.token().as_bytes();
    let mut automaton = suffix_trie::SuffixAutomaton::new();

    let mut count = vec![];
    for b in s.iter().copied().map(parse_byte) {
        let u = automaton.push(b);
        count.resize(automaton.nodes.len(), 0);
        count[u as usize] = 1u32;
    }
    let n_nodes = automaton.nodes.len();

    let mut degree = vec![1u32; n_nodes];
    for u in 1..n_nodes {
        let p = automaton.nodes[u].rev_parent as usize;
        degree[p] += 1;
    }
    degree[0] += 2;

    for mut u in 0..n_nodes {
        while degree[u] == 1 {
            let p = automaton.nodes[u as usize].rev_parent as usize;
            degree[u] = 0;
            degree[p] -= 1;

            count[p] += count[u as usize];

            u = p;
        }
    }

    const TERMINAL_FLAG: u32 = 1 << 30;
    let mut u = automaton.tail as usize;
    while u != suffix_trie::UNSET as usize {
        count[u] |= TERMINAL_FLAG;
        u = automaton.nodes[u].rev_parent as usize;
    }

    let mut u = 0;
    let mut ans = vec![];
    for (i, b) in s.iter().copied().map(parse_byte).enumerate() {
        u = automaton.nodes[u as usize].children.get(b);
        if u == suffix_trie::UNSET {
            break;
        }
        if count[u as usize] & TERMINAL_FLAG != 0 {
            ans.push((i as u32, count[u as usize] ^ TERMINAL_FLAG));
        }
    }

    writeln!(output, "{}", ans.len()).unwrap();
    for (i, count) in ans {
        writeln!(output, "{} {}", i + 1, count).unwrap();
    }
}
