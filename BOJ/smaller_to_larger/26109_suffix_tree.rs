use std::{collections::BTreeSet, io::Write, ops::RangeInclusive};

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
    // O(N) Suffix trie with suffix automaton
    // https://cp-algorithms.com/string/suffix-automaton.html#implementation

    type T = u8;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = !0;

    // Hashmap-based transition table, for generic types
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

    // SmallVec-based transition table, for small set of alphabets.
    pub const N_ALPHABET: usize = 26;
    pub const STACK_CAP: usize = 1;

    #[derive(Clone, Debug)]
    pub enum TransitionTable {
        Small([(T, NodeRef); STACK_CAP]),
        Large(Box<[NodeRef; N_ALPHABET]>),
    }

    impl TransitionTable {
        pub fn get(&self, key: T) -> NodeRef {
            match self {
                Self::Small(items) => items
                    .iter()
                    .find_map(|&(c, u)| (c == key).then(|| u))
                    .unwrap_or(UNSET),
                Self::Large(vec) => vec[key as usize],
            }
        }
        pub fn set(&mut self, key: T, u: NodeRef) {
            match self {
                Self::Small(arr) => {
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
                    *self = Self::Large(vec);
                }
                Self::Large(vec) => vec[key as usize] = u,
            }
        }
    }

    impl Default for TransitionTable {
        fn default() -> Self {
            Self::Small([(N_ALPHABET as u8, UNSET); STACK_CAP])
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
        pub is_rev_terminal: bool,
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
                is_rev_terminal: false,
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
                is_rev_terminal: true,
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
                            is_rev_terminal: false,
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
}

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn count_distinct_occurences(step: u32, sub_depths: &BTreeSet<u32>) -> u32 {
    let mut current_depth = 0;
    let mut count = 0;
    while let Some(&d) = sub_depths.range(current_depth + step..).next() {
        current_depth = d;
        count += 1;
    }
    count
}

#[derive(Debug, Default)]
struct NodeData {
    sub_depths: BTreeSet<u32>,
    k: u32,
}

impl NodeData {
    fn singleton(depth: u32, is_terminal: bool) -> Self {
        if is_terminal {
            Self {
                sub_depths: [depth].into(),
                k: 1,
            }
        } else {
            Self {
                sub_depths: Default::default(),
                k: 0,
            }
        }
    }

    fn pull_from(&mut self, mut other: Self) {
        if self.sub_depths.len() < other.sub_depths.len() {
            std::mem::swap(&mut self.sub_depths, &mut other.sub_depths);
        }
        self.sub_depths.extend(other.sub_depths);
        self.k += other.k;
    }

    fn finalize(&mut self, depth_bound: RangeInclusive<u32>, ans: &mut [(u32, u32)]) {
        let (d0, d1) = depth_bound.into_inner();
        let m = count_distinct_occurences(d0, &self.sub_depths);
        let d_max = partition_point(d0 + 1, d1 + 1, |d| {
            count_distinct_occurences(d, &self.sub_depths) == m
        }) - 1;
        ans[self.k as usize] = ans[self.k as usize].max((m, d_max));
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let s = input.token();
    let mut automaton = suffix_trie::SuffixAutomaton::new();
    for b in s.bytes().map(|b| b - b'a') {
        automaton.push(b);
    }

    let n_nodes = automaton.nodes.len();
    let mut degree = vec![1; n_nodes];
    degree[0] += 2;
    for node in &automaton.nodes[1..] {
        degree[node.rev_parent as usize] += 1;
    }

    let mut ans = vec![(0, 0); s.len() + 1];
    let mut dp: Vec<_> = automaton
        .nodes
        .iter()
        .map(|node| NodeData::singleton(node.rev_depth, node.is_rev_terminal))
        .collect();
    for mut u in 0..n_nodes as u32 {
        while degree[u as usize] == 1 {
            let node = &automaton.nodes[u as usize];
            let p = node.rev_parent;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            let parent_node = &automaton.nodes[p as usize];

            let mut dp_u = std::mem::take(&mut dp[u as usize]);
            dp_u.finalize(parent_node.rev_depth + 1..=node.rev_depth, &mut ans);
            dp[p as usize].pull_from(dp_u);

            u = p;
        }
    }
    for (_, f) in &ans[1..] {
        write!(output, "{} ", f).unwrap();
    }
    writeln!(output).unwrap();
}
