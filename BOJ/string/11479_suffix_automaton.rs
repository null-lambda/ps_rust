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
    // O(N) Suffix trie with suffix automaton
    // https://cp-algorithms.com/string/suffix-automaton.html#implementation

    type T = u8;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = !0;

    // // Hashmap-based transition table, for generic types
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

    // Array-based transition table, for small set of alphabets
    pub const N_ALPHABET: usize = 26;

    #[derive(Clone, Debug)]
    pub struct TransitionTable([NodeRef; N_ALPHABET]);

    impl TransitionTable {
        pub fn get(&self, key: T) -> NodeRef {
            debug_assert!((key as usize) < N_ALPHABET);
            self.0[key as usize]
        }

        pub fn set(&mut self, key: T, u: NodeRef) {
            debug_assert!((key as usize) < N_ALPHABET);
            self.0[key as usize] = u;
        }
    }

    impl Default for TransitionTable {
        fn default() -> Self {
            Self([UNSET; N_ALPHABET])
        }
    }

    #[derive(Debug)]
    pub struct Node {
        // DAWG of the string
        pub children: TransitionTable,

        // Suffix tree of the reversed string
        pub rev_depth: u32,
        pub rev_parent: NodeRef,
    }

    pub struct SuffixAutomaton {
        pub nodes: Vec<Node>,
        pub tail: NodeRef,

        pub n_unique_substr: i64,
    }

    impl SuffixAutomaton {
        pub fn new() -> Self {
            let root = Node {
                children: Default::default(),
                rev_parent: UNSET,
                rev_depth: 0,
            };
            Self {
                nodes: vec![root],
                tail: 0,

                n_unique_substr: 0,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let u = self.nodes.len() as u32;
            self.nodes.push(node);
            self.update_link(u, false);
            u
        }

        pub fn update_link(&mut self, u: NodeRef, inv: bool) {
            if u == UNSET || u == 0 {
                return;
            }
            let p = self.nodes[u as usize].rev_parent;
            let delta =
                self.nodes[u as usize].rev_depth as i64 - self.nodes[p as usize].rev_depth as i64;
            if !inv {
                self.n_unique_substr += delta;
            } else {
                self.n_unique_substr -= delta;
            }
        }

        pub fn push(&mut self, x: T) {
            let u = self.alloc(Node {
                children: Default::default(),
                rev_parent: 0,
                rev_depth: self.nodes[self.tail as usize].rev_depth + 1,
            });
            let mut p = self.tail;
            self.tail = u;

            while p != UNSET {
                let c = self.nodes[p as usize].children.get(x);
                if c != UNSET {
                    if self.nodes[p as usize].rev_depth + 1 == self.nodes[c as usize].rev_depth {
                        self.update_link(u, true);
                        self.nodes[u as usize].rev_parent = c;
                        self.update_link(u, false);
                    } else {
                        let c_cloned = self.alloc(Node {
                            children: self.nodes[c as usize].children.clone(),
                            rev_parent: self.nodes[c as usize].rev_parent,
                            rev_depth: self.nodes[p as usize].rev_depth + 1,
                        });

                        self.update_link(u, true);
                        self.update_link(c, true);
                        self.nodes[u as usize].rev_parent = c_cloned;
                        self.nodes[c as usize].rev_parent = c_cloned;
                        self.update_link(u, false);
                        self.update_link(c, false);

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

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let mut automaton = suffix_trie::SuffixAutomaton::new();
    for b in input.token().bytes() {
        automaton.push(b - b'a');
    }

    writeln!(output, "{}", automaton.n_unique_substr).unwrap();
}
