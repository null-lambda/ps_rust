use std::{io::Write, num::NonZeroU32};

mod simple_io {
    use std::string::*;

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

#[derive(Debug, Clone)]
struct NodeRef(NonZeroU32);

#[derive(Debug, Clone)]
struct Node {
    children: [Option<NodeRef>; 2],
}

#[derive(Debug)]
struct BinaryTrie {
    pool: Vec<Node>,
}

impl BinaryTrie {
    fn new() -> Self {
        let root = Node {
            children: [None, None],
        };
        Self { pool: vec![root] }
    }

    fn len(&self) -> usize {
        self.pool.len() - 1
    }

    fn alloc(&mut self) -> NodeRef {
        let idx = self.pool.len();
        self.pool.push(Node {
            children: [None, None],
        });
        NodeRef(unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() })
    }

    fn insert(&mut self, x: u32) {
        let mut u = 0;
        for i in (0..32).rev() {
            let branch = ((x >> i) & 1) as usize;
            if self.pool[u].children[branch].is_none() {
                self.pool[u].children[branch] = Some(self.alloc());
            }
            u = self.pool[u].children[branch].as_ref().unwrap().0.get() as usize;
        }
    }

    fn query_max_xor(&self, x: u32) -> u32 {
        assert!(self.len() > 0);
        let mut u = 0;
        let mut acc = 0;
        for i in (0..32).rev() {
            let bit = ((x >> i) & 1) as usize;
            let branch = if self.pool[u].children[1 - bit].is_some() {
                acc |= 1 << i;
                1 - bit
            } else {
                bit
            };
            u = self.pool[u].children[branch].as_ref().unwrap().0.get() as usize;
        }
        {}
        acc
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let mut trie = BinaryTrie::new();
        trie.insert(0);
        let mut prefix = 0;
        let mut ans = u32::MIN;
        for _ in 0..n {
            let x: u32 = input.value();
            let new_prefix = prefix ^ x;
            ans = ans.max(trie.query_max_xor(new_prefix));

            prefix = new_prefix;
            trie.insert(prefix);
        }
        writeln!(output, "{}", ans).unwrap();
    }
}
