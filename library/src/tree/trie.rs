mod trie {
    // trie for 'A-Z'
    pub const N_ALPHABET: usize = 26;
    pub const UNSET: usize = std::usize::MAX;

    #[derive(Debug, Clone)]
    pub struct Node {
        pub children: [usize; N_ALPHABET],
        pub word_idx: usize,
    }

    impl Node {
        fn new() -> Self {
            Self {
                children: [UNSET; N_ALPHABET],
                word_idx: UNSET,
            }
        }

        pub fn terminal(&self) -> bool {
            self.word_idx != UNSET
        }
    }

    #[derive(Debug, Clone)]
    pub struct Trie {
        pub nodes: Vec<Node>,
    }

    impl Trie {
        pub fn new() -> Self {
            Self {
                nodes: vec![Node::new()],
            }
        }

        pub fn insert(&mut self, s: &[u8], idx: usize) {
            let mut current = 0;
            for &c in s {
                let c = (c - b'A') as usize;
                if self.nodes[current].children[c] == UNSET {
                    self.nodes.push(Node::new());
                    self.nodes[current].children[c] = self.nodes.len() - 1;
                }
                current = self.nodes[current].children[c];
            }
            self.nodes[current].word_idx = idx;
        }
    }
}
