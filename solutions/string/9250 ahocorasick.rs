mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

pub mod aho_corasick {
    use std::cell::Cell;
    use std::collections::hash_map::Entry;
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::hash::Hash;
    /*
    Pin in supported in Rust 1.33.0 (2018 edition: 1.31.0)
    use std::marker::{PhantomPinned, Unpin};
    use std::pin::{*, Pin};
    */

    #[derive(Debug)]
    struct TrieNode<T> {
        children: HashMap<T, usize>,
        // is_terminal: bool,
        is_sub_terminal: Cell<bool>,
        failure: Cell<usize>,
    }

    impl<T: Eq + Hash> TrieNode<T> {
        fn new() -> Self {
            Self {
                children: HashMap::new(),
                // is_terminal: false,
                is_sub_terminal: Cell::new(false),
                failure: Cell::new(0),
            }
        }
    }

    pub struct Matcher<T> {
        trie: Vec<TrieNode<T>>,
    }

    impl<T: Hash + Eq + Debug> Matcher<T> {
        pub fn new(patterns: impl Iterator<Item = impl Iterator<Item = T>>) -> Self {
            // build trie
            let mut trie = vec![TrieNode::new()];
            for word in patterns {
                let mut i = 0;
                for c in word {
                    let n = trie.len();
                    i = match trie[i].children.entry(c) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            entry.insert(n);
                            trie.push(TrieNode::new());
                            n
                        }
                    }
                }
                // trie[i].is_terminal = true;
                trie[i].is_sub_terminal.set(true);
            }

            // build jump table
            use std::collections::VecDeque;
            let mut bfs_queue = VecDeque::new();
            bfs_queue.extend(trie[0].children.values().copied());

            while let Some(parent) = bfs_queue.pop_front() {
                for (c, &i) in &trie[parent].children {
                    let mut i_prev = trie[parent].failure.get();
                    i_prev = (|| loop {
                        if let Some(&j) = trie[i_prev].children.get(c) {
                            return j;
                        } else if i_prev == 0 {
                            return 0;
                        }
                        i_prev = trie[i_prev].failure.get();
                    })();
                    trie[i].failure.set(i_prev);
                    bfs_queue.push_back(i);

                    if trie[i_prev].is_sub_terminal.get() {
                        trie[i].is_sub_terminal.set(true);
                    }
                }
            }

            Self { trie }
        }

        pub fn contains(&self, mut s: impl Iterator<Item = T>) -> bool {
            let mut node = 0;
            s.any(|c| {
                node = (|| loop {
                    if let Some(&j) = self.trie[node].children.get(&c) {
                        return j;
                    } else if node == 0 {
                        return 0;
                    }
                    node = self.trie[node].failure.get();
                })();
                self.trie[node].is_sub_terminal.get()
            })
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let matcher = aho_corasick::Matcher::new((0..n).map(|_| {
        let word = input.token();
        // word.to_vec().into_iter()
        word.to_vec().into_iter()
    }));

    let m = input.value();
    for _ in 0..m {
        let word = input.token().iter().copied();
        if matcher.contains(word) {
            writeln!(output_buf, "{}", "YES").unwrap();
        } else {
            writeln!(output_buf, "{}", "NO").unwrap();
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
