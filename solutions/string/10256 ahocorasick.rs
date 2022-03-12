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

    const EMPTY: usize = usize::MAX;

    #[derive(Debug)]
    struct TrieNode {
        children: [usize; 4],
        is_terminal: bool,
        failure: Cell<usize>,
    }

    impl TrieNode {
        fn new() -> Self {
            Self {
                children: [EMPTY; 4],
                is_terminal: false,
                failure: Cell::new(0),
            }
        }
    }

    #[derive(Debug)]
    pub struct Matcher {
        trie: Vec<TrieNode>,
    }

    impl Matcher {
        pub fn new(patterns: impl Iterator<Item = impl Iterator<Item = u8>>) -> Self {
            // build trie
            let mut trie = vec![TrieNode::new()];
            for word in patterns {
                let mut i = 0;
                for c in word {
                    let n = trie.len();
                    if trie[i].children[c as usize] == EMPTY {
                        trie[i].children[c as usize] = n;
                        trie.push(TrieNode::new());
                    }
                    i = trie[i].children[c as usize];
                }
                trie[i].is_terminal = true;
            }

            // build jump table
            use std::collections::VecDeque;
            let mut bfs_queue: VecDeque<usize> = trie[0]
                .children
                .iter()
                .copied()
                .filter(|&i| i != EMPTY)
                .collect();

            while let Some(parent) = bfs_queue.pop_front() {
                for (c, &i) in trie[parent]
                    .children
                    .iter()
                    .enumerate()
                    .filter(|(_, &i)| i != EMPTY)
                {
                    let mut i_prev = trie[parent].failure.get();
                    i_prev = (|| loop {
                        if trie[i_prev].children[c as usize] != EMPTY {
                            return trie[i_prev].children[c as usize];
                        } else if i_prev == 0 {
                            return 0;
                        }
                        i_prev = trie[i_prev].failure.get();
                    })();
                    trie[i].failure.set(i_prev);
                    bfs_queue.push_back(i);
                }
            }

            Self { trie }
        }

        pub fn count(&self, s: impl Iterator<Item = u8>) -> u32 {
            let mut node = 0;
            s.map(|c| {
                node = (|| loop {
                    if self.trie[node].children[c as usize] != EMPTY {
                        return self.trie[node].children[c as usize];
                    } else if node == 0 {
                        return 0;
                    }
                    node = self.trie[node].failure.get();
                })();
                self.trie[node].is_terminal as u32
            })
            .sum()
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::iter::once;

    let test_cases = input.value();
    for _ in 0..test_cases {
        let n: usize = input.value();
        let m: usize = input.value();

        fn dna_to_idx(c: u8) -> u8 {
            match c {
                b'A' => 0,
                b'T' => 1,
                b'G' => 2,
                _ => 3,
            }
        }

        let text: Vec<u8> = input
            .token()
            .iter()
            .map(|&c| dna_to_idx(c))
            // .take(n)
            .collect();
        let marker: Vec<u8> = input
            .token()
            .iter()
            .map(|&c| dna_to_idx(c))
            // .take(m)
            .collect();
        let matcher = aho_corasick::Matcher::new(
            (0..m)
                .flat_map(|i| (i + 1..m).map(move |j| (i, j)))
                .chain(once((0, 0)))
                .map(|(i, j)| {
                    marker[0..i]
                        .iter()
                        .chain(marker[i..=j].iter().rev())
                        .chain(marker[j + 1..].iter())
                        .copied()
                }),
        );

        let result: u32 = matcher.count(text.into_iter());
        writeln!(output_buf, "{}", result).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
