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
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read};
// use std::io::Write;

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

#[derive(Debug, Clone)]
struct Trie {
    terminal: bool,
    children: [Option<Box<Trie>>; 26],
}

impl Trie {
    fn new() -> Self {
        Self {
            terminal: false,
            children: Default::default(),
        }
    }

    fn ascii_index(c: u8) -> usize {
        (c - b'a') as usize
    }

    fn insert(&mut self, word: &[u8]) {
        let mut current = self;
        for c in word.iter().map(|&c| Trie::ascii_index(c)) {
            if current.children[c].is_none() {
                current.children[c] = Some(Box::new(Trie::new()));
            }
            current = current.children[c].as_mut().unwrap();
        }
        current.terminal = true;
    }

    fn find(&self, word: &[u8]) -> bool {
        let mut current = self;
        for c in word.iter().map(|&c| Trie::ascii_index(c)) {
            match current.children[c].as_ref() {
                Some(child) => { current = child; }
                None => { return false; }
            }
        }
        current.terminal
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input = &input_buf[..];

    let n: usize = input.value();
    let m: usize = input.value();
    input.skip_line();
    let mut trie = Trie::new();
    for _ in 0..n {
        trie.insert(input.line());
    }
    let result: u32 = (0..m).map(|_| {
        trie.find(input.line()) as u32        
    }).sum();
    println!("{}", result);
}
