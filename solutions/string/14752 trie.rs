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

use std::io::Write;
use std::io::{BufReader, Read};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct Trie(BTreeMap<String, Trie>);

impl Trie {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn insert<I: IntoIterator<Item = String>>(&mut self, path: I) {
        let mut current = self;
        for s in path {
            current = current.0.entry(s).or_insert_with(|| Trie::new())
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let mut trie = Trie::new();
    for _ in 0..n {
        let k = input.value();
        trie.insert((0..k).map(|_| String::from_utf8(input.token().to_vec()).unwrap()));
    }

    fn print_trie(output_buf: &mut Vec<u8>, trie: &Trie) {
        fn dfs(output_buf: &mut Vec<u8>, trie: &Trie, depth: usize) {
            for (k, v) in trie.0.iter() {
                for _ in 0..depth * 2 {
                    output_buf.push(b'-');
                }
                writeln!(output_buf, "{}", k).unwrap();
                dfs(output_buf, &v, depth + 1);
            }
        }
        dfs(output_buf, trie, 0);
    }

    print_trie(&mut output_buf, &trie);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
