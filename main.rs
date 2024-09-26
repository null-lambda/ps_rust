#[allow(dead_code)]
mod fast_io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        fn value<T: FromStr>(&mut self) -> T
        where
            <T as FromStr>::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    // cheap and unsafe whitespace check
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    use std::io::{BufRead, BufReader, BufWriter, Read, Stdin, Stdout};

    pub struct InputAtOnce {
        buf: Box<[u8]>,
        cursor: usize,
    }

    impl<'a> InputAtOnce {
        pub fn new(buf: Box<[u8]>) -> Self {
            Self { buf, cursor: 0 }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.buf.len() - self.cursor);
            let slice = &self.buf[self.cursor..self.cursor + n];
            self.cursor += n;
            slice
        }
    }

    impl<'a> InputStream for InputAtOnce {
        fn token(&mut self) -> &[u8] {
            self.take(
                self.buf[self.cursor..]
                    .iter()
                    .position(|&c| !is_whitespace(c))
                    .expect("no available tokens left"),
            );
            self.take(
                self.buf[self.cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.buf.len() - self.cursor),
            )
        }

        fn line(&mut self) -> &[u8] {
            let line = self.take(
                self.buf[self.cursor..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map(|idx| idx + 1)
                    .unwrap_or_else(|| self.buf.len() - self.cursor),
            );
            trim_newline(line)
        }
    }

    pub struct LineSyncedInput<R: BufRead> {
        line_buf: Vec<u8>,
        line_cursor: usize,
        inner: R,
    }

    impl<R: BufRead> LineSyncedInput<R> {
        pub fn new(r: R) -> Self {
            Self {
                line_buf: Vec::new(),
                line_cursor: 0,
                inner: r,
            }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.line_buf.len() - self.line_cursor);
            let slice = &self.line_buf[self.line_cursor..self.line_cursor + n];
            self.line_cursor += n;
            slice
        }

        fn eol(&self) -> bool {
            self.line_cursor == self.line_buf.len()
        }

        fn refill_line_buf(&mut self) -> bool {
            self.line_buf.clear();
            self.line_cursor = 0;
            let result = self.inner.read_until(b'\n', &mut self.line_buf).is_ok();
            result
        }
    }

    impl<R: BufRead> InputStream for LineSyncedInput<R> {
        fn token(&mut self) -> &[u8] {
            loop {
                if self.eol() {
                    let b = self.refill_line_buf();
                    if !b {
                        panic!(); // EOF
                    }
                }
                self.take(
                    self.line_buf[self.line_cursor..]
                        .iter()
                        .position(|&c| !is_whitespace(c))
                        .unwrap_or_else(|| self.line_buf.len() - self.line_cursor),
                );

                let idx = self.line_buf[self.line_cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.line_buf.len() - self.line_cursor);
                if idx > 0 {
                    return self.take(idx);
                }
            }
        }

        fn line(&mut self) -> &[u8] {
            if self.eol() {
                self.refill_line_buf();
            }

            self.line_cursor = self.line_buf.len();
            trim_newline(self.line_buf.as_slice())
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let mut reader = BufReader::new(std::io::stdin().lock());
        let mut buf: Vec<u8> = vec![];
        reader.read_to_end(&mut buf).unwrap();
        let buf = buf.into_boxed_slice();
        InputAtOnce::new(buf)
    }

    pub fn stdin_buf() -> LineSyncedInput<BufReader<Stdin>> {
        LineSyncedInput::new(BufReader::new(std::io::stdin()))
    }

    pub fn stdout_buf() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

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

fn main() {
    use fast_io::InputStream;
    use std::cmp::Reverse;
    use std::collections::*;
    use std::io::Write;
    use std::iter;

    let mut input = fast_io::stdin_buf();
    let mut output = fast_io::stdout_buf();

    let n: usize = input.value();

    let mut trie = trie::Trie::new();
    let mut words: Vec<Box<[u8]>> = (0..n).map(|_| input.token().into()).collect();
    words.sort_unstable_by(|s, t| s.len().cmp(&t.len()).then_with(|| s.cmp(t).reverse()));

    let scores: Vec<u32> = words
        .iter()
        .map(|s| match s.len() {
            1 | 2 => 0,
            3 | 4 => 1,
            5 => 2,
            6 => 3,
            7 => 5,
            8 => 11,
            _ => panic!(),
        })
        .collect();
    for (i, word) in words.iter().enumerate() {
        trie.insert(word, i);
    }

    let b: usize = input.value();
    for _ in 0..b {
        let board: Vec<_> = (0..4).flat_map(|_| input.token()[..4].to_vec()).collect();

        fn dfs(
            board: &[u8],
            trie: &trie::Trie,
            node: usize,
            r: i32,
            c: i32,
            visited: &mut [bool],
            word_acc: &mut Vec<u8>,
            on_yield: &mut impl FnMut(usize),
        ) {
            let pos = (r * 4 + c) as usize;
            if visited[pos] {
                return;
            }
            if word_acc.len() > 8 {
                return;
            }
            let k = (board[pos] - b'A') as usize;
            let node_next = trie.nodes[node].children[k];
            if node_next == trie::UNSET {
                return;
            }

            visited[pos] = true;
            word_acc.push(board[pos]);

            if trie.nodes[node_next].terminal() {
                on_yield(trie.nodes[node_next].word_idx);
            }

            for (dr, dc) in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ] {
                let (r_next, c_next) = (r + dr, c + dc);
                if !((0..4).contains(&r_next) && (0..4).contains(&c_next)) {
                    continue;
                }

                dfs(
                    board, trie, node_next, r_next, c_next, visited, word_acc, on_yield,
                );
            }

            visited[pos] = false;
            word_acc.pop();
        }

        let mut found = HashSet::new();
        for r0 in 0..4 {
            for c0 in 0..4 {
                dfs(
                    &board,
                    &trie,
                    0,
                    r0 as i32,
                    c0 as i32,
                    &mut vec![false; 16],
                    &mut vec![],
                    &mut |word_idx| {
                        found.insert(word_idx);
                    },
                );
            }
        }
        let max_idx = found.iter().copied().max().unwrap();
        let score = found.iter().map(|&idx| scores[idx]).sum::<u32>();
        let s = unsafe { std::str::from_utf8_unchecked(&words[max_idx]) };
        writeln!(output, "{} {} {}", score, s, found.len()).unwrap();
    }
}
