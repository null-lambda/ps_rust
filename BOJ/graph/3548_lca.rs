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

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
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

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

const EMPTY_NODE: usize = 0;
fn rank(parent: &Vec<usize>, mut u: usize) -> usize {
    let mut cnt = 0;
    while u != EMPTY_NODE {
        u = parent[u];
        cnt += 1;
    }
    cnt
}

fn move_upward(parent: &Vec<usize>, u: usize, n: usize) -> usize {
    (0..n).fold(u, |u, _| parent[u])
}

// https://www.acmicpc.net/problem/3584
fn least_common_ancestor(parent: &Vec<usize>, mut u: usize, mut v: usize) -> usize {
    if u == v {
        return u;
    }

    let (ru, rv) = (rank(&parent, u), rank(&parent, v));
    if ru < rv {
        v = move_upward(&parent, v, rv - ru);
    } else {
        u = move_upward(&parent, u, ru - rv);
    }

    while u != v {
        u = parent[u];
        v = parent[v];
    }
    u
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let t = input.value();
    for _ in 0..t {
        let n: usize = input.value();
        let mut parent = vec![EMPTY_NODE; n + 1];
        for _ in 0..(n - 1) {
            let (u, v): (usize, usize) = (input.value(), input.value());
            parent[v] = u;
        }
        let (u, v) = (input.value(), input.value());
        writeln!(output_buf, "{}", least_common_ancestor(&parent, u, v)).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
