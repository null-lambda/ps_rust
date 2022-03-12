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

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let m: usize = input.value();
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|_| {
            let l: usize = input.value();
            (0..l).map(|_| input.value::<usize>() - 1).collect()
        })
        .collect();

    // bipartite match
    const UNDEFINED: usize = usize::MAX;
    struct DfsState {
        visited: Vec<bool>,
        assignment: Vec<usize>,
    }

    let mut state = DfsState {
        visited: vec![false; m],
        assignment: vec![UNDEFINED; m],
    };

    fn dfs(node: usize, neighbors: &Vec<Vec<usize>>, state: &mut DfsState) -> bool {
        neighbors[node].iter().any(|&target| {
            if state.visited[target] {
                return false;
            }
            state.visited[target] = true;
            if state.assignment[target] == UNDEFINED
                || dfs(state.assignment[target], neighbors, state)
            {
                state.assignment[target] = node;
                return true;
            }
            false
        })
    }

    let result = (0..n)
        .filter(|&u| {
            state.visited.fill(false);
            dfs(u, &neighbors, &mut state)
        })
        .count();

    println!("{:?}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
