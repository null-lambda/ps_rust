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

    while !input.iter().all(|b| b.is_ascii_whitespace()) {
        let n_props: usize = input.value();
        let n: usize = n_props * 2;
        let n_participants: usize = input.value();

        // implication graph of the cnf
        let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();

        // p1 ~ pn to 0 ~ n-1
        // not(p1) ~ not(pn) to n ~ 2n-1
        let mut read_prop = || -> usize {
            let p: usize = input.value();
            let b: u8 = input.token()[0];
            match b {
                b'B' => p - 1,
                b'R' => 2 * n_props - p,
                _ => unreachable!(),
            }
        };
        let idx_to_prop = |i: usize| -> (usize, bool) {
            if i < n_props {
                (i, true)
            } else {
                (2 * n_props - 1 - i, false)
            }
        };
        let neg_idx = |i: usize| -> usize { 2 * n_props - 1 - i };
        for _ in 0..n_participants {
            let p = read_prop();
            let q = read_prop();
            let r = read_prop();
            neighbors[neg_idx(p)].push(q);
            neighbors[neg_idx(p)].push(r);
            neighbors[neg_idx(q)].push(p);
            neighbors[neg_idx(q)].push(r);
            neighbors[neg_idx(r)].push(p);
            neighbors[neg_idx(r)].push(q);
        }

        // tarjan algorithm
        const NOT_VISITED: u32 = 0;
        struct DfsState {
            path_stack: Vec<usize>,
            order_count: u32,
            order: Vec<u32>,
            low_link: Vec<u32>,
            finished: Vec<bool>,
            scc_index: Vec<usize>,
            scc_count: usize,
            // scc: Vec<Vec<usize>>,
        }

        let mut state = DfsState {
            path_stack: vec![],
            order_count: 1,
            order: vec![NOT_VISITED; n],
            low_link: vec![100_000_000; n],
            finished: vec![false; n],
            scc_index: vec![100_000_000; n],
            scc_count: 0, // scc: vec![],
        };

        fn dfs(u: usize, neighbors: &[Vec<usize>], state: &mut DfsState) {
            state.order[u] = state.order_count;
            state.low_link[u] = state.order_count;
            state.order_count += 1;
            state.path_stack.push(u);

            for &v in &neighbors[u] {
                if state.order[v] == NOT_VISITED {
                    dfs(v, neighbors, state);
                    state.low_link[u] = state.low_link[u].min(state.low_link[v]);
                } else if !state.finished[v] {
                    state.low_link[u] = state.low_link[u].min(state.order[v]);
                }
            }

            // if u is a root node, pop the stack and generate an scc
            if state.low_link[u] == state.order[u] {
                // let mut component = vec![];
                // let n_scc = state.scc.len();
                while {
                    let v = state.path_stack.pop().unwrap();
                    // component.push(v);
                    state.scc_index[v] = state.scc_count;
                    state.finished[v] = true;
                    v != u
                } {}
                state.scc_count += 1;
                // state.scc.push(component);
            }
        }

        for start in 0..n {
            if state.order[start] == NOT_VISITED {
                dfs(start, &neighbors, &mut state);
            }
        }

        let DfsState { scc_index, .. } = state;

        use std::cmp::Ordering;
        // variable assignment function
        let interpretation: Option<Vec<_>> = (0..n_props)
            .map(|i| match scc_index[i].cmp(&scc_index[neg_idx(i)]) {
                Ordering::Equal => None,
                Ordering::Greater => Some(false),
                Ordering::Less => Some(true),
            })
            .collect();

        match interpretation {
            Some(interpretation) => {
                for p_value in interpretation {
                    output_buf.push(if p_value { b'B' } else { b'R' });
                }
            }
            None => {
                writeln!(output_buf, "-1").unwrap();
            }
        }
        // println!("{:?}", neighbors.iter().enumerate().collect::<Vec<_>>());
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
