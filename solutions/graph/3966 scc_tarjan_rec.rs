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

    let test_cases = input.value();
    for _ in 0..test_cases {
        let n = input.value();
        let n_edges = input.value();
        let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
        for _ in 0..n_edges {
            let u: usize = input.value();
            let v: usize = input.value();
            neighbors[u].push(v);
        }

        // tarjan algorithm
        const NOT_VISITED: u32 = 0;
        struct DfsState {
            path_stack: Vec<usize>,
            order_count: u32,
            order: Vec<u32>,
            low_link: Vec<u32>,
            scc_index: Vec<usize>,
            finished: Vec<bool>,
            scc: Vec<Vec<usize>>,
        }

        let mut state = DfsState {
            path_stack: vec![],
            order_count: 1,
            order: vec![NOT_VISITED; n],
            low_link: vec![100_000_000; n],
            scc_index: vec![100_000_000; n],
            finished: vec![false; n],
            scc: vec![],
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
                let mut component = vec![];
                let n_scc = state.scc.len();
                while {
                    let v = state.path_stack.pop().unwrap();
                    component.push(v);
                    state.scc_index[v] = n_scc;
                    state.finished[v] = true;
                    v != u
                } {}
                state.scc.push(component);
            }
        }

        for start in 0..n {
            if state.order[start] == NOT_VISITED {
                dfs(start, &neighbors, &mut state);
            }
        }

        let DfsState { scc_index, mut scc, .. } = state;
        let mut has_parent = vec![false; scc.len()];
        for u in 0..n {
            for &v in &neighbors[u] {
                // belongs to different components
                if scc_index[u] != scc_index[v] {
                    has_parent[scc_index[v]] = true;
                }
            }
        }

        let starting_area: Vec<_> = (0..scc.len())
            .filter(|&i| !has_parent[i])
            .collect();

        if let [i] = *starting_area.as_slice() {
            let component = &mut scc[i];
            component.sort_unstable();
            for j in component {
                writeln!(output_buf, "{}", j).unwrap();
            }
        } else {
            writeln!(output_buf, "Confused").unwrap();
        }
        writeln!(output_buf).unwrap();

        // println!("{:?}", scc);
        // println!("{:?}", starting_area);
    }
    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
