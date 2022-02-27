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

    let n = input.value();
    let n_edges = input.value();
    let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
    for _ in 0..n_edges {
        let u: usize = input.value();
        let v: usize = input.value();
        let (u, v) = (u - 1, v - 1);
        neighbors[u].push(v);
    }
    let score: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let start: usize = input.value::<usize>() - 1;
    let n_goal = input.value();
    let goals: Vec<u32> = (0..n_goal).map(|_| input.value::<u32>() - 1).collect();

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

    // should be optimized 
    /*
    for start in 0..n {
        if state.order[start] == NOT_VISITED {
            dfs(start, &neighbors, &mut state);
        }
    }*/
    dfs(start, &neighbors, &mut state);
    

    // contract each scc into a point
    let DfsState { scc_index, scc, order, .. } = state;
    let n_scc = scc.len();
    let mut scc_parents: Vec<Vec<usize>> = (0..n_scc).map(|_| Vec::new()).collect();
    for u in 0..n {
        if order[u] != NOT_VISITED {
            for &v in &neighbors[u] {
                // belongs to different components
                if scc_index[u] != scc_index[v] {
                    scc_parents[scc_index[v]].push(scc_index[u]);
                }
            }
        }
    }
    
    for i in 0..n_scc {
        scc_parents[i].sort_unstable();
        scc_parents[i].dedup();
    }

    let scc_scores: Vec<u32> = scc
        .iter()
        .map(|component| component.iter().map(|&i| score[i]).sum())
        .collect();

    // dp on tree
    // note that scc is sorted in reverse topological order
    let mut total_scores = scc_scores;
    for i in (0..n_scc).rev() {
        total_scores[i] += scc_parents[i]
            .iter()
            .map(|&p| total_scores[p])
            .max()
            .unwrap_or(0);
    }

    // println!("{:?}", scc);
    // println!("{:?}", scc_parents.iter().enumerate().collect::<Vec<_>>());
    // println!("{:?}", total_scores);
    let result = goals
        .into_iter()
        .filter(|&i| order[i as usize] != NOT_VISITED)
        .map(|i| total_scores[scc_index[i as usize]])
        .max()
        .unwrap();
    println!("{}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
