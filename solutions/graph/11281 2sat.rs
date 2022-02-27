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

fn two_satisfiability()

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n_props: usize = input.value();
    let n: usize = n_props * 2;
    let n_edges: usize = input.value();

    // implication graph of the cnf
    let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();

    // 1 ~ p to 0 ~ n-1
    // -1 ~ p to n ~ 2n-1
    let prop_to_idx = |p: i32| -> usize {
        (if p > 0 {
            p - 1
        } else {
            (n_props as i32) - 1 - p
        }) as usize
    };
    let idx_to_prop = |i: usize| -> i32 {
        if i < n_props {
            i as i32 + 1
        } else  {
            n_props as i32 - 1 - i as i32
        }
    };
    
    for _ in 0..n_edges {
        let p: i32 = input.value();
        let q: i32 = input.value();
        neighbors[prop_to_idx(-p)].push(prop_to_idx(q));
        neighbors[prop_to_idx(-q)].push(prop_to_idx(p));
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

    let DfsState {
        scc_index,
        scc,
        ..
    } = state;
    // let n_scc = scc.len();

    let satisfiable =
        (1..=(n_props as i32)).all(|p| scc_index[prop_to_idx(p)] != scc_index[prop_to_idx(-p)]);
    if satisfiable {
        writeln!(output_buf, "1");
        
        // variable assignment function
        let mut interpretation = vec![None; n_props];
    
        // sccs are ordered in reverse topological order
        for component in &scc {
            for &i in component.iter() {
                let p = idx_to_prop(i);
                let (p, p_value) = (p.abs() as usize - 1, p > 0);
                if interpretation[p].is_some() {
                    break;
                }
                interpretation[p] = Some(p_value);
            }
        }
        for p_value in interpretation {
            write!(output_buf, "{} ", p_value.unwrap() as i8).unwrap();
        }
    } else {
        writeln!(output_buf, "0");
    }


    // println!("{:?}", neighbors.iter().enumerate().collect::<Vec<_>>());
    // println!("{:?}", scc);
    // println!("{:?}", scc_parents.iter().enumerate().collect::<Vec<_>>());

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
