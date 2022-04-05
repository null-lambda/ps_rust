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

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::iter::once;

    let n_verts: usize = input.value();
    let b: usize = input.value();
    let n_projects: usize = input.value();
    let n_edges: usize = input.value();

    let mut neighbors: Vec<Vec<(usize, u32)>> = (0..n_verts).map(|_| Vec::new()).collect();
    let mut neighbors_rev: Vec<Vec<(usize, u32)>> = (0..n_verts).map(|_| Vec::new()).collect();
    for _ in 0..n_edges {
        let u: usize = input.value();
        let v: usize = input.value();
        let dist: u32 = input.value();
        neighbors[u - 1].push((v - 1, dist));
        neighbors_rev[v - 1].push((u - 1, dist));
    }

    fn dijkstra(start: usize, neighbors: &[Vec<(usize, u32)>]) -> Vec<u32> {
        let n = neighbors.len();
        const INF: u32 = 1_000_000_000;
        let mut dist = vec![INF; n];
        dist[start] = 0;
        let mut queue = BinaryHeap::from(vec![Reverse((0, start))]);
        while let Some(Reverse((du, u))) = queue.pop() {
            if du != dist[u] {
                continue;
            }
            for &(v, d_uv) in &neighbors[u] {
                let dv_new = du + d_uv;
                if dv_new < dist[v] {
                    dist[v] = dv_new;
                    queue.push(Reverse((dv_new, v)));
                }
            }
        }
        dist
    }

    let dist_to = dijkstra(b, &neighbors);
    let dist_from = dijkstra(b, &neighbors_rev);

    let mut cost: Vec<u32> = dist_to
        .into_iter()
        .zip(dist_from)
        .map(|(d1, d2)| d1 + d2)
        .take(b)
        .collect();
    cost.sort_unstable();

    let cost_acc: Vec<u64> = once(0)
        .chain(cost.into_iter().scan(0, |acc, x| {
            *acc += x as u64;
            Some(*acc)
        }))
        .collect();

    /*
    const N_MAX: usize = 5000;
    let mut dp = [[0; N_MAX + 1]; 2];
    for i in 1..=b {
        dp[1][i] = (i as u32 - 1) * cost_acc[i];
    }

    // O(s b^2) solution
    for s in 2..=n_projects {
        for i in 1..=b {
            dp[s % 2][i] = (0..i)
                .map(|j| {
                    dp[(s - 1) % 2][j] + (i as u64 - j as u64 - 1) * (cost_acc[i] - cost_acc[j])
                })
                .min()
                .unwrap();
        }
        println!("{:?}", dp[s % 2][b]);
    }
    */

    // O(s b log b)
    // divide & conquer optimization
    const N_MAX: usize = 5000;

    struct DnCState {
        dp: [[(u64, usize); N_MAX + 1]; 2],
    }

    struct DnCEnv<F>
    where
        F: Fn(usize, usize) -> u64,
    {
        c: F,
        s: usize,
    }

    fn dnc(
        state: &mut DnCState,
        env: &DnCEnv<impl Fn(usize, usize) -> u64>,
        start: usize,
        end: usize,
        j_start: usize,
        j_end: usize,
    ) {
        if start >= end {
            return;
        }
        let i = (start + end) / 2;
        state.dp[env.s % 2][i] = (0.max(j_start)..i.min(j_end))
            .map(|j| (state.dp[(env.s - 1) % 2][j].0 + (env.c)(j, i), j))
            .min()
            .unwrap();

        let (_, opt) = state.dp[env.s % 2][i];
        let opt = opt as usize;

        dnc(state, env, start, i, j_start, opt + 1);
        dnc(state, env, i + 1, end, opt, j_end);
    }
    // println!("{:?}", state.dp[0]);

    let mut state = DnCState {
        dp: [[(0, 0); N_MAX + 1]; 2]
    };
    for i in 1..=b {
        state.dp[1][i].0 = (i as u64 - 1) * cost_acc[i];
    }

    let c = |j: usize, i: usize| (i as u64 - j as u64 - 1) * (cost_acc[i] - cost_acc[j]);

    for s in 2..=n_projects {
        dnc(&mut state, &DnCEnv { c, s }, s + 1, b + 1, s, b + 1);
        // println!("{:?}", state.dp[g % 2]);
    }
    let (result, _) = state.dp[n_projects % 2][b];
    println!("{}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
