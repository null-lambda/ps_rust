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
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
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
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut neighbors: Vec<Vec<_>> = (0..n).map(|_| vec![]).collect();
    let mut parents: Vec<Vec<_>> = (0..n).map(|_| vec![]).collect();
    let mut indegree: Vec<u32> = vec![0; n];
    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let dist: u32 = input.value();
        neighbors[u].push(v);
        parents[v].push((u, dist));
        indegree[v] += 1;
    }

    let start = input.value::<usize>() - 1;
    let end = input.value::<usize>() - 1;

    // topological sort
    use std::collections::VecDeque;
    use std::iter;
    let mut queue: VecDeque<usize> = vec![start].into();
    let order = iter::from_fn(move || {
        queue.pop_front().map(|u| {
            for &v in &neighbors[u] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
            u
        })
    });

    let mut max_dist = vec![0; n];
    for u in order {
        if !parents[u].is_empty() {
            max_dist[u] = parents[u]
                .iter()
                .map(|&(v, d_vu)| max_dist[v] + d_vu)
                .max()
                .unwrap();
        }
        // println!("{:?}", (u, &parents[u], &dp[u]));
    }
    //println!("{:?}", max_dist);
    println!("{}", max_dist[end]);
    
    let mut visited = vec![false; n];
    let mut dfs_stack = vec![end];
    let mut count_max_path = 0;
    while let Some(u) = dfs_stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        for &(v, d_vu) in &parents[u] {
            if max_dist[v] + d_vu == max_dist[u] {
                count_max_path += 1;
                if !visited[v] {
                    dfs_stack.push(v);
                }
            }
        }
    }
    println!("{:?}", count_max_path);


    std::io::stdout().write(&output_buf[..]).unwrap();
}
