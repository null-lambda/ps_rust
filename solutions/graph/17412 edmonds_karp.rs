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
    let p: usize = input.value();
    let mut capacity: Vec<Vec<i8>> = (0..n).map(|_| vec![0; n]).collect();
    for _ in 0..p {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        capacity[u][v] = 1;
    }

    assert!(n >= 2);
    // edmonds-karp algorithm
    let mut flow: Vec<Vec<i8>> = (0..n).map(|_| vec![0; n]).collect();

    let mut total_flow: u32 = 0;
    let (start, end) = (0, 1);
    loop {
        use std::collections::VecDeque;
        const UNDEFINED: usize = usize::MAX;

        // find augumenting path, bfs
        let mut prev = vec![UNDEFINED; n];
        prev[start] = start;

        let path_found = (|| {
            let mut queue = VecDeque::from(vec![start]);
            while let Some(u) = queue.pop_front() {
                for v in 0..n {
                    if prev[v] == UNDEFINED && capacity[u][v] > flow[u][v] {
                        queue.push_back(v);
                        prev[v] = u;
                        if v == end {
                            return true;
                        }
                    }
                }
            }
            false
        })();
        if !path_found {
            break;
        }

        use std::iter::successors;
        let max_flow = successors(Some(end), |&u| (prev[u] != start).then(|| prev[u]))
            .map(|u| capacity[prev[u]][u] - flow[prev[u]][u])
            .min()
            .unwrap();
        successors(Some(end), |&u| (prev[u] != start).then(|| prev[u])).for_each(|u| {
            flow[prev[u]][u] += max_flow;
            flow[u][prev[u]] -= max_flow;
        });
        total_flow += max_flow as u32;
    }

    println!("{:?}", total_flow);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
