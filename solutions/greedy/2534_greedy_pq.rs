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

    let n: i64 = input.value();
    let k = input.value();
    let p = input.value();
    let mut neighbors: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    let mut neighbors_rev: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    let mut indegree: Vec<u32> = (0..k).map(|_| 0).collect();
    let mut indegree_rev: Vec<u32> = (0..k).map(|_| 0).collect();
    for _ in 0..p {
        let u: usize = input.value();
        let v: usize = input.value();
        neighbors[u].push(v);
        neighbors_rev[v].push(u);
        indegree[v] += 1;
        indegree_rev[u] += 1;
    }

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let min_path = {
        let mut queue: BinaryHeap<Reverse<usize>> = (0..k)
            .filter(|&u| indegree[u] == 0)
            .map(|u| Reverse(u))
            .collect();

        let mut result = vec![0; k];
        for i in (0..k as i32).rev() {
            let u = queue.pop().unwrap().0;
            for &v in &neighbors[u] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    queue.push(Reverse(v));
                }
            }
            result[u] = i;
        }
        result.into_iter()
    };

    let max_path = {
        let mut queue: BinaryHeap<Reverse<usize>> = (0..k)
            .filter(|&u| indegree_rev[u] == 0)
            .map(|u| Reverse(u))
            .collect();

        let mut result = vec![0; k];
        for i in n as i32 - k as i32..n as i32 {
            let u = queue.pop().unwrap().0;
            for &v in &neighbors_rev[u] {
                indegree_rev[v] -= 1;
                if indegree_rev[v] == 0 {
                    queue.push(Reverse(v));
                }
            }
            result[u] = i;
        }
        result.into_iter()
    };

    // println!("{:?}", min_path.clone().collect::<Vec<_>>());
    // println!("{:?}", max_path.clone().collect::<Vec<_>>());

    let p: i64 = 1_000_000_007;
    let result = min_path
        .rev()
        .zip(max_path.rev())
        .map(|(u1, u2)| (u2 - u1) as i64)
        .fold(0, |acc, u| (acc * n + u) % p);
    println!("{:?}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
