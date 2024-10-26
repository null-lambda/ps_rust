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

use std::io::Write;
use std::io::{BufReader, Read};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

const INF: i64 = 100_000_000_000_000;

// (dist, idx)
type Neighbor = (i64, usize);

// returns None if cycle occurs
fn bellman_ford(neighbors: &[Vec<Neighbor>]) -> Option<Vec<i64>> {
    let n = neighbors.len();
    let mut dist = vec![INF; n];
    dist[0] = 0;

    // relax in (n-1) iterations
    for _ in 0..(n - 1) {
        for u in 0..n {
            if dist[u] != INF {
                for &(d_uv, v) in &neighbors[u] {
                    dist[v] = dist[v].min(dist[u] + d_uv);
                }
            }
        }
    }

    // check cycles
    for u in 0..n {
        if dist[u] != INF {
            for &(d_uv, v) in &neighbors[u] {
                if dist[u] + d_uv < dist[v] {
                    return None;
                }
            }
        }
    }
    Some(dist)
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let m = input.value();

    let mut neighbors: Vec<Vec<Neighbor>> = (0..n).map(|_| Vec::new()).collect();
    for _ in 0..m {
        let (a, b, d): (usize, usize, _) = (input.value(), input.value(), input.value());
        let (a, b) = (a - 1, b - 1);
        neighbors[a].push((d, b));
    }

    let dist = bellman_ford(&neighbors);
    match dist {
        Some(dist) => dist[1..]
            .into_iter()
            .for_each(|&d| println!("{}", if d == INF { -1 } else { d })),
        None => {
            println!("-1");
        }
    }

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
