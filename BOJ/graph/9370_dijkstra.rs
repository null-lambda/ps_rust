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

const INF: u32 = 1_000_000_000;

// (dist, idx)
type Neighbor = (u32, usize);
fn dijkstra(neighbors: &[Vec<Neighbor>], start: usize) -> Vec<u32> {
    use std::collections::BinaryHeap;

    let n = neighbors.len();
    let mut dist = vec![INF; n];
    dist[start] = 0;
    let mut queue = BinaryHeap::new();
    queue.push((0, start));
    while let Some((du, u)) = queue.pop() {
        if du != dist[u] {
            continue;
        }
        for &(d_uv, v) in &neighbors[u] {
            let dv_new = du + d_uv;
            if dv_new < dist[v] {
                dist[v] = dv_new;
                queue.push((dv_new, v));
            }
        }
    }
    dist
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let test_cases = input.value();
    for _ in 0..test_cases {
        let (n, m, t) = (input.value(), input.value(), input.value());
        let (s, g, h): (usize, usize, usize) = (input.value(), input.value(), input.value());
        let (s, g, h) = (s - 1, g - 1, h - 1);
        assert!(n >= 2);

        let mut neighbors: Vec<Vec<Neighbor>> = (0..n).map(|_| Vec::new()).collect();
        for _ in 0..m {
            let (a, b, d): (usize, usize, _) = (input.value(), input.value(), input.value());
            let (a, b) = (a - 1, b - 1);
            neighbors[a].push((d, b));
            neighbors[b].push((d, a));
        }
        let neighbors = neighbors;
        let mut destinations: Vec<usize> = (0..t).map(|_| input.value::<usize>() - 1).collect();
        destinations.sort_unstable();
        let destinations = destinations;

        let dist_s = dijkstra(&neighbors, s);
        let dist_g = dijkstra(&neighbors, g);
        let dist_h = dijkstra(&neighbors, h);
        let dist_gh = neighbors[g]
            .iter()
            .find_map(|&(d, v)| (v == h).then(|| d))
            .unwrap();

        destinations.iter().for_each(|&dest| {
            let d = dist_s[dest];
            if dist_g[s] + dist_gh + dist_h[dest] == d || dist_h[s] + dist_gh + dist_g[dest] == d {
                write!(output_buf, "{} ", dest + 1);
            }
        });
        writeln!(output_buf);
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
