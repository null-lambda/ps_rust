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

use std::cell::Cell;
struct DisjointSet {
    pub parent: Vec<Cell<usize>>,
    pub size: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).map(|i| Cell::new(i)).collect(),
            size: vec![1; n],
        }
    }

    fn find_root(&self, u: usize) -> usize {
        if u == self.parent[u].get() {
            u
        } else {
            self.parent[u].set(self.find_root(self.parent[u].get()));
            self.parent[u].get()
        }
    }

    fn get_size(&self, u: usize) -> usize {
        self.size[self.find_root(u)]
    }

    // returns whether two set were different
    fn merge(&mut self, mut u: usize, mut v: usize) -> bool {
        u = self.find_root(u);
        v = self.find_root(v);
        if u == v {
            return false;
        }
        if self.size[u] > self.size[v] {
            std::mem::swap(&mut u, &mut v);
        }
        self.parent[v].set(u);
        self.size[u] += self.size[v];
        true
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::collections::HashMap;

    let n: usize = input.value();
    let n_edges: usize = input.value();
    let mut edges: Vec<(u32, u32, u32)> = (0..n_edges)
        .map(|_| {
            let u: u32 = input.value();
            let v: u32 = input.value();
            let w: u32 = input.value();
            (u - 1, v - 1, w)
        })
        .collect();
    edges.sort_unstable_by_key(|&(.., w)| w);

    let n_queries: usize = input.value();
    let queries: Vec<(usize, usize)> = (0..n_queries)
        .map(|_| {
            let u: usize = input.value();
            let v: usize = input.value();
            (u - 1, v - 1)
        })
        .collect();

    // parallel binary search
    let mut bound: Vec<(usize, usize)> = vec![(0, n_edges); n_queries];
    let mut finished: Vec<bool> = vec![false; n_queries];
    let mut result: Vec<Option<(u32, u32)>> = vec![None; n_queries];
    loop {
        let mut mid_indices = HashMap::<_, Vec<usize>>::new();
        for (i, &(left, right)) in bound.iter().enumerate() {
            if !finished[i] {
                mid_indices.entry((left + right) / 2).or_default().push(i);
            }
        }
        if mid_indices.is_empty() {
            break;
        }

        let mut dset = DisjointSet::new(n);
        for (mid, &(u, v, w)) in edges.iter().enumerate() {
            dset.merge(u as usize, v as usize);
            for &j in mid_indices
                .get(&mid)
                .into_iter()
                .flat_map(|indices| indices.iter())
            {
                let (left, right) = queries[j];
                if dset.find_root(left) == dset.find_root(right) {
                    bound[j].1 = mid;
                    result[j] = Some((w, dset.get_size(left) as u32));
                } else {
                    bound[j].0 = mid + 1;
                }
                if bound[j].0 == bound[j].1 {
                    finished[j] = true;
                }
            }
        }
    }

    for x in result.iter() {
        if let Some((w, v)) = x {
            writeln!(output_buf, "{} {}", w, v).unwrap();
        } else {
            writeln!(output_buf, "-1").unwrap();
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
