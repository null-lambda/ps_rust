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

    use std::iter::once;
    use std::collections::HashMap;

    let m: u32 = input.value();
    let n: u32 = input.value();
    let n_queries: usize = input.value();
    let n_verts = (m * n) as usize;

    let hs: Vec<u32> = (0..n_verts).map(|_| input.value()).collect();
    let hs: &Vec<_> = hs.as_ref();
    let mut coord_to_idx = |x1: u32, x2: u32| x1 * n as u32 + x2;
    let mut edges: Vec<_> = (0..m - 1)
        .flat_map(|i| {
            (0..n).flat_map(move |j| {
                let u = coord_to_idx(i, j);
                let v = coord_to_idx(i + 1, j);
                let weight = hs[u as usize].max(hs[v as usize]);
                once((u, v, weight)).chain(once((v, u, weight)))
            })
        })
        .chain((0..m).flat_map(|i| {
            (0..n - 1).flat_map(move |j| {
                let u = coord_to_idx(i, j);
                let v = coord_to_idx(i, j + 1);
                let weight = hs[u as usize].max(hs[v as usize]);
                once((u, v, weight)).chain(once((v, u, weight)))
            })
        }))
        .collect();
    edges.sort_unstable_by_key(|&(.., weight)| weight);
    let n_edges = edges.len();

    let queries: Vec<[_; 2]> = (0..n_queries)
        .map(|_| {
            let x1: u32 = input.value();
            let y1: u32 = input.value();
            let x2: u32 = input.value();
            let y2: u32 = input.value();
            [coord_to_idx(x1, y1), coord_to_idx(x2, y2)]
        })
        .collect();

    const H_MAX: u32 = 1_000_000;
    let mut bound: Vec<[u32; 2]> = (0..n_queries).map(|_| [1, H_MAX + 1]).collect();
    let mut finished: Vec<bool> = vec![false; n_queries];
    let mut result: Vec<Option<(u32, u32)>> = vec![None; n_queries];
    loop {
        let mut mid_indices = HashMap::<u32, Vec<usize>>::new();
        for (i, &[left, right]) in bound.iter().enumerate() {
            if !finished[i] {
                mid_indices.entry((left + right) / 2).or_default().push(i);
            }
        }
        if mid_indices.is_empty() {
            break;
        }

        let mut dset = DisjointSet::new(n_verts);
        for (mid, &(u, v, w)) in edges.iter().enumerate() {
            dset.merge(u as usize, v as usize);
            for &j in mid_indices
                .get(&(mid as u32))
                .into_iter()
                .flat_map(|indices| indices.iter())
            {
                let [left, right] = queries[j];
                if dset.find_root(left as usize) == dset.find_root(right as usize) {
                    bound[j][1] = mid as u32;
                    result[j] = Some(w);
                } else {
                    bound[j][0] = mid as u32 + 1;
                }
                if bound[j][0] == bound[j][1] {
                    finished[j] = true;
                }
            }
        }
    }

    for w in result.iter() {
            writeln!(output_buf, "{}", w.unwrap()).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
