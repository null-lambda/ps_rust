use std::{cmp::Reverse, io::Write};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

mod dset {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

use std::{collections::HashMap, hash::Hash};

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let q: usize = input.value();
    let mut edges: Vec<_> = (0..m)
        .map(|_| {
            (
                input.value::<u32>() - 1,
                input.value::<u32>() - 1,
                input.value::<u32>(),
            )
        })
        .collect();
    edges.sort_unstable_by_key(|&(u, v, w)| (u, v, Reverse(w)));
    edges.dedup_by_key(|&mut (u, v, _)| (u, v));
    edges.sort_by_key(|&(.., w)| Reverse(w));
    let (w_map, w_inv) = compress_coord(edges.iter().map(|&(.., w)| w));
    let w_bound = w_map.len();
    edges.iter_mut().for_each(|(_, _, w)| *w = w_inv[w]);

    let queries: Vec<_> = (0..q)
        .map(|i| {
            (
                0,
                w_bound as u32,
                input.value::<u32>() - 1,
                input.value::<u32>() - 1,
                i as u32,
            )
        })
        .collect();

    let mut buckets = vec![vec![]; w_bound];
    for &(l, r, u, v, i) in &queries {
        buckets[(l + r >> 1) as usize].push((l, r, u, v, i));
    }

    let mut ans = vec![0; q];
    loop {
        let mut finished = true;

        let mut conn = dset::DisjointSet::new(n);
        let mut edges = edges.iter().cloned().peekable();

        for mid in (0..w_bound as u32).rev() {
            for (mut l, mut r, u, v, i) in std::mem::take(&mut buckets[mid as usize]) {
                if l == r {
                    continue;
                }
                finished = false;

                while let Some((u, v, _)) = edges.next_if(|&(.., w)| w >= mid) {
                    conn.merge(u as usize, v as usize);
                }

                if conn.find_root(u as usize) == conn.find_root(v as usize) {
                    l = mid + 1;
                } else {
                    r = mid;
                }

                if l == r {
                    ans[i as usize] = l - 1;
                } else {
                    buckets[(l + r >> 1) as usize].push((l, r, u, v, i));
                }
            }
        }

        if finished {
            break;
        }
    }

    for a in ans {
        writeln!(output, "{}", w_map[a as usize]).unwrap();
    }
}
