use std::{cmp::Reverse, collections::HashMap, io::Write};

use collections::DisjointMap;

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

mod collections {
    use std::cell::Cell;
    use std::mem::{self, MaybeUninit};

    pub struct DisjointMap<T> {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        values: Vec<MaybeUninit<T>>,
    }

    impl<T> DisjointMap<T> {
        pub fn new(values: impl IntoIterator<Item = T>) -> Self {
            let values: Vec<_> = values.into_iter().map(MaybeUninit::new).collect();
            let n = values.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                values,
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

        pub fn get_size(&self, u: usize) -> u32 {
            self.find_root_with_size(u).1
        }

        pub fn get_mut(&mut self, u: usize) -> &mut T {
            let r = self.find_root(u);
            unsafe { self.values[r].assume_init_mut() }
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(
            &mut self,
            u: usize,
            v: usize,
            mut combine_values: impl FnMut(T, T) -> T,
        ) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            let value = unsafe {
                MaybeUninit::new(combine_values(
                    self.values[u].assume_init_read(),
                    self.values[v].assume_init_read(),
                ))
            };
            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            self.values[u] = value;
            true
        }
    }

    impl<T> Drop for DisjointMap<T> {
        fn drop(&mut self) {
            for u in 0..self.parent_or_size.len() {
                if self.get_parent_or_size(u).is_err() {
                    unsafe {
                        self.values[u].assume_init_drop();
                    }
                }
            }
        }
    }
}

struct CityData {
    storages: HashMap<u32, u32>,
}

impl CityData {
    fn singleton(owner: u32) -> Self {
        Self {
            storages: [(owner, 1)].into(),
        }
    }

    fn union(&mut self, mut other: Self, mut count_links: impl FnMut(u32, u32)) {
        if self.storages.len() < other.storages.len() {
            std::mem::swap(self, &mut other);
        }

        for (owner, count) in other.storages {
            self.storages
                .entry(owner)
                .and_modify(|c| {
                    count_links(owner, *c * count);
                    *c += count
                })
                .or_insert(count);
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let m: usize = input.value();

    let owners = (0..n).map(|_| input.value::<u32>() - 1);
    let mut data = DisjointMap::new(owners.map(CityData::singleton));
    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let w: u32 = input.value();
        edges.push((w, u, v));
    }

    edges.sort_unstable_by_key(|&(w, _, _)| Reverse(w));

    let mut ans = vec![0; k];
    for (w, u, v) in edges {
        data.merge(u, v, |mut u, v| {
            u.union(v, |owner, count| {
                ans[owner as usize] += w as u64 * count as u64;
            });
            u
        });
    }

    for i in 1..n {
        data.merge(0, i, |mut u, v| {
            u.union(v, |_, _| {});
            u
        });
    }
    for x in ans {
        writeln!(output, "{}", x).unwrap();
    }
}
