use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut dmap = DisjointMap::new((0..n).map(|i| Some(i as u32)));
    let mut pq: BinaryHeap<(u32, Reverse<u32>)> = (0..n).map(|i| (1, Reverse(i as u32))).collect();
    for _ in 0..q {
        match input.token() {
            "1" => {
                let u = input.value::<usize>() - 1;
                let v = input.value::<usize>() - 1;
                let previously_disjoint =
                    dmap.merge(u, v, |mu, mv| mu.zip(mv).map(|(mu, mv)| mu.min(mv)));
                let entry = dmap.get_mut(u);
                if previously_disjoint && entry.is_some() {
                    let size = dmap.get_size(u);
                    let mu = dmap.get_mut(u).unwrap();
                    pq.push((size, Reverse(mu)));
                } else {
                    *entry = None;
                }
            }
            "2" => loop {
                match pq.pop() {
                    Some((_size, Reverse(u))) => {
                        let entry = dmap.get_mut(u as usize);
                        if let Some(mu) = entry.as_mut().copied() {
                            *entry = None;
                            writeln!(output, "{}", mu + 1).unwrap();
                            break;
                        }
                    }
                    None => {
                        panic!("Input constraint violated");
                    }
                }
            },
            _ => panic!(),
        }
    }
}
