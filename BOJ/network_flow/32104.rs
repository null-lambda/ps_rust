use std::{
    collections::{BTreeMap, VecDeque},
    io::Write,
};

mod simple_io {
    use std::string::*;

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

struct DisjointIntervals {
    endpoints: BTreeMap<u32, bool>,
    count: u32,
}

impl DisjointIntervals {
    fn singleton(l: u32, r: u32) -> Self {
        Self {
            endpoints: [(l, true), (r, false)].into(),
            count: r - l,
        }
    }

    fn union(&mut self, mut other: Self) {
        if self.endpoints.len() < other.endpoints.len() {
            std::mem::swap(self, &mut other);
        }
        // println!("{:?}", self.endpoints);
        // println!("{:?}", other.endpoints);

        let mut other_endpoints = other.endpoints.into_keys();
        while let Some(l) = other_endpoints.next() {
            let r = other_endpoints.next().unwrap();

            let inner = || self.endpoints.range(l..=r);
            if inner().next().is_none() {
                let is_outer_closed = self
                    .endpoints
                    .range(..l)
                    .next_back()
                    .map_or(false, |(_, &b)| b);
                if !is_outer_closed {
                    self.endpoints.insert(l, true);
                    self.endpoints.insert(r, false);
                    self.count += r - l;
                }
                continue;
            }

            let (&_, &is_left_closed) = inner().next().unwrap();
            let (&_, &is_right_open) = inner().next_back().unwrap();
            let is_right_closed = !is_right_open;

            let mut to_remove: VecDeque<u32> = inner().map(|(&x, _)| x).collect();
            for &x in &to_remove {
                self.endpoints.remove(&x);
            }

            if is_left_closed {
                self.endpoints.insert(l, true);
                to_remove.push_front(l);
            }

            if is_right_closed {
                self.endpoints.insert(r, false);
                to_remove.push_back(r);
            }

            // println!("{:?}", to_remove);
            for i in 0..to_remove.len() / 2 {
                let l = to_remove[i * 2];
                let r = to_remove[i * 2 + 1];
                self.count += r - l;
            }
        }
        // println!("{:?} {}", self.endpoints, self.count);
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let intervals = (0..n).map(|_| (input.value::<u32>(), input.value::<u32>() + 1));
    let mut dsu =
        collections::DisjointMap::new(intervals.map(|(l, r)| DisjointIntervals::singleton(l, r)));

    for _ in 0..q {
        match input.token() {
            "1" => {
                let a = input.value::<usize>() - 1;
                let b = input.value::<usize>() - 1;
                dsu.merge(a, b, |mut a, b| {
                    a.union(b);
                    a
                });
            }
            "2" => {
                let a = input.value::<usize>() - 1;
                writeln!(output, "{}", dsu.get_mut(a).count).unwrap();
            }
            _ => panic!(),
        }
    }
}
