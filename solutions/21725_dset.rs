use std::{collections::HashMap, io::Write};

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
    use std::mem;
    use std::mem::MaybeUninit;

    pub struct DisjointMap<T> {
        // represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        values: Vec<MaybeUninit<T>>,
    }

    impl<T> DisjointMap<T> {
        pub fn new(values: impl IntoIterator<Item = T>) -> Self {
            let node_weights: Vec<_> = values.into_iter().map(|c| MaybeUninit::new(c)).collect();
            let n = node_weights.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                values: node_weights,
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            -self.parent_or_size[self.find_root(u)].get() as u32
        }

        pub fn get_mut(&mut self, u: usize) -> &mut T {
            let r = self.find_root(u);
            unsafe { self.values[r].assume_init_mut() }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if self.parent_or_size[u].get() < 0 {
                u
            } else {
                let root = self.find_root(self.parent_or_size[u].get() as usize);
                self.parent_or_size[u].set(root as i32);
                root
            }
        }
        // returns whether two set were different
        pub fn merge(
            &mut self,
            mut u: usize,
            mut v: usize,
            mut combine_values: impl FnMut(T, T) -> T,
        ) -> bool {
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            let size_u = -self.parent_or_size[u].get() as i32;
            let size_v = -self.parent_or_size[v].get() as i32;
            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.parent_or_size[v].set(u as i32);
            self.parent_or_size[u].set(-(size_u + size_v));

            unsafe {
                self.values[u] = MaybeUninit::new(combine_values(
                    self.values[u].assume_init_read(),
                    self.values[v].assume_init_read(),
                ))
            }
            true
        }
    }

    impl<T> Drop for DisjointMap<T> {
        fn drop(&mut self) {
            for i in 0..self.parent_or_size.len() {
                if self.parent_or_size[i].get() < 0 {
                    unsafe {
                        self.values[i].assume_init_drop();
                    }
                }
            }
        }
    }
}

struct Debt {
    individual: HashMap<u32, i64>,
    common: i64, // Lazy evaluation
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut dmap = DisjointMap::new((0..n as u32).map(|i| Debt {
        individual: [(i, 0)].into(),
        common: 0,
    }));

    for _ in 0..m {
        match input.token() {
            "1" => {
                let x = input.value::<u32>() - 1;
                let y = input.value::<u32>() - 1;

                dmap.merge(x as usize, y as usize, |mut x, mut y| {
                    if x.individual.len() < y.individual.len() {
                        std::mem::swap(&mut x, &mut y);
                    }
                    let dx = y.common - x.common;
                    x.individual
                        .extend(y.individual.into_iter().map(|(i, y)| (i, y + dx)));
                    x
                });
            }
            "2" => {
                let x = input.value::<usize>() - 1;
                let c: i64 = input.value();

                let debt = dmap.get_mut(x);
                debt.common += c / debt.individual.len() as i64;
                *debt.individual.get_mut(&(x as u32)).unwrap() -= c;
            }
            _ => panic!(),
        }
    }

    assert!(dmap.get_size(0) == n as u32);
    let debt = dmap.get_mut(0);
    for x in debt.individual.values_mut() {
        *x += debt.common;
    }
    debt.common = 0;

    let n_transactions = debt
        .individual
        .iter()
        .filter(|&(&i, &x)| i != 0 && x != 0)
        .count();
    writeln!(output, "{}", n_transactions).unwrap();
    for (&i, &d) in debt.individual.iter().filter(|&(&i, &x)| i != 0 && x != 0) {
        if d > 0 {
            writeln!(output, "{} 1 {}", i + 1, d).unwrap();
        } else if d < 0 {
            writeln!(output, "1 {} {}", i + 1, -d).unwrap();
        }
    }
}
