use std::io::Write;

use fenwick_tree::FenwickTree;

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

pub mod fenwick_tree {
    pub trait Group {
        type Elem: Clone;
        fn id(&self) -> Self::Elem;
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
    }

    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n_ceil = n.next_power_of_two();
            let data = (0..n_ceil).map(|_| group.id()).collect();
            Self { n, group, data }
        }

        pub fn add(&mut self, mut idx: usize, value: G::Elem) {
            while idx < self.n {
                self.group.add_assign(&mut self.data[idx], value.clone());
                idx |= idx + 1;
            }
        }
        pub fn get(&self, idx: usize) -> G::Elem {
            self.sum_range(idx..idx + 1)
        }

        pub fn sum_prefix(&self, idx: usize) -> G::Elem {
            let mut res = self.group.id();
            let mut r = idx;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::Elem {
            let mut res = self.sum_prefix(range.end);
            self.group
                .sub_assign(&mut res, self.sum_prefix(range.start));
            res
        }
    }
}

const P: u32 = 1_000_000_007;
struct ModP;

impl ModP {
    fn add(lhs: u32, rhs: u32) -> u32 {
        if lhs + rhs >= P {
            lhs + rhs - P
        } else {
            lhs + rhs
        }
    }

    fn sub(lhs: u32, rhs: u32) -> u32 {
        if lhs >= rhs {
            lhs - rhs
        } else {
            lhs + P - rhs
        }
    }
}

impl fenwick_tree::Group for ModP {
    type Elem = u32;
    fn id(&self) -> u32 {
        0
    }
    fn add_assign(&self, lhs: &mut u32, rhs: u32) {
        *lhs = Self::add(*lhs, rhs);
    }
    fn sub_assign(&self, lhs: &mut u32, rhs: u32) {
        *lhs = Self::sub(*lhs, rhs);
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

    for i_tc in 1..=input.value() {
        let n: usize = input.value();
        let m: usize = input.value();
        let x: u64 = input.value();
        let y: u64 = input.value();
        let z: u64 = input.value();

        let mut bs: Vec<u32> = (0..m).map(|_| input.value()).collect();
        let mut xs: Vec<u32> = (0..n)
            .map(|i| {
                let res = bs[i % m];
                bs[i % m] = ((x * bs[i % m] as u64 + y * (i as u64 + 1)) % z) as u32;
                res
            })
            .collect();

        let (_, x_map_inv) = compress_coord(xs.iter().cloned());
        for x in &mut xs {
            *x = x_map_inv[x];
        }

        let x_max = x_map_inv.len() - 1;
        let mut count = FenwickTree::new(x_max + 1, ModP);

        let mut acc = 0u32;
        for x in xs {
            let y = ModP::add(1, count.sum_prefix(x as usize));
            acc = ModP::add(acc as u32, y as u32) as u32;
            count.add(x as usize, y);
        }
        writeln!(output, "Case #{}: {}", i_tc, acc).unwrap();
    }
}
