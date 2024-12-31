use std::io::Write;

use fenwick_tree::*;

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
        n_ceil: usize,
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n_ceil = n.next_power_of_two();
            let data = (0..n_ceil).map(|_| group.id()).collect();
            Self {
                n,
                n_ceil,
                group,
                data,
            }
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

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::Elem {
            let mut res = self.group.id();
            let mut r = range.end;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            let mut l = range.start;
            while l > 0 {
                self.group.sub_assign(&mut res, self.data[l - 1].clone());
                l &= l - 1;
            }

            res
        }
    }
}

struct Additive;

impl Group for Additive {
    type Elem = i64;

    fn id(&self) -> Self::Elem {
        0
    }

    fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs += rhs;
    }

    fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs -= rhs;
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let c: usize = input.value();
    let mut ls = vec![];
    let mut ts = vec![];
    for _ in 0..n {
        ls.push(input.value::<i64>());
        ts.push(input.value::<i64>());
    }

    let mut l_sum = ls.iter().sum::<i64>();

    let t_max = 100_000;
    let mut count = FenwickTree::new(t_max as usize + 1, Additive);
    let mut prod = FenwickTree::new(t_max as usize + 1, Additive);
    for &t in &ts {
        count.add(t as usize, 1);
        prod.add(t as usize, t);
    }

    let mut ts_sorted = ts.clone();
    ts_sorted.sort_unstable();

    let mut t_conv = (0..n).map(|i| (n - i) as i64 * ts_sorted[i]).sum::<i64>();
    writeln!(output, "{}", l_sum - t_conv).unwrap();

    for _ in 0..c {
        let r = input.value::<usize>() - 1;
        let l_new = input.value::<i64>();
        let t_new = input.value::<i64>();
        l_sum += l_new - ls[r];
        ls[r] = l_new;

        let t_old = ts[r];
        ts[r] = t_new;
        t_conv -= t_old * (n as i64 - count.sum_range(0..t_old as usize))
            + prod.sum_range(0..t_old as usize);
        count.add(t_old as usize, -1);
        prod.add(t_old as usize, -t_old);

        count.add(t_new as usize, 1);
        prod.add(t_new as usize, t_new);
        t_conv += t_new * (n as i64 - count.sum_range(0..t_new as usize))
            + prod.sum_range(0..t_new as usize);
        writeln!(output, "{}", l_sum - t_conv).unwrap();
    }
}
