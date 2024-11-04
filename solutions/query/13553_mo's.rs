use std::io::Write;

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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
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

use fenwick_tree::*;

struct Additive<T>(std::marker::PhantomData<T>);

impl<T> Additive<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: Default + std::ops::AddAssign + std::ops::SubAssign + Clone> Group for Additive<T> {
    type Elem = T;
    fn id(&self) -> Self::Elem {
        T::default()
    }
    fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs += rhs;
    }
    fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs -= rhs;
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: i32 = input.value();

    let xs: Vec<i32> = (0..n).map(|_| input.value()).collect();
    let x_max: i32 = *xs.iter().max().unwrap();

    let bucket_size = ((n as f64).sqrt().round() as usize).max(1);
    let mut count = FenwickTree::new(x_max as usize + 1, Additive::<i32>::new());

    let m: usize = input.value();
    let mut queries: Vec<(usize, usize, usize)> = (0..m)
        .map(|i| {
            let l = input.value::<usize>() - 1;
            let r = input.value::<usize>() - 1;
            (i, l, r)
        })
        .collect();
    queries.sort_unstable_by_key(|&(_, l, r)| (l / bucket_size, r));

    // Mo's
    let mut update_ans = |ans: &mut i64, x: i32, sign: i32| {
        let lb = (x - k).max(0) as usize;
        let ub = (x + k).min(x_max) as usize;
        if sign > 0 {
            *ans += count.sum_range(lb..ub + 1 as usize) as i64;
            count.add(x as usize, 1);
        } else {
            count.add(x as usize, -1);
            *ans -= count.sum_range(lb..ub + 1 as usize) as i64;
        }
    };

    let mut l_cur = 1;
    let mut r_cur = 0;
    let mut acc = 0;
    let mut ans = vec![0; m];
    for (i, l, r) in queries {
        while l_cur < l {
            update_ans(&mut acc, xs[l_cur], -1);
            l_cur += 1;
        }
        while l_cur > l {
            l_cur -= 1;
            update_ans(&mut acc, xs[l_cur], 1);
        }
        while r_cur < r {
            r_cur += 1;
            update_ans(&mut acc, xs[r_cur], 1);
        }
        while r_cur > r {
            update_ans(&mut acc, xs[r_cur], -1);
            r_cur -= 1;
        }
        ans[i] = acc;
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
