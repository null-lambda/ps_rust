use std::io::Write;

use fenwick_tree::{DeltaFenwickTree, Group};

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

pub mod fenwick_tree {
    pub trait Group {
        type Elem: Clone;
        fn id(&self) -> Self::Elem;
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
    }

    #[derive(Clone)]
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

    #[derive(Clone)]
    pub struct DeltaFenwickTree<G: Group> {
        delta: FenwickTree<G>,
    }

    impl<G: Group> DeltaFenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            Self {
                delta: FenwickTree::new(n + 1, group),
            }
        }

        pub fn add_postfix(&mut self, idx: usize, value: G::Elem) {
            debug_assert!(idx < self.delta.n);
            self.delta.add(idx, value);
        }

        pub fn add_range(&mut self, range: std::ops::Range<usize>, value: G::Elem) {
            debug_assert!(range.start <= self.delta.n && range.end <= self.delta.n);
            let mut neg = self.delta.group.id();
            self.delta.group.sub_assign(&mut neg, value.clone());
            self.delta.add(range.start, value.clone());
            self.delta.add(range.end, neg);
        }

        pub fn get(&self, idx: usize) -> G::Elem {
            self.delta.sum_prefix(idx + 1)
        }
    }
}

#[derive(Clone)]
struct Additive;

impl Group for Additive {
    type Elem = i32;
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

    let n: i64 = input.value();
    let mut t_curr: usize = input.value();

    let x_bound = 1_000_001usize;
    let mut jump_by_even = vec![(0, 0)];
    let mut jump_by_odd = vec![(0, 0)];
    assert!(x_bound % 2 == 1);

    let mut acc = 0;
    let mut pad = 4;
    for i in 1.. {
        acc += i;
        if acc % 2 == 0 {
            jump_by_even.push((i, acc / 2));
        } else {
            jump_by_odd.push((i, acc / 2 + 1));
        }

        if acc as usize >= x_bound {
            pad -= 1;
            if pad == 0 {
                break;
            }
        }
    }
    let common_len = jump_by_even.len().min(jump_by_odd.len());
    jump_by_even.truncate(common_len);
    jump_by_odd.truncate(common_len);

    let x_bound_half = (x_bound as usize + 1) / 2;
    let mut ys = vec![DeltaFenwickTree::new(x_bound_half, Additive); 2];

    let add_frog = |ys: &mut [DeltaFenwickTree<_>], x: usize, inv: bool| {
        debug_assert!(x <= x_bound);
        let sub_clip = |x: usize, dx: usize| x.checked_sub(dx).unwrap_or(0);
        let add_clip = |x: usize, dx: usize| (x + dx).min(x_bound_half);

        let (q, r) = (x / 2, x % 2);
        for i in 1..jump_by_even.len() {
            let dx_prev = jump_by_even[i - 1].1 as usize + 1;
            let dx = jump_by_even[i].1 as usize;
            let mut dy = jump_by_even[i].0 as i32;
            if inv {
                dy = -dy;
            }
            ys[r].add_range(add_clip(q, dx_prev)..add_clip(q + 1, dx), dy);
            ys[r].add_range(sub_clip(q, dx)..sub_clip(q + 1, dx_prev), dy);
        }
        for i in 1..jump_by_odd.len() {
            let dx_prev = jump_by_odd[i - 1].1 as usize + 1;
            let dx = jump_by_odd[i].1 as usize;
            let mut dy = jump_by_odd[i].0 as i32;
            if inv {
                dy = -dy;
            }

            ys[1 - r].add_range(
                add_clip(q + r, dx_prev - 1)..add_clip(q + r + 1, dx - 1),
                dy,
            );
            ys[1 - r].add_range(sub_clip(q + r, dx)..sub_clip(q + r + 1, dx_prev), dy);
        }
    };
    let query = |ys: &[DeltaFenwickTree<_>], x: usize| ys[x % 2].get(x / 2);

    for _ in 0..n {
        let x: usize = input.value();
        add_frog(&mut ys, x, false);
    }

    for _ in 0..input.value() {
        match input.token() {
            "t" => t_curr = input.value(),
            "+" => add_frog(&mut ys, input.value(), false),
            "-" => add_frog(&mut ys, input.value(), true),
            _ => panic!(),
        }
        writeln!(output, "{}", query(&ys, t_curr)).unwrap();
    }
}
