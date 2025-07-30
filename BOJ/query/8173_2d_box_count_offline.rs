use std::{collections::HashMap, io::Write};

use fenwick_tree::{FenwickTree, Group};

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
        type X: Clone;
        fn id(&self) -> Self::X;
        fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X);
        fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X);
    }

    #[derive(Clone)]
    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        sum: Vec<G::X>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n = n.next_power_of_two(); // Required for binary search
            let sum = (0..n).map(|_| group.id()).collect();
            Self { n, group, sum }
        }

        pub fn from_iter(iter: impl IntoIterator<Item = G::X>, group: G) -> Self {
            let mut sum: Vec<_> = iter.into_iter().collect();
            let n = sum.len();

            let n = n.next_power_of_two(); // Required for binary search
            sum.resize_with(n, || group.id());

            for i in 1..n {
                let prev = sum[i - 1].clone();
                group.add_assign(&mut sum[i], prev);
            }
            for i in (1..n).rev() {
                let j = i & (i + 1);
                if j >= 1 {
                    let prev = sum[j - 1].clone();
                    group.sub_assign(&mut sum[i], prev);
                }
            }

            Self { n, group, sum }
        }

        pub fn add(&mut self, mut idx: usize, value: G::X) {
            debug_assert!(idx < self.n);
            while idx < self.n {
                self.group.add_assign(&mut self.sum[idx], value.clone());
                idx |= idx + 1;
            }
        }

        // Exclusive prefix sum (0..idx)
        pub fn sum_prefix(&self, idx: usize) -> G::X {
            debug_assert!(idx <= self.n);
            let mut res = self.group.id();
            let mut r = idx;
            while r > 0 {
                self.group.add_assign(&mut res, self.sum[r - 1].clone());
                r &= r - 1;
            }
            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let mut res = self.sum_prefix(range.end);
            self.group
                .sub_assign(&mut res, self.sum_prefix(range.start));
            res
        }

        pub fn get(&self, idx: usize) -> G::X {
            self.sum_range(idx..idx + 1)
        }

        // find the first i, such that equiv pred(sum_range(0..=i)) == false
        pub fn partition_point_prefix(&self, mut pred: impl FnMut(&G::X) -> bool) -> usize {
            let p1_log2 = usize::BITS - self.n.leading_zeros();
            let mut idx = 0;
            let mut sum = self.group.id();
            for i in (0..p1_log2).rev() {
                let idx_next = idx | (1 << i);
                if idx_next > self.n {
                    continue;
                }
                let mut sum_next = sum.clone();
                self.group
                    .add_assign(&mut sum_next, self.sum[idx_next - 1].clone());
                if pred(&sum_next) {
                    sum = sum_next;
                    idx = idx_next;
                }
            }
            idx
        }
    }
}

struct Additive;

impl Group for Additive {
    type X = i32;

    fn id(&self) -> Self::X {
        0
    }

    fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs += rhs;
    }

    fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs -= rhs;
    }
}

fn naive() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let _n: i64 = input.value();
    let _m: i64 = input.value();
    let z: usize = input.value();
    let p: usize = input.value();

    let ps: Vec<[i64; 2]> = (0..z)
        .map(|_| std::array::from_fn(|_| input.value()))
        .collect();

    let dist = |a: [i64; 2], b: [i64; 2]| (a[0] - b[0]).abs() + (a[1] - b[1]).abs();
    for _ in 0..p {
        let r: [i64; 2] = std::array::from_fn(|_| input.value());
        let s: [i64; 2] = std::array::from_fn(|_| input.value());

        let mut ans = [0; 2];
        for &p in &ps {
            let d1 = dist(r, p);
            let d2 = dist(s, p);
            if d1 <= d2 {
                ans[0] += 1;
            }
            if d1 >= d2 {
                ans[1] += 1;
            }
        }

        let [a, b] = ans;
        let c = a + b - z;
        writeln!(output, "{} {} {}", a - c, b - c, c).unwrap();
    }
}

fn with_segtree() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let _n: i64 = input.value();
    let _m: i64 = input.value();
    let z: usize = input.value();
    let q: usize = input.value();

    let ps: Vec<[i64; 2]> = (0..z)
        .map(|_| std::array::from_fn(|_| input.value()))
        .collect();

    const UNSET: i32 = -1;

    let mut qs1d = HashMap::<_, Vec<_>>::new();
    let mut qs2d = HashMap::<_, Vec<_>>::new();
    let mut ans = vec![0i64; 2 * q];
    for i in 0..q as i32 {
        let r: [i64; 2] = std::array::from_fn(|_| input.value());
        let s: [i64; 2] = std::array::from_fn(|_| input.value());

        for (r, s, k) in [(r, s, 2 * i), (s, r, 2 * i + 1)] {
            let c = [s[0] - r[0], s[1] - r[1]];

            let t = c.map(|x| x.signum());
            let trans = |x: [i64; 2]| -> [i64; 2] { std::array::from_fn(|i| x[i] * t[i]) };
            let c = trans(c);
            let r = trans(r);
            let _s = trans(s);

            // Coordinate projections
            let t_x = [t[0], 0];
            let t_y = [0, t[1]];
            let t_x_y = [t[0], t[1]];
            let t_nx_ny = [-t[0], -t[1]];

            if [c[0], c[1]] == [0; 2] {
                ans[k as usize] += z as i64;
            } else if c[1] == 0 {
                qs1d.entry(t_x).or_default().push((c[0] / 2 + r[0], k));
            } else if c[0] == 0 {
                qs1d.entry(t_y).or_default().push((c[1] / 2 + r[1], k));
            } else if c[0] == c[1] {
                let h = c[0] + r[0] + r[1];
                qs1d.entry(t_x_y).or_default().push((h, k));
                qs2d.entry([t_x, t_nx_ny])
                    .or_default()
                    .push(([r[0], -h - 1], k));
                qs2d.entry([t_y, t_nx_ny])
                    .or_default()
                    .push(([r[1], -h - 1], k));
            } else if c[0] > c[1] {
                let p = (c[0] - c[1]) / 2 + r[0];
                let q = (c[0] + c[1]) / 2 + r[0];
                let h = (c[0] + c[1]) / 2 + r[0] + r[1];
                qs2d.entry([t_x, t_nx_ny])
                    .or_default()
                    .push(([p, -h - 1], k));
                qs2d.entry([t_x, t_x_y]).or_default().push(([q, h], k));
            } else {
                let p = (c[1] - c[0]) / 2 + r[1];
                let q = (c[0] + c[1]) / 2 + r[1];
                let h = (c[0] + c[1]) / 2 + r[0] + r[1];
                qs2d.entry([t_y, t_nx_ny])
                    .or_default()
                    .push(([p, -h - 1], k));
                qs2d.entry([t_y, t_x_y]).or_default().push(([q, h], k));
            }
        }
    }

    let dot = |p: [_; 2], q: [_; 2]| p[0] * q[0] + p[1] * q[1];
    for (tu, mut qs) in qs1d {
        let xs = ps.iter().map(|&p| dot(tu, p));
        qs.extend(xs.map(|x| (x, UNSET)));
        qs.sort_unstable();

        let mut acc = 0;
        for (_, k) in qs {
            if k == UNSET {
                acc += 1;
            } else {
                ans[k as usize] += acc;
            }
        }
    }

    for ([tu, tv], mut qs) in qs2d {
        let ps = ps.iter().map(|&p| [dot(tu, p), dot(tv, p)]);
        qs.extend(ps.map(|p| (p, UNSET)));

        qs.sort_unstable_by_key(|&(p, _)| p[1]);
        let mut y_bound = 0;
        let mut y_prev = 1 << 60;
        for i in 0..qs.len() {
            let y = qs[i].0[1];
            if y != y_prev {
                y_bound += 1;
            }
            qs[i].0[1] = y_bound - 1;
            y_prev = y;
        }

        qs.sort_unstable();
        let mut counter = FenwickTree::new(y_bound as usize, Additive);
        for (q, k) in qs {
            if k == UNSET {
                counter.add(q[1] as usize, 1);
            } else {
                ans[k as usize] += counter.sum_prefix(q[1] as usize + 1) as i64;
            }
        }
    }

    for i in 0..q {
        let a = ans[2 * i];
        let b = ans[2 * i + 1];
        let c = a + b - z as i64;
        writeln!(output, "{} {} {}", a - c, b - c, c).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|s| s == "naive") {
        naive();
    } else {
        with_segtree();
        // with_segtree();
    }
}
