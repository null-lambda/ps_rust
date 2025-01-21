use std::io::Write;
use std::{collections::HashMap, hash::Hash};

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

impl fenwick_tree::Group for Additive {
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

fn on_lower_half(base: [i64; 2], p: [i64; 2]) -> bool {
    (p[1], p[0]) < (base[1], base[0])
}

fn signed_area(p: [i64; 2], q: [i64; 2], r: [i64; 2]) -> i64 {
    let dq = [q[0] - p[0], q[1] - p[1]];
    let dr = [r[0] - p[0], r[1] - p[1]];
    dq[0] * dr[1] - dq[1] * dr[0]
}

fn cmp_angle(p: [i64; 2], q: [i64; 2], r: [i64; 2]) -> std::cmp::Ordering {
    on_lower_half(p, q)
        .cmp(&on_lower_half(p, r))
        .then_with(|| 0.cmp(&signed_area(p, q, r)))
}

const UNSET: u32 = !0;

#[derive(Debug, Clone)]
struct HalfEdge {
    src: u32,
    cycle_next: u32,
}

fn twin(e: u32) -> u32 {
    e ^ 1
}

const X_BOUND: i32 = 1_000_000_010;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let q: usize = input.value();
    let ps: Vec<[i64; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();

    // Construct the planar graph
    let mut edges = vec![];
    let mut vert_neighbors = vec![vec![]; n];
    for e in 0..m as u32 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push(HalfEdge {
            src: u,
            cycle_next: UNSET,
        });
        edges.push(HalfEdge {
            src: v,
            cycle_next: UNSET,
        });
        vert_neighbors[u as usize].push(2 * e);
        vert_neighbors[v as usize].push(2 * e + 1);
    }

    // Connect adjacent edges in a cycle
    for u in 0..n {
        vert_neighbors[u].sort_unstable_by(|&e, &f| {
            cmp_angle(
                ps[u],
                ps[edges[twin(e) as usize].src as usize],
                ps[edges[twin(f) as usize].src as usize],
            )
        });

        for (&e, &f) in vert_neighbors[u]
            .iter()
            .zip(vert_neighbors[u].iter().cycle().skip(1))
        {
            edges[twin(f) as usize].cycle_next = e;
        }
    }

    // Find the ranges of x-coordinates of each face
    let mut fs = vec![];
    let mut visited = vec![false; 2 * m];
    for mut e in 0..2 * m as u32 {
        if visited[e as usize] {
            continue;
        }

        let mut x_bound = [i32::MAX, i32::MIN];
        let e0 = e;
        loop {
            let x = ps[edges[e as usize].src as usize][0] as i32;
            x_bound = [x_bound[0].min(x), x_bound[1].max(x)];

            visited[e as usize] = true;
            e = edges[e as usize].cycle_next;
            if e == e0 {
                break;
            }
        }
        fs.push(x_bound);
    }

    #[derive(Debug)]
    enum IntervalQueryEvent {
        QueryPrefix(u32),
        AddPoint(i32),
    }

    #[derive(Debug)]
    enum RectQueryEvent {
        QueryRect(i32, i32, u32),
        AddPoint(i32),
    }

    let mut interval_queries = vec![];
    let mut rect_queries = vec![];
    let mut ys = vec![X_BOUND];
    for w in edges.chunks_exact(2) {
        let mut p = ps[w[0].src as usize][0] as i32;
        let mut q = ps[w[1].src as usize][0] as i32;
        if p > q {
            std::mem::swap(&mut p, &mut q);
        }
        interval_queries.push((p, IntervalQueryEvent::AddPoint(1)));
        interval_queries.push((q, IntervalQueryEvent::AddPoint(-1)));
    }

    for &x_bound in &fs {
        ys.push(x_bound[0]);
        ys.push(x_bound[1]);
        rect_queries.push((x_bound[0], RectQueryEvent::AddPoint(x_bound[1])));
    }

    for i in 0..q as u32 {
        let a: i32 = input.value();
        let b: i32 = input.value();
        ys.push(a);
        ys.push(b);
        interval_queries.push((a, IntervalQueryEvent::QueryPrefix(i)));
        interval_queries.push((b, IntervalQueryEvent::QueryPrefix(i)));
        rect_queries.push((a, RectQueryEvent::QueryRect(a, b, i)));
        rect_queries.push((b, RectQueryEvent::QueryRect(b, X_BOUND, i)));
    }
    interval_queries.sort_unstable_by_key(|&(x, _)| x);
    rect_queries.sort_unstable_by_key(|&(x, _)| x);

    // println!("{:?}", interval_queries);
    // println!("{:?}", rect_queries);

    let (_, y_map_inv) = compress_coord(ys);
    let y_bound = y_map_inv.len() as usize;

    let mut ans = vec![0; q];
    {
        let mut counter = 0i64;
        for (_, event) in interval_queries {
            match event {
                IntervalQueryEvent::QueryPrefix(i) => {
                    ans[i as usize] += counter;
                }
                IntervalQueryEvent::AddPoint(delta) => {
                    counter += delta as i64;
                }
            }
        }
    }
    {
        let mut counter = fenwick_tree::FenwickTree::new(y_bound, Additive);
        for (_, event) in rect_queries {
            match event {
                RectQueryEvent::QueryRect(x0, x1, i) => {
                    let x0 = y_map_inv[&x0];
                    let x1 = y_map_inv[&x1];
                    ans[i as usize] -= counter.sum_range(x0 as usize..x1 as usize) as i64 - 1;
                }
                RectQueryEvent::AddPoint(y) => {
                    let y = y_map_inv[&y];
                    counter.add(y as usize, 1);
                }
            }
        }
    }

    for &a in &ans {
        writeln!(output, "{}", a).unwrap();
    }

    // println!("{:?}", fs);
}
