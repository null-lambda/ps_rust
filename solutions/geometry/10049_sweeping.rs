use std::io::Write;

use std::{collections::HashMap, hash::Hash};

use collections::DisjointSet;
use fenwick_tree::Group;
use segtree::MonoidAction;

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
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn with_size(n: usize, group: G) -> Self {
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

pub mod segtree {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type Elem;
        type Action;
        fn id(&self) -> Self::Elem;
        fn combine(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
        fn combine_to_left(&self, lhs: &mut Self::Elem, rhs: &Self::Elem) {
            *lhs = self.combine(lhs, rhs);
        }
        fn combine_to_right(&self, lhs: &Self::Elem, rhs: &mut Self::Elem) {
            *rhs = self.combine(lhs, rhs);
        }
        fn apply(&self, x: &mut Self::Elem, action: &Self::Action);
        fn optimize(&self, _x: &mut Self::Elem) {}
    }

    #[derive(Debug)]
    pub struct SegTree<M>
    where
        M: MonoidAction,
    {
        n: usize,
        sum: Vec<M::Elem>,
        pub monoid: M,
    }

    impl<M: MonoidAction> SegTree<M> {
        pub fn with_size(n: usize, monoid: M) -> Self {
            Self {
                n,
                sum: (0..2 * n).map(|_| monoid.id()).collect(),
                monoid,
            }
        }

        pub fn from_iter<I>(n: usize, iter: I, monoid: M) -> Self
        where
            I: Iterator<Item = M::Elem>,
        {
            let mut sum: Vec<_> = (0..n)
                .map(|_| monoid.id())
                .chain(iter)
                .chain(iter::repeat_with(|| monoid.id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = monoid.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn apply(&mut self, mut idx: usize, value: M::Action) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.monoid.apply(&mut self.sum[idx], &value);
            while idx > 1 {
                idx >>= 1;
                // self.sum[idx] = self
                //     .monoid
                //     .combine(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
                self.monoid.apply(&mut self.sum[idx], &value);
                // self.monoid.optimize(&mut self.sum[idx]);
            }
        }
        pub fn query_range(&mut self, range: Range<usize>) -> M::Elem {
            let Range { mut start, mut end } = range;
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (self.monoid.id(), self.monoid.id());
            while start < end {
                if start & 1 != 0 {
                    self.monoid.optimize(&mut self.sum[start]);
                    self.monoid
                        .combine_to_left(&mut result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    self.monoid.optimize(&mut self.sum[end - 1]);
                    self.monoid
                        .combine_to_right(&self.sum[end - 1], &mut result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.monoid.combine(&result_left, &result_right)
        }
    }
}

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

struct CrossSectionOp {
    dset: DisjointSet,
}

impl MonoidAction for CrossSectionOp {
    type Elem = HashMap<u32, u32>;
    type Action = (u32, i8);
    fn id(&self) -> Self::Elem {
        Default::default()
    }
    fn optimize(&self, x: &mut Self::Elem) {
        if x.is_empty() {
            return;
        }
        let mut x_new = HashMap::new();
        for (&i, &f) in x.iter() {
            x_new
                .entry(self.dset.find_root(i as usize) as u32)
                .and_modify(|e| *e += f)
                .or_insert(f);
        }
        *x = x_new;
    }

    fn combine_to_left(&self, lhs: &mut Self::Elem, rhs: &Self::Elem) {
        for (&i, &f) in rhs.iter() {
            lhs.entry(i).and_modify(|e| *e += f).or_insert(f);
        }
    }

    fn combine_to_right(&self, lhs: &Self::Elem, rhs: &mut Self::Elem) {
        self.combine_to_left(rhs, lhs);
    }

    fn combine(&self, lhs: &Self::Elem, rhs: &Self::Elem) -> Self::Elem {
        let mut res = lhs.clone();
        self.combine_to_left(&mut res, rhs);
        res
    }
    fn apply(&self, x: &mut Self::Elem, action: &Self::Action) {
        let (key, delta_freq) = action;
        match delta_freq {
            1 => {
                x.entry(*key).and_modify(|e| *e += 1).or_insert(1);
            }
            -1 => {
                if !x.contains_key(key) {
                    self.optimize(x);
                }
                if x[&key] == 1 {
                    x.remove(&key);
                } else {
                    x.entry(*key).and_modify(|e| *e -= 1);
                }
            }
            _ => panic!(),
        }
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

static mut N_FIND_ROOT_CALLS: u32 = 0;

mod collections {
    use std::cell::Cell;

    pub struct DisjointSet {
        // represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            -self.parent_or_size[self.find_root(u)].get() as u32
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
        pub fn merge(&mut self, mut u: usize, mut v: usize) -> bool {
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            let size_u = -self.parent_or_size[u].get() as i32;
            let size_v = -self.parent_or_size[v].get() as i32;
            if size_u < size_v {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent_or_size[v].set(u as i32);
            self.parent_or_size[u].set(-(size_u + size_v));
            true
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let w: u32 = input.value();
    let h: u32 = input.value();
    let n: usize = input.value();

    let mut h_segs = vec![(0, w, 0), (0, w, h)];
    let mut v_segs = vec![(0, h, 0), (0, h, w)];
    for _ in 0..n {
        let x1: u32 = input.value();
        let y1: u32 = input.value();
        let x2: u32 = input.value();
        let y2: u32 = input.value();
        if x1 == x2 {
            v_segs.push((y1, y2, x1));
        } else {
            debug_assert_eq!(y1, y2);
            h_segs.push((x1, x2, y1));
        }
    }

    let (_, x_map_inv) = compress_coord(
        (h_segs.iter().flat_map(|(x1, x2, _)| [*x1, *x2])).chain(v_segs.iter().map(|(_, _, x)| *x)),
    );
    let (_, y_map_inv) = compress_coord(
        (v_segs.iter().flat_map(|(y1, y2, _)| [*y1, *y2])).chain(h_segs.iter().map(|(_, _, y)| *y)),
    );

    for (x1, x2, y) in h_segs.iter_mut() {
        *x1 = x_map_inv[&x1];
        *x2 = x_map_inv[&x2];
        *y = y_map_inv[&y];
    }
    for (y1, y2, x) in v_segs.iter_mut() {
        *y1 = y_map_inv[&y1];
        *y2 = y_map_inv[&y2];
        *x = x_map_inv[&x];
    }
    let w = x_map_inv[&w];
    let _h = y_map_inv[&h];

    // 1st Sweep - count intersections
    let mut n_inter: u64 = 0;
    let mut x_count = fenwick_tree::FenwickTree::with_size(w as usize + 1, Additive);

    #[derive(Debug, Clone, Copy)]
    enum EventTag {
        Query(u32, u32),
        Insert(u32, u32),
        Remove(u32, u32),
    }
    let mut events = vec![];
    for &(x1, x2, y) in &h_segs {
        events.push((3 * y + 1, EventTag::Query(x1, x2)));
    }
    for (i, &(y1, y2, x)) in v_segs.iter().enumerate() {
        events.push((3 * y1 + 0, EventTag::Insert(x, i as u32)));
        events.push((3 * y2 + 2, EventTag::Remove(x, i as u32)));
    }
    events.sort_unstable_by_key(|(y, _)| *y);
    for &(_y, tag) in &events {
        match tag {
            EventTag::Query(x1, x2) => {
                n_inter += x_count.sum_range(x1 as usize..x2 as usize + 1) as u64;
            }
            EventTag::Insert(x, _) => {
                x_count.add(x as usize, 1);
            }
            EventTag::Remove(x, _) => {
                x_count.add(x as usize, -1);
            }
        }
    }

    // 2nd Sweep - count number of connected components
    let mut n_components: u32 = 0;
    let dset = DisjointSet::new(v_segs.len() as usize);
    let mut active_components =
        segtree::SegTree::with_size(w as usize + 1, CrossSectionOp { dset });
    for &(_, tag) in &events {
        match tag {
            EventTag::Query(x1, x2) => {
                let freq = active_components.query_range(x1 as usize..x2 as usize + 1);
                let mut indices = freq.keys();
                if freq.len() == 0 {
                    n_components += 1;
                } else {
                    let dset = &mut active_components.monoid.dset;
                    let indices_base = indices.next().unwrap();
                    for &idx in indices {
                        if dset.merge(*indices_base as usize, idx as usize) {
                            n_components -= 1;
                        }
                    }
                }
            }
            EventTag::Insert(x, component_idx) => {
                active_components.apply(x as usize, (component_idx, 1));
                n_components += 1;
            }
            EventTag::Remove(x, component_idx) => {
                let component_idx = active_components
                    .monoid
                    .dset
                    .find_root(component_idx as usize) as u32;
                active_components.apply(x as usize, (component_idx, -1));
            }
        }
    }

    let ans = n_components as u64 + n_inter - h_segs.len() as u64 - v_segs.len() as u64;
    writeln!(output, "{}", ans).unwrap();
}
