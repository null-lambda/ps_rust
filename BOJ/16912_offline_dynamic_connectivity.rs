use std::{collections::BTreeMap, io::Write, ops::Range};

use dset::DisjointSet;

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

mod mem_static {
    use std::mem::MaybeUninit;

    pub struct Stack<T, const N: usize> {
        pos: [MaybeUninit<T>; N],
        len: usize,
    }

    impl<T, const N: usize> Stack<T, N> {
        pub fn new() -> Self {
            Self {
                pos: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
            }
        }

        #[must_use]
        pub fn push(&mut self, value: T) -> Option<()> {
            if self.len == N {
                return None;
            }
            unsafe { self.push_unchecked(value) };
            Some(())
        }

        pub unsafe fn push_unchecked(&mut self, value: T) {
            *self.pos.get_unchecked_mut(self.len) = MaybeUninit::new(value);
            self.len += 1;
        }

        pub fn pop(&mut self) -> Option<T> {
            self.len = self.len.checked_sub(1)?;
            Some(unsafe { self.pos.get_unchecked(self.len).assume_init_read() })
        }
    }
}

mod dset {
    use std::mem;

    use crate::mem_static::Stack;

    pub struct DisjointSet {
        parent: Vec<u32>,
        rank: Vec<u32>,
        history: Stack<(u32, u32), 100_000>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent: (0..n).map(|x| x as u32).collect(),
                rank: vec![0; n],
                history: Stack::new(),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if self.parent[u] == u as u32 {
                u
            } else {
                self.find_root(self.parent[u] as usize)
            }
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let mut u = self.find_root(u);
            let mut v = self.find_root(v);
            if u == v {
                return false;
            }

            let rank_u = self.rank[u];
            let rank_v = self.rank[v];
            if rank_u < rank_v {
                mem::swap(&mut u, &mut v);
            }
            let prev_rank_u = self.rank[u];
            self.parent[v] = u as u32;
            if rank_u == rank_v {
                self.rank[u] += 1;
            }
            unsafe { self.history.push_unchecked((v as u32, prev_rank_u as u32)) };
            true
        }

        pub fn rollback(&mut self) -> bool {
            let Some((v, prev_rank_u)) = self.history.pop() else {
                return false;
            };
            let u = self.parent[v as usize] as usize;
            self.rank[u] = prev_rank_u;
            self.parent[v as usize] = v;
            true
        }
    }
}

fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let n = xs.len();
    let mut i = 0;
    for j in 0..n {
        if pred(&xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    xs.split_at_mut(i)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();
    let n: usize = input.value();
    let m: u32 = input.value();

    let mut connected = BTreeMap::new();
    let mut intervals = vec![];
    let mut queries = vec![None; m as usize];
    for time in 0..m {
        let cmd = input.token();
        let mut a = input.value::<u32>() - 1;
        let mut b = input.value::<u32>() - 1;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        match cmd {
            "1" => {
                assert!(connected.insert((a, b), time).is_none());
            }
            "2" => {
                let start = connected.remove(&(a, b)).unwrap();
                intervals.push((start..time + 1, a, b));
            }
            "3" => {
                queries[time as usize] = Some((a, b));
            }
            _ => panic!(),
        }
    }
    for ((a, b), start) in connected {
        intervals.push((start..m, a, b));
    }

    fn dnc(
        dset: &mut DisjointSet,
        intervals: &mut [(Range<u32>, u32, u32)],
        queries: &[Option<(u32, u32)>],
        ans: &mut [bool],
        time_range: Range<u32>,
    ) {
        debug_assert!(time_range.start < time_range.end);
        let (intervals, _) = partition_in_place(intervals, |(interval, _, _)| {
            !(interval.end <= time_range.start || time_range.end <= interval.start)
        });
        let (full, partial) = partition_in_place(intervals, |(interval, _, _)| {
            interval.start <= time_range.start && time_range.end <= interval.end
        });
        let mut full_count = 0;

        for &(_, a, b) in full.iter() {
            full_count += dset.merge(a as usize, b as usize) as u32;
        }

        if time_range.start + 1 == time_range.end {
            let i = time_range.start as usize;
            if let Some((a, b)) = queries[i] {
                ans[i] = dset.find_root(a as usize) == dset.find_root(b as usize);
            }
        } else {
            let mid = (time_range.start + time_range.end) / 2;
            dnc(dset, partial, queries, ans, time_range.start..mid);
            dnc(dset, partial, queries, ans, mid..time_range.end);
        }

        for _ in 0..full_count {
            dset.rollback();
        }
    }

    let mut dset = DisjointSet::new(n);
    let mut ans = vec![false; m as usize];
    dnc(&mut dset, &mut intervals, &queries, &mut ans, 0..m);
    for (q, &a) in queries.iter().zip(ans.iter()) {
        if q.is_some() {
            writeln!(output, "{}", a as u8).unwrap();
        }
    }
}
