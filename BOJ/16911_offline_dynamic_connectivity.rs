use std::{collections::HashMap, io::Write, ops::Range};

use dset::DisjointSet;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

mod mem_reserved {
    use std::mem::MaybeUninit;

    pub struct Stack<T> {
        pos: Box<[MaybeUninit<T>]>,
        len: usize,
    }

    impl<T> Stack<T> {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                pos: (0..capacity).map(|_| MaybeUninit::uninit()).collect(),
                len: 0,
            }
        }

        #[must_use]
        pub fn push(&mut self, value: T) -> bool {
            if self.len == self.pos.len() {
                return false;
            }
            unsafe { self.push_unchecked(value) };
            return true;
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

    use crate::mem_reserved::Stack;

    pub struct DisjointSet {
        parent: Vec<u32>,
        rank: Vec<u32>,
        history: Stack<(u32, u32)>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent: (0..n).map(|x| x as u32).collect(),
                rank: vec![0; n],
                history: Stack::with_capacity(n - 1),
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
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();
    let n: usize = input.value();
    let m: u32 = input.value();

    let mut active = HashMap::new();
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
                active.insert((a, b), time);
            }
            "2" => {
                let start = active.remove(&(a, b)).unwrap();
                intervals.push((start..time + 1, a, b));
            }
            "3" => {
                queries[time as usize] = Some((a, b));
            }
            _ => panic!(),
        }
    }
    for ((a, b), start) in active {
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
