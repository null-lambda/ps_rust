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

mod dset {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

fn partition_point<P>(mut left: i64, mut right: i64, mut pred: P) -> i64
where
    P: FnMut(i64) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();
    let w: u32 = input.value();

    let mut blazed = vec![false; n];
    for _ in 0..k {
        let u = input.value::<usize>() - 1;
        blazed[u] = true;
    }

    let scale = 2;
    let (mut es1, mut es0) = (0..m)
        .map(|_| {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            let w: i64 = input.value();
            (u, v, w * scale)
        })
        .partition::<Vec<_>, _>(|&(u, v, _)| blazed[u as usize] ^ blazed[v as usize]);
    es0.sort_unstable_by_key(|&(.., w)| w);
    es1.sort_unstable_by_key(|&(.., w)| w);

    let get_mst: _ = |slope: i64| {
        let es0 = es0.iter().map(|&(u, v, w)| (u, v, w, 0));
        let mut es1 = es1.iter().map(|&(u, v, w)| (u, v, w - slope, 1)).peekable();

        let mut mst_len = 0;
        let mut n_bridge = 0;
        let mut conn = dset::DisjointSet::new(n);
        let mut update = |(u, v, w, is_bridge)| {
            if conn.merge(u as usize, v as usize) {
                mst_len += w as i64;
                n_bridge += is_bridge;
            }
        };
        let key = |(_, _, w, _)| w;
        for x in es0 {
            while let Some(y) = es1.next_if(|&y| key(y) < key(x)) {
                update(y);
            }
            update(x);
        }
        for y in es1 {
            update(y);
        }

        let success = conn.find_root_with_size(0).1 == n as u32;
        (mst_len, n_bridge, success)
    };

    let slope_bound = 1e5 as i64 + 2;
    let opt = partition_point(-slope_bound, slope_bound, |slope| {
        get_mst(slope * scale + 1).1 <= w
    });
    let (_, w_lower, _) = get_mst(opt * scale - 1);
    let (mst_len_upper, w_upper, success) = get_mst(opt * scale + 1);

    if !(w_lower <= w && w <= w_upper && success) {
        writeln!(output, "-1").unwrap();
        return;
    }

    let mut ans = mst_len_upper + (opt * scale + 1) * w_upper as i64;
    ans += opt * scale * (w as i64 - w_upper as i64);
    ans /= scale;
    writeln!(output, "{}", ans).unwrap();
}
