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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let t: u32 = input.value();

    let idx = |i: usize, j: usize| i * m + j;

    let h_bound = 1_000_000_000;
    let hs: Vec<i32> = (0..n * m).map(|_| input.value()).collect();

    let mut edges = vec![];
    for i in 0..n {
        for j in 1..m {
            let u = idx(i, j - 1);
            let v = idx(i, j);
            edges.push(((hs[u] - hs[v]).abs(), u as u32, v as u32));
        }
    }
    for i in 1..n {
        for j in 0..m {
            let u = idx(i - 1, j);
            let v = idx(i, j);
            edges.push(((hs[u] - hs[v]).abs(), u as u32, v as u32));
        }
    }
    edges.sort_unstable();

    let mut queries = vec![];
    for i in 0..n {
        for j in 0..m {
            let x = input.token();
            if x == "1" {
                queries.push((0, h_bound, idx(i, j) as u32));
            }
        }
    }
    queries.sort_unstable_by_key(|&(l, r, _)| l + r >> 1);

    // Parallel binary search
    let mut ans = 0u64;
    loop {
        let mut finished = true;

        let mut conn = dset::DisjointSet::new(n * m);
        let mut edges = edges.iter().cloned().peekable();

        for (l, r, u) in &mut queries {
            if l == r {
                continue;
            }
            finished = false;

            let mid = *l + *r >> 1;
            while let Some((_, u, v)) = edges.next_if(|&(w, _, _)| w <= mid) {
                conn.merge(u as usize, v as usize);
            }

            let (_, size) = conn.find_root_with_size(*u as usize);
            if size < t {
                *l = mid + 1;
            } else {
                *r = mid;
            }

            if *l == *r {
                ans += *l as u64;
            }
        }

        if finished {
            break;
        }
        queries.sort_by_key(|&(l, r, _)| l + r >> 1);
    }

    writeln!(output, "{}", ans).unwrap();
}
