use std::io::Write;

use collections::DisjointSet;

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
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let edges: Vec<(u32, u32)> = (0..m).map(|_| (input.value(), input.value())).collect();

    let src = 0;
    let dst = n - 1;
    let mut dset = DisjointSet::new(n);
    let mut min_cut = vec![false; m];
    for (i, &(u, v)) in edges.iter().enumerate().rev() {
        let r_src = dset.find_root(src);
        let r_dst = dset.find_root(dst);
        let r_u = dset.find_root(u as usize);
        let r_v = dset.find_root(v as usize);
        if (r_u, r_v) == (r_src, r_dst) || (r_u, r_v) == (r_dst, r_src) {
            min_cut[i] = true;
        } else {
            dset.merge(u as usize, v as usize);
        }
    }

    let p: u64 = 1_000_000_007;
    let mut pow3 = 1u64;
    let mut ans = 0u64;
    for i in 0..m {
        if min_cut[i] {
            ans = (ans + pow3) % p;
        }
        pow3 = pow3 * 3 % p;
    }
    writeln!(output, "{}", ans).unwrap();
}
