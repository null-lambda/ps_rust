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
    use std::collections::HashSet;
    use std::mem;

    pub struct DisjointSet {
        // represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        colors: Vec<HashSet<u32>>,
    }

    impl DisjointSet {
        pub fn new(colors: impl IntoIterator<Item = u32>) -> Self {
            let colors: Vec<_> = colors.into_iter().map(|c| [c].into()).collect();
            let n = colors.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                colors,
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            -self.parent_or_size[self.find_root(u)].get() as u32
        }

        pub fn n_colors(&self, u: usize) -> usize {
            self.colors[self.find_root(u)].len()
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
                mem::swap(&mut u, &mut v);
            }
            self.parent_or_size[v].set(u as i32);
            self.parent_or_size[u].set(-(size_u + size_v));

            if self.colors[u].len() < self.colors[v].len() {
                self.colors.swap(u, v);
            }
            let tmp = mem::take(&mut self.colors[v]);
            self.colors[u].extend(tmp);
            true
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut parents: Vec<usize> = (0..n).collect();
    let mut active = vec![true; n];
    active[0] = false;

    for i in 1..n {
        parents[i] = input.value::<usize>() - 1;
    }
    let colors = (0..n).map(|_| input.value::<u32>());

    let mut dset = DisjointSet::new(colors);

    let queries: Vec<(u8, usize)> = (0..n - 1 + q)
        .map(|_| {
            let cmd = input.value();
            let u = input.value::<usize>() - 1;
            (cmd, u)
        })
        .collect();

    for &(cmd, u) in &queries {
        if cmd == 1 {
            active[u] = false;
        }
    }
    for i in 0..n {
        if active[i] {
            dset.merge(i, parents[i]);
        }
    }

    let mut ans = vec![];
    for &(cmd, u) in queries.iter().rev() {
        match cmd {
            1 => {
                dset.merge(u, parents[u]);
            }
            2 => {
                ans.push(dset.n_colors(u));
            }
            _ => panic!(),
        }
    }

    for x in ans.into_iter().rev() {
        writeln!(output, "{}", x).unwrap();
    }
}
