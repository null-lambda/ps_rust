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
    let mut dset = collections::DisjointSet::new(n);
    let mut xs: Vec<usize> = (0..n).map(|_| input.value()).collect();
    for i in 0..n {
        dset.merge(i, xs[i]);
    }

    let n_components = (0..n).filter(|&i| dset.find_root(i) == i).count();
    if n_components >= 2 {
        let mut pivot = 0;
        let root = dset.find_root(pivot);
        let next = (0..n)
            .filter(|&i| dset.find_root(i) != root)
            .min_by_key(|&i| xs[i])
            .unwrap();
        if xs[pivot] < xs[next] {
            let mut u = pivot;
            loop {
                if xs[u] > xs[next] {
                    pivot = u;
                    break;
                }

                let prev = u;
                u = xs[u];
                if u == pivot {
                    pivot = prev;
                    break;
                }
            }
        }

        for i in 0..n_components - 1 {
            let root = dset.find_root(pivot);
            let next = (0..n)
                .filter(|&i| dset.find_root(i) != root)
                .min_by_key(|&i| xs[i])
                .unwrap();
            xs.swap(pivot, next);
            dset.merge(pivot, next);
            pivot = next;
        }
    }

    for x in xs {
        write!(output, "{} ", x).unwrap();
    }
}
