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

type View<'a, const N: usize> = [&'a [u8]; N];
type View5x5<'a> = View<'a, 5>;
type View3x3<'a> = View<'a, 3>;

const WILDCARD: u8 = b'*';
const FRAME: View5x5 = [b"+---+", b"|***|", b"|***|", b"|***|", b"+---+"];

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let h: usize = input.value();
    let w: usize = input.value();

    let h_pad = h + 5;
    let w_pad = w + 5;

    let mut grid = vec![b'.'; h_pad * w_pad];
    for i in 0..h {
        grid[i * w_pad..][..w].copy_from_slice(input.token().as_bytes());
    }

    let view5x5 = |u: usize| -> View5x5 {
        let (i, j) = (u / w_pad, u % w_pad);
        std::array::from_fn::<_, 5, _>(|i_shift| &grid[(i + i_shift) * w_pad..][j..][..5])
    };
    let view3x3 = |u: usize| -> View3x3 {
        let (i, j) = (u / w_pad, u % w_pad);
        std::array::from_fn::<_, 3, _>(|i_shift| &grid[(i + i_shift) * w_pad..][j..][..3])
    };

    let eq_pattern = |sample: View5x5, pattern: View5x5| -> bool {
        (0..5).all(|i| {
            sample[i]
                .iter()
                .zip(pattern[i])
                .all(|(&x, &p)| p == WILDCARD || x == p)
        })
    };

    let mut eyes: Vec<Option<View3x3>> = vec![None; h_pad * w_pad];
    for i in 0..h {
        for j in 0..w {
            let u = i * w_pad + j;
            let cell = view5x5(u);
            if eq_pattern(cell, FRAME) {
                eyes[u] = Some(view3x3(u + w_pad + 1));
            }
        }
    }

    let mut conn = dset::DisjointSet::new(h_pad * w_pad);
    for i in 0..h {
        for j in 0..w {
            let u = i * w_pad + j;
            if eyes[u].is_some() && eyes[u + 4].is_some() {
                conn.merge(u, u + 4);
            }
            if eyes[u].is_some() && eyes[u + w_pad * 4].is_some() {
                conn.merge(u, u + w_pad * 4);
            }
        }
    }

    let mut pieces = vec![vec![]; h_pad * w_pad];
    for i in 0..h {
        for j in 0..w {
            let u = i * w_pad + j;
            if eyes[u].is_some() {
                pieces[conn.find_root(u)].push((u, eyes[u].take()));
            }
        }
    }

    pieces.retain(|x| !x.is_empty());
    println!("{:?}", pieces.len());
    println!("{:?}", pieces);
}
