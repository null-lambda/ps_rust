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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

mod collections {
    use std::cell::Cell;

    #[derive(Clone)]
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
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let h: usize = input.value();
    let w: usize = input.value();
    let k: usize = input.value();

    let h_pad = h + 2;
    let w_pad = w + 2;

    let mut current_row = vec![1; w_pad];
    let mut owned_by = vec![usize::MAX; h_pad * w_pad];
    let mut row_acc = vec![DisjointSet::new(h_pad * w_pad); 2];
    let mut col_acc = vec![DisjointSet::new(h_pad * w_pad); 2];
    let mut ldiag_acc = vec![DisjointSet::new(h_pad * w_pad); 2];
    let mut rdiag_acc = vec![DisjointSet::new(h_pad * w_pad); 2];

    for i in 0..h * w {
        let player = i % 2;
        let col = input.value::<usize>();

        let row = current_row[col];
        let u = row * w_pad + col;
        owned_by[u] = player;
        for v in [u - w_pad, u, u + w_pad].iter().copied() {
            if owned_by[v] == player {
                col_acc[player].merge(u, v);
            }
        }
        for v in [u - 1, u + 1].iter().copied() {
            if owned_by[v] == player {
                row_acc[player].merge(u, v);
            }
        }
        for v in [u - w_pad - 1, u + w_pad + 1].iter().copied() {
            if owned_by[v] == player {
                ldiag_acc[player].merge(u, v);
            }
        }
        for v in [u - w_pad + 1, u + w_pad - 1].iter().copied() {
            if owned_by[v] == player {
                rdiag_acc[player].merge(u, v);
            }
        }
        for dset in [
            &row_acc[player],
            &col_acc[player],
            &ldiag_acc[player],
            &rdiag_acc[player],
        ] {
            if dset.get_size(u) >= k as u32 {
                let player_code = if player == 0 { "A" } else { "B" };
                writeln!(output, "{} {}", player_code, i + 1).unwrap();
                return;
            }
        }

        current_row[col] += 1;
    }
    writeln!(output, "D").unwrap();
}
