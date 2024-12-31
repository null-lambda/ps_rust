use std::{cmp::Ordering, io::Write};

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

pub mod reroot {
    pub mod invertible {
        // O(n) rerooting dp for trees, with invertible pulling operation.

        pub trait RootData {
            // Constraints: (x, inv) |-> (p |-> pull_from(p, x, inv)) must be a commutative group action. i.e.
            //   { p.pull_from(c1, inv1); p.pull_from(c2, inv2); }
            //   is equivalent to:
            //   { p.pull_from(c2, inv2); p.pull_from(c1, inv1); }
            fn pull_from(&mut self, child: &Self, inv: bool);

            fn finalize(&mut self);
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
        }

        pub fn run<R: RootData + Clone>(
            neighbors: &[Vec<usize>],
            data: &mut [R],
            root_init: usize,
            yield_node_dp: &mut impl FnMut(usize, &R),
        ) {
            fn dfs_init<R: RootData>(neighbors: &[Vec<usize>], data: &mut [R], u: usize, p: usize) {
                for &v in &neighbors[u] {
                    if v == p {
                        continue;
                    }
                    dfs_init(neighbors, data, v, u);
                    let (data_u, data_v) = unsafe { get_two(data, u, v).unwrap_unchecked() };
                    data_u.pull_from(&data_v, false);
                }

                data[u].finalize();
            }

            fn dfs_reroot<R: RootData + Clone>(
                neighbors: &[Vec<usize>],
                data: &mut [R],
                yield_node_dp: &mut impl FnMut(usize, &R),
                u: usize,
                p: usize,
            ) {
                let reroot = |data: &mut [R], u: usize, p: usize| {
                    let (data_u, data_p) = unsafe { get_two(data, u, p).unwrap_unchecked() };
                    data_p.pull_from(data_u, true);
                    data_p.finalize();

                    data_u.pull_from(data_p, false);
                    data_u.finalize();
                };

                yield_node_dp(u, &data[u]);
                for &v in &neighbors[u] {
                    if v == p {
                        continue;
                    }
                    let data_u_old = data[u].clone();
                    let data_v_old = data[v].clone();
                    reroot(data, v, u);
                    dfs_reroot(neighbors, data, yield_node_dp, v, u);
                    data[v] = data_v_old;
                    data[u] = data_u_old;
                }
            }

            dfs_init(neighbors, data, root_init, root_init);
            dfs_reroot(neighbors, data, yield_node_dp, root_init, root_init);
        }
    }
}

#[derive(Clone)]
struct Evenness {
    is_black: bool,
    is_even: bool,
    n_black_rel: i32,
    n_black_rel_child: i32,
    n_unharmonic_child: u32,
}

impl Evenness {
    fn new(is_black: bool) -> Self {
        Self {
            is_black,
            is_even: false,
            n_black_rel: 0,
            n_black_rel_child: 0,
            n_unharmonic_child: 0,
        }
    }
}

impl reroot::invertible::RootData for Evenness {
    fn pull_from(&mut self, child: &Evenness, inv: bool) {
        if !inv {
            self.n_black_rel_child += child.n_black_rel;
            self.n_unharmonic_child += !child.is_even as u32;
        } else {
            self.n_black_rel_child -= child.n_black_rel;
            self.n_unharmonic_child -= !child.is_even as u32;
        }
    }

    fn finalize(&mut self) {
        self.is_even = self.n_unharmonic_child == 0
            && match 0.cmp(&self.n_black_rel_child) {
                Ordering::Less => self.is_black,
                Ordering::Greater => !self.is_black,
                Ordering::Equal => true,
            };
        self.n_black_rel = self.n_black_rel_child + if self.is_black { 1 } else { -1 };
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let is_black = (0..n).map(|_| input.token() == "1");
    let mut data: Vec<_> = is_black.map(Evenness::new).collect();

    let mut neighbors = vec![vec![]; n];
    for _ in 1..n {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        neighbors[u].push(v);
        neighbors[v].push(u);
    }

    let mut is_even = vec![false; n];
    reroot::invertible::run(&neighbors, &mut data, 0, &mut |u, x| {
        is_even[u] = x.is_even;
    });

    let ans: Vec<_> = (0..n).filter(|&u| is_even[u]).collect();
    write!(output, "{}", ans.len()).unwrap();
    if !ans.is_empty() {
        writeln!(output).unwrap();
        for &u in &ans {
            write!(output, "{} ", u + 1).unwrap();
        }
    }
}
