use std::io::Write;

use segtree::{Monoid, SegTree};

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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    // BOJ 18378
    // Approach:
    // 1. Find a DFS spanning tree. Identify all back-edges and thus cycles.
    // 2. Build a tree of cycles and apply methods of BOJ 13516, 'Trees and Queries 7'.
    //    Manage a cycle DP with a segment tree. (do a linear DP on the cycle minus back-edge.
    //    and impose cyclic constraints by erasing diagonal values of the interval matrix.)
    // In total, there are three levels of indirection:
    // 1. Min-plus segtree for in-cycle DP, where each element is a node weight combined with light edge aggregates.
    // 2. Min-plus segtree for in-chain DP, where each element is a cycle DP.
    // 3. Min segtree for light edges, where each element is a collapsed chain DP.
    //
    // TODO: Obtain a well-tested subroutine for BCCs and cactus. (no more debugging hell)
}
