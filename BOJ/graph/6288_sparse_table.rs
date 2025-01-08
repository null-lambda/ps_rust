use std::{collections::BTreeMap, io::Write};

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

const UNSET: u32 = u32::MAX;

fn dfs1(children: &[Vec<u32>], smallest_in_subtree: &mut [u32], depth: &mut [u32], u: u32) {
    smallest_in_subtree[u as usize] = u;
    for &v in &children[u as usize] {
        depth[v as usize] = depth[u as usize] + 1;
        dfs1(children, smallest_in_subtree, depth, v);
        smallest_in_subtree[u as usize] =
            smallest_in_subtree[u as usize].min(smallest_in_subtree[v as usize]);
    }
}

fn dfs2(children: &[Vec<u32>], postorder_idx: &mut [u32], timer: &mut u32, u: u32) {
    for &v in &children[u as usize] {
        dfs2(children, postorder_idx, timer, v);
    }
    postorder_idx[u as usize] = *timer;
    *timer += 1;
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut root = UNSET;
    let mut children = vec![vec![]; n];
    let mut parent = vec![UNSET; n];
    for u in 0..n {
        if let Some(p) = input.value::<u32>().checked_sub(1) {
            parent[u] = p;
            children[p as usize].push(u as u32);
        } else {
            root = u as u32;
        }
    }

    // Build sparse table
    let n_log2 = (usize::BITS - usize::leading_zeros(n)) as usize;
    let mut parent_sparse = vec![parent];
    for exp in 1..n_log2 {
        let prev = &parent_sparse[exp - 1];
        let mut row = vec![UNSET; n];
        for u in 0..n {
            if prev[u] != UNSET {
                row[u] = prev[prev[u] as usize];
            }
        }
        parent_sparse.push(row);
    }

    // Reorder children according to min index in subtree
    let mut smallest_in_subtree = vec![UNSET; n];
    let mut depth = vec![0; n];
    dfs1(&children, &mut smallest_in_subtree, &mut depth, root);
    for u in 0..n {
        children[u].sort_unstable_by_key(|&v| smallest_in_subtree[v as usize]);
    }

    let mut postorder_idx = vec![UNSET; n];
    dfs2(&children, &mut postorder_idx, &mut 0, root);

    let mut vacants: BTreeMap<u32, u32> = (0..n).map(|u| (postorder_idx[u], u as u32)).collect();
    for _ in 0..q {
        match input.token() {
            "1" => {
                let k: usize = input.value();
                let mut last = UNSET;
                for _ in 0..k {
                    last = vacants.pop_first().unwrap().1;
                }
                writeln!(output, "{}", last + 1).unwrap();
            }
            "2" => {
                // Ascend to the topmost occupied ancestor
                let mut u = input.value::<usize>() - 1;
                let mut step = 0;
                for exp in (0..n_log2).rev() {
                    let ancestor = parent_sparse[exp][u];
                    if ancestor != UNSET && !vacants.contains_key(&postorder_idx[ancestor as usize])
                    {
                        u = ancestor as usize;
                        step += 1 << exp;
                    }
                }
                vacants.insert(postorder_idx[u], u as u32);
                writeln!(output, "{}", step).unwrap();
            }
            _ => panic!(),
        }
    }
}
