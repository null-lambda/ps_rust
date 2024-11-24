use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, VecDeque},
    io::Write,
    iter,
};

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
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: u32 = input.value();

    let mut freq = vec![0u32; m as usize];
    for _ in 0..n {
        let a: u32 = input.value();
        freq[a as usize] += 1;
    }

    let f = |i: u32| i * 2 % m;
    let mut parent = vec![vec![]; m as usize];
    let mut indegree = vec![0; m as usize];
    for i in 0..m {
        parent[f(i) as usize].push(i);
        indegree[f(i) as usize] += 1;
    }

    // Toposort & cycle detection & dp propagation on branches
    let mut freq_acc = freq.clone();
    let mut cost_acc = vec![0u64; m as usize];
    let mut queue: VecDeque<_> = (0..m).filter(|&i| indegree[i as usize] == 0).collect();
    while let Some(u) = queue.pop_front() {
        let v = f(u);
        freq_acc[v as usize] += freq_acc[u as usize];
        cost_acc[v as usize] += cost_acc[u as usize] + freq_acc[u as usize] as u64;

        indegree[v as usize] -= 1;
        if indegree[v as usize] == 0 {
            queue.push_back(v);
        }
    }

    let in_cycle = |u: u32| indegree[u as usize] != 0;

    let mut dset = collections::DisjointSet::new(m as usize);
    for i in 0..m {
        dset.merge(i as usize, f(i) as usize);
    }

    let mut components: HashMap<u32, (Vec<_>, u32)> = HashMap::new();
    for i in 0..m {
        let root = dset.find_root(i as usize);
        let (component, size) = components.entry(root as u32).or_default();
        component.push(i);
        *size += freq[i as usize];
    }

    let max_size = *components.values().map(|(_, s)| s).max().unwrap();
    let mut min_cost = u64::MAX;

    for (_, (nodes, total_freq)) in &components {
        if *total_freq != max_size {
            continue;
        }

        let n_branches = nodes
            .iter()
            .filter(|&&i| in_cycle(i) && freq_acc[i as usize] > 0)
            .count();
        if n_branches == 0 {
            panic!()
        } else if n_branches == 1 {
            min_cost = min_cost.min(
                nodes
                    .iter()
                    .filter(|&&i| freq_acc[i as usize] == *total_freq as u32)
                    .map(|&i| cost_acc[i as usize])
                    .min()
                    .unwrap(),
            );
        } else {
            let base_cost = nodes
                .iter()
                .filter(|&&i| in_cycle(i))
                .map(|&i| cost_acc[i as usize])
                .sum::<u64>();

            let mut freq_on_cycle: Vec<u64> = vec![];

            let start = *nodes.iter().find(|&&i| in_cycle(i)).unwrap();
            let mut u = start;
            loop {
                freq_on_cycle.push(freq_acc[u as usize] as u64);

                u = f(u);
                if u == start {
                    break;
                }
            }
            freq_on_cycle.reverse();
            let n_cycle = freq_on_cycle.len();

            // // O(n_cycle^2) solution
            // let mut total_cost = u64::MAX;
            // for shift in 0..n_cycle as u64 {
            //     total_cost = total_cost.min(
            //         (shift..n_cycle as u64)
            //             .chain(0..shift)
            //             .zip(freq_on_cycle.iter())
            //             .map(|(z, &f)| z * f as u64)
            //             .sum(),
            //     );
            // }

            // O(n_cycle) solution
            let sum_cost = freq_on_cycle.iter().map(|&f| f as u64).sum::<u64>();
            let mut acc = (0..n_cycle as u64)
                .zip(freq_on_cycle.iter())
                .map(|(z, &f)| z * f as u64)
                .sum::<u64>();

            let mut total_cost = acc;
            for shift in 1..n_cycle {
                acc = acc + freq_on_cycle[shift - 1] as u64 * n_cycle as u64 - sum_cost;
                total_cost = total_cost.min(acc);
            }

            min_cost = min_cost.min(total_cost + base_cost);
        }
    }

    write!(output, "{} {}", max_size, min_cost).unwrap();
}
