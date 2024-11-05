use std::{io::Write, iter, ops::Range};

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

pub mod fenwick_tree {
    pub trait Group {
        type Elem: Clone;
        fn id(&self) -> Self::Elem;
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
    }

    pub struct FenwickTree<G: Group> {
        n: usize,
        n_ceil: usize,
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n_ceil = n.next_power_of_two();
            let data = (0..n_ceil).map(|_| group.id()).collect();
            Self {
                n,
                n_ceil,
                group,
                data,
            }
        }

        pub fn add(&mut self, mut idx: usize, value: G::Elem) {
            while idx < self.n {
                self.group.add_assign(&mut self.data[idx], value.clone());
                idx |= idx + 1;
            }
        }
        pub fn get(&self, idx: usize) -> G::Elem {
            self.sum_range(idx..idx + 1)
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::Elem {
            let mut res = self.group.id();
            let mut r = range.end;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            let mut l = range.start;
            while l > 0 {
                self.group.sub_assign(&mut res, self.data[l - 1].clone());
                l &= l - 1;
            }

            res
        }
    }
}

struct AddGroup;

impl fenwick_tree::Group for AddGroup {
    type Elem = i32;
    fn id(&self) -> Self::Elem {
        0
    }
    fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs += rhs;
    }
    fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
        *lhs -= rhs;
    }
}

#[derive(Debug, Default, Clone)]
struct Node {
    idx_euler: u32,
    children: Vec<u32>,
    interval: Range<u32>,
    height: u32,
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let root = input.value::<usize>() - 1;

    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        neighbors[u as usize].push(v);
        neighbors[v as usize].push(u);
    }

    struct AddGroup;

    impl fenwick_tree::Group for AddGroup {
        type Elem = i32;
        fn id(&self) -> Self::Elem {
            0
        }
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
            *lhs += rhs;
        }
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem) {
            *lhs -= rhs;
        }
    }

    let mut visited = vec![false; n];
    let mut nodes = vec![Node::default(); n];

    // build tree
    fn dfs(
        neighbors: &[Vec<u32>],
        nodes: &mut [Node],
        visited: &mut [bool],
        u: u32,
        idx_euler: &mut u32,
    ) {
        visited[u as usize] = true;
        nodes[u as usize].idx_euler = *idx_euler;
        let (mut left, mut right) = (*idx_euler, *idx_euler + 1);
        *idx_euler += 1;

        for &v in &neighbors[u as usize] {
            if visited[v as usize] {
                continue;
            }
            nodes[u as usize].children.push(v);
            nodes[v as usize].height = nodes[u as usize].height + 1;
            dfs(neighbors, nodes, visited, v, idx_euler);

            left = left.min(nodes[v as usize].interval.start);
            right = right.max(nodes[v as usize].interval.end);
        }
        nodes[u as usize].interval = left..right;
    }
    nodes[root].height = 1;
    dfs(&neighbors, &mut nodes, &mut visited, root as u32, &mut 0);

    let mut tree = fenwick_tree::FenwickTree::new(n, AddGroup);

    let q: usize = input.value();
    for _ in 0..q {
        let cmd = input.token();
        let u = input.value::<usize>() - 1;
        match cmd {
            "1" => tree.add(nodes[u].idx_euler as usize, 1),
            "2" => {
                let interval = nodes[u].interval.start as usize..nodes[u].interval.end as usize;
                let ans = tree.sum_range(interval) as u64 * nodes[u].height as u64;
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
