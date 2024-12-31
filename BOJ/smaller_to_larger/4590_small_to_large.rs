use std::{
    collections::{BTreeMap, HashSet},
    io::Write,
    iter, usize,
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

struct BTreeMultiSet<T> {
    values: BTreeMap<T, u32>,
}

impl<T: Ord> BTreeMultiSet<T> {
    fn len_unique(&self) -> usize {
        self.values.len()
    }

    fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let mut values = BTreeMap::new();
        for value in iter {
            *values.entry(value).or_default() += 1;
        }
        Self { values }
    }

    fn insert(&mut self, value: T) {
        *self.values.entry(value).or_default() += 1;
    }

    fn remove(&mut self, value: T) {
        let count = self.values.get_mut(&value).unwrap();
        *count -= 1;
        if *count == 0 {
            self.values.remove(&value);
        }
    }
}

fn dfs_tree_lazy<'a>(
    neighbors: &'a [HashSet<usize>],
    node: usize,
    parent: usize,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let mut stack = vec![(node, parent, neighbors[node].iter())];
    iter::once((node, parent)).chain(
        iter::from_fn(move || {
            stack.pop().map(|(node, parent, mut iter_child)| {
                let child = *iter_child.next()?;
                stack.push((node, parent, iter_child));
                if child == parent {
                    return None;
                }
                stack.push((child, node, neighbors[child].iter()));
                Some((child, node))
            })
        })
        .flatten(),
    )
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let root = 0;
    let mut parent = vec![0; n];
    let mut neighbors = vec![HashSet::new(); n];
    for i in 1..n {
        let p = input.value::<usize>() - 1;
        parent[i] = p;
        neighbors[p].insert(i);
        neighbors[i].insert(p);
    }
    let mut colors = vec![0; n];
    for i in 0..n {
        colors[i] = input.value();
    }

    let mut n_component = 1;
    let mut component_indices = vec![0; n];
    let mut components: Vec<_> = vec![BTreeMultiSet::from_iter(colors.iter().copied())];

    let mut prev_ans = 0;
    for _ in 0..n + q - 1 {
        let cmd = input.token();
        let k = input.value::<usize>();
        let a = (k ^ prev_ans) - 1;
        match cmd {
            "1" => {
                assert!(a != root);
                let p = parent[a];
                assert!(component_indices[a] == component_indices[p]);

                let mut left = dfs_tree_lazy(&neighbors, a, p);
                let mut right = dfs_tree_lazy(&neighbors, p, a);
                let (small, large) = loop {
                    match (left.next(), right.next()) {
                        (Some(..), Some(..)) => continue,
                        (None, Some(_)) => break (a, p),
                        _ => break (p, a),
                    }
                };
                drop((left, right));

                let mut c_small = BTreeMultiSet::from_iter(None);
                let c_large = &mut components[component_indices[large]];

                for (u, _) in dfs_tree_lazy(&neighbors, small, large) {
                    component_indices[u] = n_component;
                    c_small.insert(colors[u]);
                    c_large.remove(colors[u]);
                }
                neighbors[small].remove(&large);
                neighbors[large].remove(&small);

                components.push(c_small);
                n_component += 1;
            }
            "2" => {
                let ans = components[component_indices[a]].len_unique();
                writeln!(output, "{}", ans).unwrap();
                prev_ans = ans;
            }
            _ => panic!(),
        }
    }
}
