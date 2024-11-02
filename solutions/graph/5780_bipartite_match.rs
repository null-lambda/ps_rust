use collections::Grid;
use iter::product;
use std::collections::{HashMap, HashSet, VecDeque};

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        let iter = buf.split_ascii_whitespace();
        InputAtOnce(iter)
    }
}

#[allow(dead_code)]
pub mod iter {
    pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)>
    where
        I: IntoIterator,
        I::Item: Clone,
        J: IntoIterator,
        J::IntoIter: Clone,
    {
        let j = j.into_iter();
        i.into_iter()
            .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
    }
}

#[allow(dead_code)]
mod collections {
    use std::{
        cmp::Reverse,
        collections::HashMap,
        fmt::Display,
        iter::{empty, once},
        ops::{Index, IndexMut},
    };

    #[derive(Debug, Clone)]
    pub struct Grid<T> {
        pub w: usize,
        pub data: Vec<T>,
    }

    impl<T> Grid<T> {
        pub fn with_shape(self, w: usize) -> Self {
            debug_assert_eq!(self.data.len() % w, 0);
            Grid { w, data: self.data }
        }
    }

    impl<T> FromIterator<T> for Grid<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self {
                w: 1,
                data: iter.into_iter().collect(),
            }
        }
    }

    impl<T: Clone> Grid<T> {
        pub fn sized(fill: T, h: usize, w: usize) -> Self {
            Grid {
                w,
                data: vec![fill; w * h],
            }
        }
    }

    impl<T: Display> Display for Grid<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for row in self.data.chunks(self.w) {
                for cell in row {
                    cell.fmt(f)?;
                    write!(f, " ")?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
            Ok(())
        }
    }

    impl<T> Index<(usize, usize)> for Grid<T> {
        type Output = T;
        fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
            debug_assert!(i < self.data.len() / self.w && j < self.w);
            &self.data[i * self.w + j]
        }
    }

    impl<T> IndexMut<(usize, usize)> for Grid<T> {
        fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
            debug_assert!(i < self.data.len() / self.w && j < self.w);
            &mut self.data[i * self.w + j]
        }
    }

    pub struct PrettyColored<'a>(&'a Grid<u8>);

    impl Display for PrettyColored<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let colors = (once(37).chain(31..=36))
                .map(|i| (format!("\x1b[{}m", i), format!("\x1b[0m")))
                .collect::<Vec<_>>();

            let mut freq = HashMap::new();
            for c in self.0.data.iter() {
                *freq.entry(c).or_insert(0) += 1;
            }
            let mut freq = freq.into_iter().collect::<Vec<_>>();
            freq.sort_unstable_by_key(|(_, f)| Reverse(*f));

            let mut color_map = HashMap::new();
            let mut idx = 0;
            for (c, _) in freq {
                color_map.insert(c, &colors[idx % colors.len()]);
                idx += 1;
            }

            for row in self.0.data.chunks(self.0.w) {
                for cell in row {
                    let (pre, suff) = color_map[&cell];
                    write!(f, "{}{}{}", pre, *cell as char, suff)?;
                }
                writeln!(f)?;
            }
            Ok(())
        }
    }

    impl Grid<u8> {
        pub fn colored(&self) -> PrettyColored {
            PrettyColored(&self)
        }
    }

    pub fn neighbors_eswn(
        u: (usize, usize),
        h: usize,
        w: usize,
    ) -> impl Iterator<Item = (usize, usize)> {
        let (i, j) = u;
        empty()
            .chain((j + 1 < w).then(|| (i, j + 1)))
            .chain((i + 1 < h).then(|| (i + 1, j)))
            .chain((j > 0).then(|| (i, j - 1)))
            .chain((i > 0).then(|| (i - 1, j)))
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();

    loop {
        let h: usize = input.value();
        let w: usize = input.value();
        if h == 0 && w == 0 {
            break;
        }

        let k: usize = input.value();
        let mut grid = Grid::sized(true, h, w);

        for _ in 0..k {
            let r: usize = input.value();
            let c: usize = input.value();
            grid[(r - 1, c - 1)] = false;
        }

        let mut verts_map = HashMap::new();
        for u in product(0..h, 0..w) {
            if grid[u] {
                let idx = verts_map.len();
                verts_map.insert(u, idx);
            }
        }

        let n = verts_map.len();
        let edges = product(0..h, 1..w)
            .flat_map(|u| {
                let left = (u.0, u.1 - 1);
                (grid[left] && grid[u]).then(|| (left, u))
            })
            .chain(product(1..h, 0..w).flat_map(|u| {
                let up = (u.0 - 1, u.1);
                (grid[up] && grid[u]).then(|| (up, u))
            }));

        let mut neighbors = vec![vec![]; n];
        for (mut u, mut v) in edges {
            if (u.0 + u.1) % 2 == 1 {
                std::mem::swap(&mut u, &mut v);
            }
            let u = verts_map[&u];
            let v = verts_map[&v];
            neighbors[u].push(v);
        }

        // bipartite match
        const UNDEFINED: usize = usize::MAX;
        struct DfsState {
            visited: Vec<bool>,
            src: Vec<usize>,
        }

        fn dfs(node: usize, neighbors: &Vec<Vec<usize>>, state: &mut DfsState) -> bool {
            neighbors[node].iter().any(|&target| {
                if state.visited[target] {
                    return false;
                }
                state.visited[target] = true;
                if state.src[target] == UNDEFINED || dfs(state.src[target], neighbors, state) {
                    state.src[target] = node;
                    return true;
                }
                false
            })
        }

        let mut state = DfsState {
            visited: vec![false; n],
            src: vec![UNDEFINED; n],
        };
        let result = (0..n)
            .filter(|&u| {
                state.visited.fill(false);
                dfs(u, &neighbors, &mut state)
            })
            .count();
        println!("{:?}", result);
    }
}
