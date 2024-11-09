use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

use collections::Jagged;

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

pub mod collections {
    use std::fmt::Debug;
    use std::ops::Index;

    // compressed sparse row format for jagged array
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Jagged<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for Jagged<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for Jagged<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            Jagged { data, head }
        }
    }

    impl<T> Jagged<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }
    }

    impl<T> Index<usize> for Jagged<T> {
        type Output = [T];
        fn index(&self, index: usize) -> &[T] {
            let start = self.head[index] as usize;
            let end = self.head[index + 1] as usize;
            &self.data[start..end]
        }
    }

    impl<T> Jagged<T> {
        pub fn iter(&self) -> Iter<T> {
            Iter { src: self, pos: 0 }
        }
    }

    impl<'a, T> IntoIterator for &'a Jagged<T> {
        type Item = &'a [T];
        type IntoIter = Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }

    pub struct Iter<'a, T> {
        src: &'a Jagged<T>,
        pos: usize,
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = &'a [T];
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < self.src.len() {
                let item = &self.src[self.pos];
                self.pos += 1;
                Some(item)
            } else {
                None
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        neighbors[u].push(v);
        neighbors[v].push(u);
    }
    let neighbors: Jagged<_> = neighbors.into_iter().collect();

    let mut size = vec![0; n];
    fn dfs_size(neighbors: &Jagged<usize>, size: &mut [u32], u: usize, parent: usize) {
        size[u] = 1;
        for &v in &neighbors[u] {
            if v != parent {
                dfs_size(neighbors, size, v, u);
                size[u] += size[v];
            }
        }
    }

    fn get_centroid(neighbors: &Jagged<usize>, size: &[u32], u: usize, parent: usize) -> usize {
        let n = neighbors.len() as u32;
        for &v in &neighbors[u] {
            if v != parent && size[v] > n / 2 {
                return get_centroid(neighbors, size, v, u);
            }
        }
        u
    }

    dfs_size(&neighbors, &mut size, 0, n);
    let centroid = get_centroid(&neighbors, &size, 0, n);

    fn extract_code(neighbors: &Jagged<usize>, u: usize, parent: usize, res: &mut Vec<u32>) {
        res.push(neighbors[u].len() as u32);
        if let Some(&v) = neighbors[u].iter().filter(|&&v| v != parent).next() {
            extract_code(neighbors, v, u, res);
        }
    }

    fn test_code(neighbors: &Jagged<usize>, u: usize, parent: usize, code: &[u32]) -> bool {
        // all branch has the same code
        let (c, rest) = code.split_first().unwrap();
        neighbors[u].len() == *c as usize
            && neighbors[u]
                .iter()
                .filter(|&&v| v != parent)
                .all(|&v| test_code(neighbors, v, u, rest))
    }

    fn find_any_leaf(neighbors: &Jagged<usize>, u: usize, parent: usize) -> usize {
        if neighbors[u].len() == 1 {
            u
        } else {
            find_any_leaf(
                neighbors,
                *neighbors[u]
                    .iter()
                    .filter(|&&v| v != parent)
                    .next()
                    .unwrap(),
                u,
            )
        }
    }

    fn is_code_linear(code: &[u32]) -> bool {
        // 1, ..., 1, 0
        debug_assert!(!code.is_empty());
        let (tail, rest) = code.split_last().unwrap();
        rest.iter().all(|&x| x == 2) && *tail == 1
    }

    let ans = || {
        let mut child_codes: HashMap<_, Vec<u32>> = Default::default();
        for &v in &neighbors[centroid] {
            let mut code = vec![];
            extract_code(&neighbors, v, centroid, &mut code);
            if !test_code(&neighbors, v, centroid, &code) {
                return None;
            }
            child_codes.entry(code).or_default().push(v as u32);
        }

        if child_codes.len() <= 1 {
            return Some(centroid);
        }

        if child_codes.len() == 2 {
            // 1 linear code + n-1 duplicates of same code
            for (code, count) in child_codes {
                if is_code_linear(&code) {
                    if count.len() != 1 {
                        continue;
                    }
                    let leaf = find_any_leaf(&neighbors, count[0] as usize, centroid);
                    let mut leaf_code = vec![];
                    extract_code(&neighbors, leaf, centroid, &mut leaf_code);
                    if test_code(&neighbors, leaf, centroid, &leaf_code) {
                        return Some(leaf);
                    }
                }
            }
        }
        None
    };
    if let Some(ans) = ans() {
        writeln!(output, "{}", ans + 1).unwrap();
    } else {
        writeln!(output, "-1").unwrap();
    }
}
