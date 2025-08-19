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

pub mod jagged {
    use std::fmt::Debug;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Compressed sparse row format, for static jagged array
    // Provides good locality for graph traversal
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        pub links: Vec<T>,
        head: Vec<u32>,
    }

    impl<T: Clone> CSR<T> {
        pub fn from_pairs(n: usize, pairs: impl Iterator<Item = (u32, T)> + Clone) -> Self {
            let mut head = vec![0u32; n + 1];

            for (u, _) in pairs.clone() {
                debug_assert!(u < n as u32);
                head[u as usize] += 1;
            }
            for i in 0..n {
                head[i + 1] += head[i];
            }
            let mut data: Vec<_> = (0..head[n]).map(|_| MaybeUninit::uninit()).collect();

            for (u, v) in pairs {
                head[u as usize] -= 1;
                data[head[u as usize] as usize] = MaybeUninit::new(v.clone());
            }

            // Rustc is likely to perform in‑place iteration without new allocation.
            // [https://doc.rust-lang.org/stable/std/iter/trait.FromIterator.html#impl-FromIterator%3CT%3E-for-Vec%3CT%3E]
            let data = data
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();

            CSR { links: data, head }
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
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
            CSR { links: data, head }
        }
    }

    impl<T> CSR<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }

        pub fn edge_range(&self, index: usize) -> std::ops::Range<usize> {
            self.head[index] as usize..self.head[index as usize + 1] as usize
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.links[self.edge_range(index)]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            let es = self.edge_range(index);
            &mut self.links[es]
        }
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }
}

const UNSET: u32 = !0;
fn bipartite_match(n: usize, m: usize, neighbors: &jagged::CSR<u32>) -> [Vec<u32>; 2] {
    // Hopcroft-Karp
    const INF: u32 = u32::MAX / 2;

    let mut assignment = [vec![UNSET; n], vec![UNSET; m]];
    let mut left_level = vec![INF; n];
    let mut queue = std::collections::VecDeque::new();
    loop {
        left_level.fill(INF);
        queue.clear();
        for u in 0..n {
            if assignment[0][u] == UNSET {
                queue.push_back(u as u32);
                left_level[u] = 0;
            }
        }

        while let Some(u) = queue.pop_front() {
            for &v in &neighbors[u as usize] {
                let w = assignment[1][v as usize];
                if w == UNSET || left_level[w as usize] != INF {
                    continue;
                }
                left_level[w as usize] = left_level[u as usize] + 1;
                queue.push_back(w);
            }
        }

        fn dfs(
            u: u32,
            neighbors: &jagged::CSR<u32>,
            assignment: &mut [Vec<u32>; 2],
            left_level: &Vec<u32>,
        ) -> bool {
            for &v in &neighbors[u as usize] {
                let w = assignment[1][v as usize];
                if w == UNSET
                    || left_level[w as usize] == left_level[u as usize] + 1
                        && dfs(w, neighbors, assignment, left_level)
                {
                    assignment[0][u as usize] = v as u32;
                    assignment[1][v as usize] = u;
                    return true;
                }
            }
            false
        }

        let mut found_augmenting_path = false;
        for u in 0..n {
            if assignment[0][u] == UNSET
                && dfs(u as u32, neighbors, &mut assignment, &mut left_level)
            {
                found_augmenting_path = true;
            }
        }

        if !found_augmenting_path {
            break;
        }
    }

    assignment
}

fn sorted2<T: PartialOrd>(mut xs: [T; 2]) -> [T; 2] {
    if xs[0] > xs[1] {
        xs.swap(0, 1);
    }
    xs
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let ss: Vec<u32> = (0..n).map(|_| input.value()).collect();
        let ds: Vec<u32> = (0..n).map(|_| input.value()).collect();

        let mut edges = vec![];
        for i in 0..n {
            input.token();
            for j in i + 1..n {
                let t: u32 = input.value();
                for [i, j] in [[i, j], [j, i]] {
                    if ss[i] + ds[i] + t <= ss[j] {
                        edges.push((i as u32, j as u32));
                    }
                }
            }
        }

        let neighbors = jagged::CSR::from_pairs(n, edges.iter().copied());
        let assignment = bipartite_match(n, n, &neighbors);

        let mut visited = [vec![false; n], vec![false; n]];
        let mut bfs = vec![];
        for u in 0..n {
            if assignment[0][u] == UNSET {
                visited[0][u] = true;
                bfs.push(u as u32);
            }
        }

        let mut timer = 0;
        while let Some(&u) = bfs.get(timer) {
            timer += 1;
            for &v in &neighbors[u as usize] {
                if v == assignment[0][u as usize] {
                    continue;
                }
                visited[1][v as usize] = true;

                let w = assignment[1][v as usize];
                if w == UNSET || visited[0][w as usize] {
                    continue;
                }
                visited[0][w as usize] = true;
                bfs.push(w);
            }
        }

        let mut vcover = vec![];
        for u in 0..n {
            if visited[0][u] && !visited[1][u] {
                vcover.push(u as u32);
            }
        }

        writeln!(output, "{}", vcover.len()).unwrap();
        for a in vcover {
            write!(output, "{} ", a + 1).unwrap();
        }
        writeln!(output).unwrap();
    }
}
