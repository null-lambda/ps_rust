use std::{collections::VecDeque, io::Write};

use jagged::Jagged;

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
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> Self::ItemRef;
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> Self::ItemRef {
            let res = <Self as AsRef<[Vec<T>]>>::as_ref(self)[u].iter();
            res
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self.get(i).collect()).collect();
            v.fmt(f)
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
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> Self::ItemRef {
            self.data[self.head[u] as usize..self.head[u + 1] as usize].iter()
        }
    }
}

const UNSET: u32 = u32::MAX;
fn bipartite_match<'a>(n: usize, m: usize, neighbors: &'a impl Jagged<'a, u32>) -> [Vec<u32>; 2] {
    // Hopcroft-Karp
    const INF: u32 = u32::MAX / 2;

    let mut assignment = [vec![UNSET; n], vec![UNSET; m]];
    let mut left_level = vec![INF; n];
    let mut queue = VecDeque::new();
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
            for &v in neighbors.get(u as usize) {
                let w = assignment[1][v as usize];
                if w == UNSET || left_level[w as usize] != INF {
                    continue;
                }
                left_level[w as usize] = left_level[u as usize] + 1;
                queue.push_back(w);
            }
        }

        fn dfs<'a>(
            u: u32,
            neighbors: &'a impl Jagged<'a, u32>,
            assignment: &mut [Vec<u32>; 2],
            left_level: &Vec<u32>,
        ) -> bool {
            for &v in neighbors.get(u as usize) {
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

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges = vec![];
    for u in 0..n {
        for _ in 0..input.value() {
            let v = input.value::<usize>() - 1;
            edges.push((u as u32, v as u32));
        }
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);
    let assignment = bipartite_match(n, m, &neighbors);

    let max_match = assignment[0].iter().filter(|&&u| u != UNSET).count();
    writeln!(output, "{}", max_match).unwrap();
}
