use std::{
    collections::{HashMap, VecDeque},
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

pub mod collections {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;
    use std::ops::Index;

    // Compressed sparse row format for jagged array
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

    impl<T: Clone> Jagged<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
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

const UNSET: u32 = u32::MAX;
fn bipartite_match(n: usize, m: usize, neighbors: &Jagged<u32>) -> [Vec<u32>; 2] {
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
            neighbors: &Jagged<u32>,
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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();

    let mut edges = vec![];
    let mut words = HashMap::<&[u8], usize>::new();
    for i in 0..n {
        let s = input.token().as_bytes();
        let mut s_swapped = s.to_vec();
        for p in 0..s.len() {
            for q in p + 1..s.len() {
                s_swapped.swap(p, q);
                if let Some(&j) = words.get(&s_swapped[..]) {
                    edges.push((j, i));
                }
                s_swapped.swap(p, q);
            }
        }

        words.insert(s, i);
    }

    edges.sort_unstable();
    edges.dedup();

    let mut neighbors = vec![vec![]; n];
    for &(u, v) in &edges {
        neighbors[u].push(v as u32);
        neighbors[v].push(u as u32);
    }

    let mut parity = vec![None; n];
    for u in 0..n {
        if parity[u].is_some() {
            continue;
        }
        parity[u] = Some(false);
        let mut stack = vec![(u, false)];
        while let Some((u, p)) = stack.pop() {
            for &v in &neighbors[u] {
                if parity[v as usize].is_none() {
                    parity[v as usize] = Some(!p);
                    stack.push((v as usize, !p));
                }
            }
        }
    }

    let n_left = parity.iter().filter(|&&p| p == Some(false)).count();
    let n_right = n - n_left;
    let mut left_index_map = HashMap::<usize, usize>::new();
    let mut right_index_map = HashMap::<usize, usize>::new();
    for u in 0..n {
        if parity[u] == Some(false) {
            let idx = left_index_map.len();
            left_index_map.insert(u, idx);
        } else if parity[u] == Some(true) {
            let idx = right_index_map.len();
            right_index_map.insert(u, idx);
        } else {
            panic!()
        }
    }

    let mut neighbors = vec![vec![]; n_left];
    for (u, v) in edges {
        if parity[u] == Some(false) {
            assert!(parity[v] == Some(true));
            neighbors[left_index_map[&u]].push(right_index_map[&v] as u32);
        } else {
            assert!(parity[u] == Some(true));
            assert!(parity[v] == Some(false));
            neighbors[left_index_map[&v]].push(right_index_map[&u] as u32);
        }
    }

    let n_independent = n - bipartite_match(n_left, n_right, &Jagged::from_iter(neighbors))[0]
        .iter()
        .filter(|&&v| v != UNSET)
        .count();
    writeln!(output, "{}", n_independent).unwrap();
}
