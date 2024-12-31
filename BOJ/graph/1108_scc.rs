use std::{collections::HashMap, io::Write};

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

fn gen_scc<'a>(neighbors: &Jagged<u32>) -> (usize, Vec<u32>) {
    // Tarjan algorithm, iterative
    let n = neighbors.len();

    const UNSET: u32 = u32::MAX;
    let mut scc_index: Vec<u32> = vec![UNSET; n];
    let mut scc_count = 0;

    let mut path_stack = vec![];
    let mut dfs_stack = vec![];
    let mut order_count: u32 = 1;
    let mut order: Vec<u32> = vec![0; n];
    let mut low_link: Vec<u32> = vec![UNSET; n];

    for u in 0..n {
        if order[u] > 0 {
            continue;
        }

        const UPDATE_LOW_LINK: u32 = 1 << 31;

        dfs_stack.push((u as u32, 0));
        while let Some((u, iv)) = dfs_stack.pop() {
            if iv & UPDATE_LOW_LINK != 0 {
                let v = iv ^ UPDATE_LOW_LINK;
                low_link[u as usize] = low_link[u as usize].min(low_link[v as usize]);
                continue;
            }

            if iv == 0 {
                // Enter node
                order[u as usize] = order_count;
                low_link[u as usize] = order_count;
                order_count += 1;
                path_stack.push(u);
            }

            if iv < neighbors[u as usize].len() as u32 {
                // Iterate neighbors
                dfs_stack.push((u, iv + 1));

                let v = neighbors[u as usize][iv as usize];
                if order[v as usize] == 0 {
                    dfs_stack.push((u, v | UPDATE_LOW_LINK));
                    dfs_stack.push((v, 0));
                } else if scc_index[v as usize] == UNSET {
                    low_link[u as usize] = low_link[u as usize].min(order[v as usize]);
                }
            } else {
                // Exit node
                if low_link[u as usize] == order[u as usize] {
                    // Found a strongly connected component
                    loop {
                        let v = path_stack.pop().unwrap();
                        scc_index[v as usize] = scc_count;
                        if v == u {
                            break;
                        }
                    }
                    scc_count += 1;
                }
            }
        }
    }
    (scc_count as usize, scc_index)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let m = input.value();
    let mut index_map = HashMap::new();
    let mut query_index = |x| {
        let idx = index_map.len() as u32;
        *index_map.entry(x).or_insert_with(|| idx)
    };

    let mut edges = vec![];
    for _ in 0..m {
        let u = query_index(input.token());
        for _ in 0..input.value() {
            let v = query_index(input.token());
            edges.push((u, v));
        }
    }
    let target = index_map[input.token()];
    let n = index_map.len();

    let neighbors = Jagged::from_assoc_list(n, &edges);
    let (scc_count, scc_index) = gen_scc(&neighbors);

    let mut components = vec![vec![]; scc_count];
    for (i, &c) in scc_index.iter().enumerate() {
        components[c as usize].push(i);
    }

    let mut score = vec![u64::MAX; n];
    for c in 0..scc_count {
        for &u in &components[c] {
            score[u] = 1;
            for &v in &neighbors[u] {
                if scc_index[v as usize] != scc_index[u] {
                    score[u] += score[v as usize];
                }
            }
        }
    }

    let ans = score[target as usize];
    writeln!(output, "{}", ans).unwrap();
}
