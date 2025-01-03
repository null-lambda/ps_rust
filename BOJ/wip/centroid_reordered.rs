use std::io::Write;

use jagged::Jagged;

mod simple_io {
    use std::string::*;

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

pub mod centroid {
    // WIP
    pub mod reordered {
        /// Centroid Decomposition
        /// Keeps reindexing graph by BFS order, for better locality
        use crate::jagged::Jagged;

        const UNSET: u32 = !0;

        fn reroot_to_centroid<'a, E: 'a>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            size: &mut [u32],
            visited: &[bool],
            mut u: usize,
        ) -> usize {
            let threshold = size[u] / 2;
            let mut p = u;
            'outer: loop {
                for &(v, _) in neighbors.get(u) {
                    if v as usize == p || visited[v as usize] {
                        continue;
                    }
                    if size[v as usize] > threshold {
                        size[u] -= size[v as usize];
                        size[v as usize] += size[u];

                        // Why no auto tail call
                        p = u;
                        u = v as usize;
                        continue 'outer;
                    }
                }
                return u;
            }
        }

        pub fn reindex_by_bfs<'a, E: 'a + Default + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            root: u32,
        ) -> Vec<(u32, E)> {
            let n = neighbors.len();
            let mut timer = 0;
            let mut bfs_order = vec![(root as u32, root as u32, E::default())];
            let mut index_map = vec![UNSET; n];
            index_map[root as usize] = 0;
            while let Some(&(u, p, _)) = bfs_order.get(timer) {
                for (v, w) in neighbors.get(u as usize).cloned() {
                    if v == p {
                        continue;
                    }
                    bfs_order.push((v, u, w));
                    index_map[v as usize] = timer as u32;
                }
                timer += 1;
            }
            assert!(timer == n, "Invalid tree structure");

            let mut parent = vec![(0, E::default()); n];
            for &(u, p, _) in bfs_order.iter().skip(1) {
                parent[index_map[u as usize] as usize] = (index_map[p as usize], E::default());
            }

            parent
        }

        pub fn dnc<'a, E: 'a + Default + Clone, F>(
            parent_reordered: &[(u32, E)],
            size: &mut [u32],
            yield_rooted_tree: &mut F,
        ) where
            F: FnMut(&[(u32, E)]), // TODO
        {
            let n = parent_reordered.len();

            let centroid = 'outer: {
                let mut size = vec![1; n];
                let threshold = size[0] / 2;
                for (u, &(p, _)) in parent_reordered.iter().enumerate().skip(1).rev() {
                    if size[u] > threshold {
                        break 'outer u;
                    }

                    size[p as usize] += size[u];
                }
                panic!()
            };

            let mut path_to_parent = vec![(centroid, E::default())];
            let mut u = centroid;
            loop {
                let (p, w) = parent_reordered[u].clone();
                path_to_parent.push((u, w));
                u = p as usize;
            }

            yield_rooted_tree(parent_reordered);

            for &(v, _) in neighbors.get(root) {
                if visited[v as usize] {
                    continue;
                }
                dnc(neighbors, size, visited, yield_rooted_tree, v as usize);
            }
        }
    }
}

const INF: u32 = u32::MAX / 3;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: u32 = input.value();

    let mut edges = vec![];
    for _ in 0..n - 1 {
        let u: u32 = input.value();
        let v: u32 = input.value();
        let w: u32 = input.value();
        edges.push((u, (v, w)));
        edges.push((v, (u, w)));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let mut ans = INF;
    let mut dp = vec![INF; k as usize + 1];
    let mut size = vec![0; n];
    let mut visited = vec![false; n];
    let mut depth = vec![0; n];
    let mut dist = vec![0; n];
    let mut bfs_order = vec![];
    // centroid::reordered::init_size(&neighbors, &mut size, &mut visited, 0, 0);
    centroid::reordered::dnc(
        &neighbors,
        &mut size,
        &mut visited,
        &mut |visited, root| {
            dp[0] = 0;
            let mut dists_buffer = vec![];
            for &(entry, w) in neighbors.get(root) {
                if visited[entry as usize] {
                    continue;
                }

                dist[entry as usize] = w;
                depth[entry as usize] = 1;

                let dists_start = dists_buffer.len();
                if dist[entry as usize] <= k {
                    dists_buffer.push((dist[entry as usize], depth[entry as usize]));
                }

                bfs_order.push((entry, root as u32));

                let mut timer = 0;
                while let Some(&(u, p)) = bfs_order.get(timer) {
                    timer += 1;
                    for &(v, w) in neighbors.get(u as usize) {
                        if visited[v as usize] || v == p {
                            continue;
                        }
                        dist[v as usize] = dist[u as usize] + w;
                        depth[v as usize] = depth[u as usize] + 1;
                        dists_buffer.push((dist[v as usize], depth[v as usize]));

                        bfs_order.push((v, u));
                    }
                }
                let dists = &dists_buffer[dists_start..];

                for &(dist, depth) in dists {
                    if dist <= k {
                        ans = ans.min(depth + dp[(k - dist) as usize]);
                    }
                }
                for &(dist, depth) in dists {
                    if dist <= k {
                        dp[dist as usize] = dp[dist as usize].min(depth);
                    }
                }

                bfs_order.clear();
            }

            for (dist, _) in dists_buffer {
                if dist <= k {
                    dp[dist as usize] = INF;
                }
            }
        },
        0,
    );
    if ans == INF {
        writeln!(output, "-1").unwrap();
    } else {
        writeln!(output, "{}", ans).unwrap();
    }
}
