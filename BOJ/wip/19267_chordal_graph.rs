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

pub mod chordal_graph {
    use crate::jagged::*;

    mod linked_list {
        use std::{
            marker::PhantomData,
            num::NonZeroU32,
            ops::{Index, IndexMut},
        };

        #[derive(Debug)]
        pub struct Cursor<T> {
            idx: NonZeroU32,
            _marker: PhantomData<*const T>,
        }

        // Arena-allocated pool of doubly linked lists.
        // Semantically unsafe, as cursors can outlive and access removed elements.
        #[derive(Clone, Debug)]
        pub struct MultiList<T> {
            links: Vec<[Option<Cursor<T>>; 2]>,
            values: Vec<T>,
            freed: Vec<Cursor<T>>,
        }

        impl<T> Clone for Cursor<T> {
            fn clone(&self) -> Self {
                Self::new(self.idx.get() as usize)
            }
        }

        impl<T> Copy for Cursor<T> {}

        impl<T> Cursor<T> {
            fn new(idx: usize) -> Self {
                Self {
                    idx: NonZeroU32::new(idx as u32).unwrap(),
                    _marker: PhantomData,
                }
            }

            pub fn usize(&self) -> usize {
                self.idx.get() as usize
            }
        }

        impl<T> Index<Cursor<T>> for MultiList<T> {
            type Output = T;
            fn index(&self, index: Cursor<T>) -> &Self::Output {
                &self.values[index.usize()]
            }
        }

        impl<T> IndexMut<Cursor<T>> for MultiList<T> {
            fn index_mut(&mut self, index: Cursor<T>) -> &mut Self::Output {
                &mut self.values[index.usize()]
            }
        }

        impl<T: Default> MultiList<T> {
            pub fn new() -> Self {
                Self {
                    links: vec![[None; 2]],
                    values: vec![Default::default()],
                    freed: vec![],
                }
            }

            pub fn next(&self, i: Cursor<T>) -> Option<Cursor<T>> {
                self.links[i.usize()][1]
            }

            pub fn prev(&self, i: Cursor<T>) -> Option<Cursor<T>> {
                self.links[i.usize()][0]
            }

            pub fn singleton(&mut self, value: T) -> Cursor<T> {
                if let Some(idx) = self.freed.pop() {
                    self.links[idx.usize()] = [None; 2];
                    self.values[idx.usize()] = value;
                    idx
                } else {
                    let idx = self.links.len();
                    self.links.push([None; 2]);
                    self.values.push(value);
                    Cursor::new(idx)
                }
            }

            fn link(&mut self, u: Cursor<T>, v: Cursor<T>) {
                self.links[u.usize()][1] = Some(v);
                self.links[v.usize()][0] = Some(u);
            }

            pub fn insert_left(&mut self, i: Cursor<T>, value: T) -> Cursor<T> {
                let v = self.singleton(value);
                if let Some(j) = self.prev(i) {
                    self.link(j, v);
                }
                self.link(v, i);
                v
            }

            pub fn insert_right(&mut self, i: Cursor<T>, value: T) -> Cursor<T> {
                let v = self.singleton(value);
                if let Some(j) = self.next(i) {
                    self.link(v, j);
                }
                self.link(i, v);
                v
            }

            pub fn erase(&mut self, i: Cursor<T>) {
                let l = self.prev(i);
                let r = self.next(i);
                if let Some(l_inner) = l {
                    self.links[l_inner.usize()][1] = r;
                }
                if let Some(r_inner) = r {
                    self.links[r_inner.usize()][0] = l;
                }
                self.links[i.usize()] = [None; 2];

                self.freed.push(i);
            }
        }
    }

    // Lexicographic bfs, O(N)
    pub fn lex_bfs(neighbors: &CSR<u32>) -> (Vec<u32>, Vec<u32>) {
        let n = neighbors.len();

        let mut heads = linked_list::MultiList::new();
        let h_begin = heads.singleton(0u32);
        let h_end = heads.insert_right(h_begin, n as u32);

        let mut bfs: Vec<_> = (0..n as u32).collect();
        let mut t_in: Vec<_> = (0..n as u32).collect();
        let mut visited = vec![false; n];
        let mut owner = vec![h_begin; n];

        let erase_head = |heads: &mut linked_list::MultiList<u32>, h: &mut _| {
            if heads[*h] + 1 != heads[heads.next(*h).unwrap()] {
                heads[*h] += 1;
            } else {
                heads.erase(*h);
                *h = h_end;
            }
        };

        let mut t_split = vec![!0; n];
        for i in 0..n {
            let u = bfs[i] as usize;
            visited[u] = true;
            erase_head(&mut heads, &mut owner[u]);

            for &v in &neighbors[u] {
                let v = v as usize;
                if visited[v] {
                    continue;
                }

                let h = bfs[heads[owner[v]] as usize] as usize;
                t_in.swap(h, v);
                bfs.swap(t_in[h] as usize, t_in[v] as usize);

                if t_split.len() <= owner[v].usize() {
                    t_split.resize(owner[v].usize() + 1, !0);
                }
                let p = if t_split[owner[v].usize()] == i as u32 {
                    heads.prev(owner[v]).unwrap()
                } else {
                    t_split[owner[v].usize()] = i as u32;
                    heads.insert_left(owner[v], t_in[v])
                };
                erase_head(&mut heads, &mut owner[v]);
                owner[v] = p;
            }
        }

        (bfs, t_in)
    }

    // Perfect elimination ordering, reversed.
    pub fn rev_peo(neighbors: &CSR<u32>) -> Option<(Vec<u32>, Vec<u32>)> {
        let n = neighbors.len();
        let (bfs, t_in) = lex_bfs(neighbors);

        let mut successors = vec![];
        for u in 0..n {
            let mut t_prev = None;
            for &v in &neighbors[u] {
                if t_in[v as usize] < t_in[u] {
                    t_prev = t_prev.max(Some(t_in[v as usize]));
                }
            }

            if let Some(t_prev) = t_prev {
                successors.push((t_prev, u as u32));
            }
        }
        let successors = CSR::from_pairs(n, successors.iter().copied());

        let mut marker = vec![!0; n];
        for t_prev in 0..n {
            if successors[t_prev].is_empty() {
                continue;
            }

            let prev = bfs[t_prev as usize] as usize;
            for &v in &neighbors[prev] {
                marker[v as usize] = t_prev as u32;
            }

            for &u in &successors[t_prev] {
                for &v in &neighbors[u as usize] {
                    if t_in[v as usize] < t_in[prev] && marker[v as usize] != t_prev as u32 {
                        return None;
                    }
                }
            }
        }

        Some((bfs, t_in))
    }
}

const UNSET: u32 = !0;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let edges = (0..m)
        .map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1])
        .collect::<Vec<_>>();
    let edges = || edges.iter().flat_map(|&[u, v]| [(u, v), (v, u)]);
    let neighbors = jagged::CSR::from_pairs(n, edges());
    let (_bfs, t_in) = chordal_graph::rev_peo(&neighbors).unwrap();

    let clique_edges = edges()
        .map(|(u, v)| (t_in[u as usize], t_in[v as usize]))
        .filter(|&(u, v)| u > v)
        .chain((0..n as u32).map(|u| (u, u)))
        .collect::<Vec<_>>();
    let mut clique = jagged::CSR::from_pairs(n, clique_edges.iter().copied());
    for u in 0..n {
        clique[u].sort_unstable();
    }

    let mut dp: Vec<_> = (0..n)
        .map(|u| {
            let k = clique[u].len();
            let k_pad = k + 1;
            vec![0u32; k_pad * k_pad]
        })
        .collect();

    let local_map = |k: usize| {
        let k_pad = k + 1;

        let empty = k * k_pad + k;
        let single = move |j| k * k_pad + j;
        let double = move |i, j| i * k_pad + j;
        (empty, single, double)
    };

    let mut inv_u = vec![UNSET; n];
    let mut embedding = vec![UNSET; n];

    for u in (0..n).rev() {
        let k = clique[u].len();
        let (u_empty, u_single, u_double) = local_map(k);

        // Link internal edges
        for i in 0..k {
            dp[u][u_single(i)] = dp[u][u_single(i)].max(dp[u][u_empty] + 1);
        }
        for i in 0..k {
            for j in 0..i {
                dp[u][u_double(i, j)] = dp[u][u_double(i, j)]
                    .max(dp[u][u_single(i)] + 1)
                    .max(dp[u][u_single(j)] + 1);
            }
        }

        println!("{:?}", &clique[u]);
        println!("{:?}", dp[u]);

        if k == 1 {
            break;
        }

        // Forget {u}
        let iu = k - 1;
        dp[u][u_empty] = dp[u][u_empty].max(dp[u][u_single(iu)]);
        for i in 0..k - 1 {
            for j in 0..i {
                if i == iu || j == iu {
                    dp[u][u_single(iu)] = dp[u][u_single(iu)].max(dp[u][u_double(i, j)]);
                }
            }
        }

        // Introduce bag_p \ bag_u, and join
        let ip = k - 2;
        let p = dp[u][ip] as usize;
        for i in 0..k - 1 {
            inv_u[clique[u][i] as usize] = i as u32;
        }

        let m = clique[p].len();
        let (p_empty, p_single, p_double) = local_map(m);
        let embedding = &mut embedding[..k];
        for j in 0..m {
            let v = clique[p][j] as usize;
            if inv_u[v] != UNSET {
                embedding[inv_u[v] as usize] = j as u32;
            }
        }

        dp[p][p_empty] += dp[u][u_empty];
        for i in 0..k - 1 {
            let x = embedding[i] as usize;
            if x == UNSET as usize {
                continue;
            }
            dp[p][p_single(x)] += dp[u][u_single(i)];

            for j in 0..i {
                let y = embedding[j] as usize;
                if y == UNSET as usize {
                    continue;
                }
                dp[p][p_double(x, y)] += dp[u][u_double(i, j)];
            }
        }

        embedding.fill(UNSET);
        for i in 0..k - 1 {
            inv_u[clique[u][i] as usize] = UNSET;
        }
    }

    let k = clique[0].len();
    let k_pad = k + 1;
    let mut ans = dp[0][k * k_pad + k];
    for i in 0..k + 1 {
        for j in 0..i {
            ans = ans.max(dp[0][i * k_pad + j]);
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
