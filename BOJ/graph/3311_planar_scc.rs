use std::io::Write;

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
        fn get(&'a self, u: usize) -> &'a [T];
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
        fn get(&'a self, u: usize) -> &'a [T] {
            &self.as_ref()[u]
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
            let v: Vec<Vec<&T>> = (0..self.len())
                .map(|i| self.get(i).iter().collect())
                .collect();
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

        fn get(&'a self, u: usize) -> &'a [T] {
            &self.data[self.head[u] as usize..self.head[u + 1] as usize]
        }
    }
}

fn gen_scc<'a>(neighbors: &'a impl jagged::Jagged<'a, u32>) -> (usize, Vec<u32>) {
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

            if iv < neighbors.get(u as usize).len() as u32 {
                // Iterate neighbors
                dfs_stack.push((u, iv + 1));

                let v = neighbors.get(u as usize)[iv as usize];
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

// chunk_by in std >= 1.77
fn group_by<T, P, F>(xs: &[T], mut pred: P, mut f: F)
where
    P: FnMut(&T, &T) -> bool,
    F: FnMut(&[T]),
{
    let mut i = 0;
    while i < xs.len() {
        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        f(&xs[i..j]);
        i = j;
    }
}

const INF: i32 = 1_000_000_010;
const NEG_INF: i32 = -INF;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let x_max: i32 = input.value();
    let _y_max: i32 = input.value();

    let ps: Vec<[i32; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();
    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let k = input.token();
        edges.push((u, v));
        if k != "1" {
            edges.push((v, u));
        }
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let (_scc_count, scc_index) = gen_scc(&neighbors);

    let mut topological_order: Vec<_> = (0..n).map(|u| (scc_index[u], u as u32)).collect();
    topological_order.sort_unstable();

    let mut ps_asc: Vec<_> = (0..n).map(|u| (ps[u], u as u32)).collect();
    ps_asc.sort_unstable_by_key(|&(p, _)| p[1]);

    let mut reachable = vec![false; n];
    topological_order.reverse();
    group_by(
        &topological_order,
        |&(scc_u, _), &(scc_v, _)| scc_u == scc_v,
        |scc| {
            let mut acc = false;
            for &(_, u) in scc {
                acc |= ps[u as usize][0] == 0;
                acc |= reachable[u as usize];
            }

            for &(_, u) in scc {
                reachable[u as usize] |= acc;
                for &v in neighbors.get(u as usize) {
                    reachable[v as usize] |= reachable[u as usize];
                }
            }
        },
    );

    topological_order.reverse();

    let mut dp = vec![[INF, NEG_INF]; n];
    let mut n_ends = 0;
    for &(p, u) in ps_asc.iter() {
        if p[0] == x_max && reachable[u as usize] {
            dp[u as usize] = [n_ends, n_ends + 1];
            n_ends += 1;
        }
    }

    group_by(
        &topological_order,
        |&(scc_u, _), &(scc_v, _)| scc_u == scc_v,
        |scc| {
            let mut acc = [INF, NEG_INF];
            for &(_, u) in scc {
                acc[0] = acc[0].min(dp[u as usize][0]);
                acc[1] = acc[1].max(dp[u as usize][1]);
                for &v in neighbors.get(u as usize) {
                    acc[0] = acc[0].min(dp[v as usize][0]);
                    acc[1] = acc[1].max(dp[v as usize][1]);
                }
            }

            for &(_, u) in scc {
                dp[u as usize] = acc;
            }
        },
    );

    for &(p, u) in ps_asc.iter().rev() {
        if p[0] == 0 {
            let ans = (dp[u as usize][1] - dp[u as usize][0]).max(0);
            writeln!(output, "{}", ans).unwrap();
        }
    }
}
