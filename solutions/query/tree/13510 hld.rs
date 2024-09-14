mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

pub trait Monoid {
    fn id() -> Self;
    fn op(self, rhs: Self) -> Self;
}

pub trait CommMonoid: Monoid {}

#[derive(Debug)]
pub struct SegTree<T> {
    n: usize,
    sum: Vec<T>,
}

impl<T> SegTree<T>
where
    T: Monoid + Copy + Eq,
{
    pub fn with_size(n: usize) -> Self {
        Self {
            n,
            sum: vec![T::id(); 2 * n],
        }
    }

    pub fn from_iter<I>(n: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        use std::iter::repeat;
        let mut sum: Vec<T> = repeat(T::id())
            .take(n)
            .chain(iter.into_iter())
            .chain(repeat(T::id()))
            .take(2 * n)
            .collect();
        for i in (0..n).rev() {
            sum[i] = sum[i << 1].op(sum[i << 1 | 1]);
        }
        Self { n, sum }
    }

    pub fn set(&mut self, mut idx: usize, value: T) {
        debug_assert!(idx < self.n);
        idx += self.n;
        self.sum[idx] = value;
        while idx > 1 {
            idx >>= 1;
            self.sum[idx] = self.sum[idx << 1].op(self.sum[idx << 1 | 1]);
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> T {
        self.sum[idx + self.n]
    }

    // sum on interval [left, right)
    pub fn query_range(&self, mut start: usize, mut end: usize) -> T {
        debug_assert!(start <= end && end <= self.n);
        start += self.n;
        end += self.n;
        let (mut result_left, mut result_right) = (T::id(), T::id());
        while start < end {
            if start & 1 != 0 {
                result_left = result_left.op(self.sum[start]);
            }
            if end & 1 != 0 {
                result_right = self.sum[end - 1].op(result_right);
            }
            start = (start + 1) >> 1;
            end >>= 1;
        }

        result_left.op(result_right)
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::iter::once;

    let n: usize = input.value();
    let mut neighbors: Vec<Vec<(usize, usize, u32)>> = (0..n).map(|_| Vec::new()).collect();
    for i_edge in 0..n - 1 {
        let u: usize = input.value();
        let v: usize = input.value();
        let weight: u32 = input.value();
        neighbors[u - 1].push((i_edge, v - 1, weight));
        neighbors[v - 1].push((i_edge, u - 1, weight));
    }

    let mut children: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    let mut weights = vec![0; n];
    let mut edge_bottom = vec![0; n];
    {
        let mut visited = vec![false; n];
        visited[0] = true;
        fn dfs(
            neighbors: &[Vec<(usize, usize, u32)>],
            children: &mut [Vec<usize>],
            weights: &mut [u32],
            edge_bottom: &mut [usize],
            visited: &mut [bool],
            u: usize,
        ) {
            for &(i_edge, v, weight) in &neighbors[u] {
                if !visited[v] {
                    visited[v] = true;
                    children[u].push(v);
                    weights[v] = weight;
                    edge_bottom[i_edge] = v;
                    dfs(neighbors, children, weights, edge_bottom, visited, v);
                }
            }
        }
        dfs(
            &neighbors,
            &mut children,
            &mut weights,
            &mut edge_bottom,
            &mut visited,
            0,
        );
    }

    #[derive(Debug)]
    struct HeavyLightDecomposition {
        size: Vec<u32>,
        depth: Vec<u32>,
        parent: Vec<usize>,
        heavy_child: Vec<usize>,
        chain_top: Vec<usize>,
        euler_idx: Vec<usize>,
    }

    impl HeavyLightDecomposition {
        fn dfs_size(&mut self, children: &[Vec<usize>], u: usize) {
            self.size[u] = 1;
            if children.is_empty() {
                return;
            }
            for &v in &children[u] {
                self.depth[v] = self.depth[u] + 1;
                self.parent[v] = u;
                self.dfs_size(children, v);
                self.size[u] += self.size[v];
            }
            self.heavy_child[u] = children[u]
                .iter()
                .copied()
                .max_by_key(|&v| self.size[v])
                .unwrap_or(0);
        }

        fn dfs_decompose(&mut self, children: &[Vec<usize>], u: usize, order: &mut usize) {
            self.euler_idx[u] = *order;
            *order += 1;
            if children[u].is_empty() {
                return;
            }
            let h = self.heavy_child[u];
            self.chain_top[h] = self.chain_top[u];
            self.dfs_decompose(children, h, order);
            for &v in children[u].iter().filter(|&&v| v != h) {
                self.chain_top[v] = v;
                self.dfs_decompose(children, v, order);
            }
        }

        pub fn build(children: &[Vec<usize>]) -> Self {
            let n = children.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![0; n],
                heavy_child: vec![0; n],
                chain_top: vec![0; n],
                euler_idx: vec![0; n],
            };
            hld.dfs_size(children, 0);
            hld.dfs_decompose(children, 0, &mut 0);
            hld
        }

        // note: work only if M is a commutative monoid.
        pub fn query_path<M>(&self, mut u: usize, mut v: usize, segtree: &SegTree<M>) -> M
        where
            M: CommMonoid + Copy + Eq,
        {
            debug_assert!(u.max(v) < self.parent.len());

            let mut result = M::id();
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u]] < self.depth[self.chain_top[v]] {
                    std::mem::swap(&mut u, &mut v);
                }
                result = result
                    .op(segtree
                        .query_range(self.euler_idx[self.chain_top[u]], self.euler_idx[u] + 1));
                u = self.parent[self.chain_top[u]];
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            result = result.op(segtree.query_range(self.euler_idx[u] + 1, self.euler_idx[v] + 1));
            result
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    struct Max(u32);

    impl Monoid for Max {
        fn id() -> Self {
            Self(0)
        }
        fn op(self, other: Self) -> Self {
            Self(self.0.max(other.0))
        }
    }

    impl CommMonoid for Max {}

    let hld = HeavyLightDecomposition::build(&children[..]);
    let mut segtree = SegTree::with_size(n);
    for u in 0..n {
        segtree.set(hld.euler_idx[u], Max(weights[u]));
    }

    let n_queries = input.value();
    for _ in 0..n_queries {
        let q = input.value();
        match q {
            1 => {
                let i: usize = input.value();
                let c = input.value();
                segtree.set(hld.euler_idx[edge_bottom[i - 1]], Max(c));
            }
            2 => {
                let u = input.value::<usize>() - 1;
                let v = input.value::<usize>() - 1;
                writeln!(output_buf, "{}", hld.query_path(u, v, &segtree).0).unwrap();
            }
            _ => panic!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
