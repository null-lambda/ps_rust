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

pub mod segtree {
    // monoid, not necesserily commutative
    pub trait Monoid {
        fn id() -> Self;
        fn op(self, rhs: Self) -> Self;
    }

    pub trait CommMonoid: Monoid {}

    pub trait PowMonoid: Monoid {
        fn pow(self, n: u32) -> Self;
    }

    // monoid action A -> End(M), where A is a monoid and M is a set.
    // the image of A is a submonoid of End(M)
    pub trait MonoidAction<M>: Monoid {
        fn apply_to_sum(self, x_sum: M, x_count: u32) -> M;
    }

    // monoid action on itself
    impl<M: PowMonoid> MonoidAction<M> for M {
        fn apply_to_sum(self, x_sum: M, x_count: u32) -> M {
            self.pow(x_count).op(x_sum)
        }
    }

    pub struct LazySegTree<T, F> {
        n: usize,
        max_height: u32,
        pub sum: Vec<T>,
        pub lazy: Vec<F>,
    }

    impl<T, F> LazySegTree<T, F>
    where
        T: Monoid + Copy + Eq,
        F: MonoidAction<T> + Copy + Eq,
    {
        pub fn with_size(n: usize) -> Self {
            let n = n.next_power_of_two();
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: vec![T::id(); 2 * n],
                lazy: vec![F::id(); n],
            }
        }

        pub fn from_iter<I>(n: usize, iter: I) -> Self
        where
            I: IntoIterator<Item = T>,
        {
            use std::iter::repeat;
            let n = n.next_power_of_two();
            let mut sum: Vec<T> = repeat(T::id())
                .take(n)
                .chain(iter.into_iter())
                .chain(repeat(T::id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = sum[i << 1].op(sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: vec![F::id(); n],
            }
        }

        #[inline]
        fn apply(&mut self, node: usize, width: u32, value: F) {
            self.sum[node] = value.apply_to_sum(self.sum[node], width);
            if node < self.n {
                // function application is right associative
                self.lazy[node] = value.op(self.lazy[node]);
            }
        }

        #[inline]
        fn propagate_lazy(&mut self, mut idx: usize) {
            idx += self.n;
            for height in (1..=self.max_height).rev() {
                let node = idx >> height;
                if self.lazy[node] != F::id() {
                    let width: u32 = 1 << (height - 1);
                    self.apply(node << 1, width, self.lazy[node]);
                    self.apply(node << 1 | 1, width, self.lazy[node]);
                    self.lazy[node] = F::id();
                }
            }
        }

        #[inline]
        fn update_sum(&mut self, node: usize, width: u32) {
            self.sum[node] = self.sum[node << 1].op(self.sum[node << 1 | 1]);
            if self.lazy[node] != F::id() {
                self.sum[node] = self.lazy[node].apply_to_sum(self.sum[node], width);
            };
        }

        // sum on interval [left, right)
        pub fn apply_range(&mut self, mut start: usize, mut end: usize, value: F) {
            if value == F::id() || start == end {
                return;
            }
            debug_assert!(end <= self.n);
            self.propagate_lazy(start);
            self.propagate_lazy(end - 1);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut update_left, mut update_right) = (false, false);
            while start < end {
                if update_left {
                    self.update_sum(start - 1, width);
                }
                if update_right {
                    self.update_sum(end, width);
                }
                if start & 1 != 0 {
                    self.apply(start, width, value);
                    update_left = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, value);
                    update_right = true;
                }
                start = (start + 1) >> 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if update_left {
                    self.update_sum(start, width);
                }
                if update_right && !(update_left && start == end) {
                    self.update_sum(end, width);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, mut start: usize, mut end: usize) -> T {
            self.propagate_lazy(start);
            self.propagate_lazy(end - 1);
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
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use segtree::*;

    let n: usize = input.value();
    let n_queries: usize = input.value();
    let mut neighbors: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    for _ in 0..n - 1 {
        let u: usize = input.value();
        let v: usize = input.value();
        neighbors[u - 1].push(v - 1);
        neighbors[v - 1].push(u - 1);
    }

    let mut children: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    {
        let mut visited = vec![false; n];
        visited[0] = true;
        fn dfs(
            neighbors: &[Vec<usize>],
            children: &mut [Vec<usize>],
            visited: &mut [bool],
            u: usize,
        ) {
            for &v in &neighbors[u] {
                if !visited[v] {
                    visited[v] = true;
                    children[u].push(v);
                    dfs(neighbors, children, visited, v);
                }
            }
        }
        dfs(&neighbors, &mut children, &mut visited, 0);
    }

    #[derive(Debug)]
    struct HeavyLightDecomposition {
        size: Vec<u32>,
        depth: Vec<u32>,
        parent: Vec<usize>,
        heavy_child: Vec<usize>,
        chain_top: Vec<usize>,
        euler_in: Vec<usize>,
        euler_out: Vec<usize>,
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
            self.euler_in[u] = *order;
            *order += 1;
            if !children[u].is_empty() {
                let h = self.heavy_child[u];
                self.chain_top[h] = self.chain_top[u];
                self.dfs_decompose(children, h, order);
                for &v in children[u].iter().filter(|&&v| v != h) {
                    self.chain_top[v] = v;
                    self.dfs_decompose(children, v, order);
                }
            }
            self.euler_out[u] = *order;
        }

        pub fn build(children: &[Vec<usize>]) -> Self {
            let n = children.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![0; n],
                heavy_child: vec![0; n],
                chain_top: vec![0; n],
                euler_in: vec![0; n],
                euler_out: vec![0; n],
            };
            hld.dfs_size(children, 0);
            hld.dfs_decompose(children, 0, &mut 0);
            hld
        }

        // note: work only if M is a commutative monoid.
        pub fn iterate_path(&self, mut u: usize, mut v: usize, mut f: impl FnMut(&usize, &usize)) {
            debug_assert!(u.max(v) < self.parent.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u]] < self.depth[self.chain_top[v]] {
                    std::mem::swap(&mut u, &mut v);
                }
                f(&self.euler_in[self.chain_top[u]], &self.euler_in[u]);
                u = self.parent[self.chain_top[u]];
            }
            if self.euler_in[u] > self.euler_in[v] {
                std::mem::swap(&mut u, &mut v);
            }
            f(&self.euler_in[u], &self.euler_in[v]);
        }
    }

    const P: u64 = 1 << 32;
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    struct Add(u64);
    impl Monoid for Add {
        fn id() -> Self {
            Self(0)
        }
        fn op(self, other: Self) -> Self {
            Self((self.0 + other.0) % P)
        }
    }
    impl CommMonoid for Add {}

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    struct AffineTrans(u64, u64);
    impl Monoid for AffineTrans {
        fn id() -> Self {
            AffineTrans(0, 1)
        }
        fn op(self, other: Self) -> Self {
            AffineTrans(
                (self.0 + (self.1 * other.0) % P) % P,
                (self.1 * other.1) % P,
            )
        }
    }
    impl MonoidAction<Add> for AffineTrans {
        fn apply_to_sum(self, x_sum: Add, x_count: u32) -> Add {
            Add((self.0 * x_count as u64 + (self.1 * x_sum.0) % P) % P)
        }
    }

    let hld = HeavyLightDecomposition::build(&children[..]);
    let mut segtree = LazySegTree::with_size(n);

    for _ in 0..n_queries {
        use std::hint::unreachable_unchecked;
        let q = input.value();
        let action = |value| match q {
            1 | 2 => AffineTrans(value, 1),
            3 | 4 => AffineTrans(0, value),
            _ => unsafe { unreachable_unchecked() },
        };
        match q {
            1 | 3 => {
                let u: usize = input.value();
                let value = input.value();
                segtree.apply_range(hld.euler_in[u - 1], hld.euler_out[u - 1], action(value));
            }
            2 | 4 => {
                let u: usize = input.value();
                let v: usize = input.value();
                let value = input.value();
                hld.iterate_path(u - 1, v - 1, |&i, &j| {
                    segtree.apply_range(i, j + 1, action(value));
                });
            }
            5 => {
                let u: usize = input.value();
                let result = segtree.query_range(hld.euler_in[u - 1], hld.euler_out[u - 1]);
                writeln!(output_buf, "{}", result.0).unwrap();
            }
            6 => {
                let u: usize = input.value();
                let v: usize = input.value();
                let mut result = Add::id();
                hld.iterate_path(u - 1, v - 1, |&i, &j| {
                    result = result.op(segtree.query_range(i, j + 1));
                });
                writeln!(output_buf, "{}", result.0).unwrap();
            }
            _ => panic!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
