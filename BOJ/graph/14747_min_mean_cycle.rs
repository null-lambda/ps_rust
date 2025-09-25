use std::io::Write;

use frac::Frac;

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

    impl<T> Default for CSR<T> {
        fn default() -> Self {
            Self {
                links: vec![],
                head: vec![0],
            }
        }
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

fn gen_scc(children: &jagged::CSR<u32>) -> (usize, Vec<u32>) {
    // Tarjan algorithm, iterative
    let n = children.len();

    const UNSET: u32 = !0;
    let mut scc_index = vec![UNSET; n];
    let mut n_scc = 0;

    // Stackless DFS
    let mut parent = vec![UNSET; n];
    let mut current_edge: Vec<_> = (0..n)
        .map(|u| children.edge_range(u).start as u32)
        .collect();
    let mut t_in = vec![0u32; n];
    let mut timer = 1;

    let mut low_link = vec![UNSET; n];
    let mut path_stack = vec![];

    for mut u in 0..n as u32 {
        if t_in[u as usize] > 0 {
            continue;
        }

        parent[u as usize] = u;
        loop {
            let e = current_edge[u as usize];
            current_edge[u as usize] += 1;

            if e == children.edge_range(u as usize).start as u32 {
                // On enter
                t_in[u as usize] = timer;
                low_link[u as usize] = timer;
                timer += 1;
                path_stack.push(u);
            }

            if e < children.edge_range(u as usize).end as u32 {
                let v = children.links[e as usize];
                if t_in[v as usize] == 0 {
                    // Front edge
                    parent[v as usize] = u;

                    u = v;
                } else if scc_index[v as usize] == UNSET {
                    // Back edge or cross edge, scc not constructed yet
                    low_link[u as usize] = low_link[u as usize].min(t_in[v as usize]);
                }
            } else {
                // On exit
                if low_link[u as usize] == t_in[u as usize] {
                    // Found a scc
                    loop {
                        let v = path_stack.pop().unwrap();
                        scc_index[v as usize] = n_scc;
                        if v == u {
                            break;
                        }
                    }
                    n_scc += 1;
                }

                let p = parent[u as usize];
                if p == u {
                    break;
                }
                low_link[p as usize] = low_link[p as usize].min(low_link[u as usize]);
                u = p;
            }
        }
    }
    (n_scc as usize, scc_index)
}

pub mod frac {
    use std::ops::*;

    type S = i64;
    type U = u64;

    fn gcd(mut a: U, mut b: U) -> U {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Frac(S, S);

    impl Frac {
        pub fn new(n: S, d: S) -> Self {
            assert!(d > 0, "Denominator must be always positive");
            Self(n, d).normalized()
        }

        pub fn numer(&self) -> S {
            self.0
        }

        pub fn denom(&self) -> S {
            self.1
        }

        pub fn inner(&self) -> (S, S) {
            (self.0, self.1)
        }

        pub fn normalized(self) -> Self {
            let Self(n, d) = self;
            let g = gcd(n.abs() as U, d.abs() as U) as S * d.signum();
            Self(n / g, d / g)
        }

        pub fn zero() -> Self {
            Self(0, 1)
        }

        pub fn one() -> Self {
            Self(1, 1)
        }

        pub fn abs(self) -> Self {
            Self(self.0.abs(), self.1)
        }
    }

    impl Add for Frac {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 + rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Sub for Frac {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 - rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Mul for Frac {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.0, self.1 * rhs.1)
        }
    }

    impl Div for Frac {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let s = rhs.0.signum();
            Self::new(self.0 * rhs.1 * s, self.1 * rhs.0 * s)
        }
    }

    macro_rules! forward_binop {
        ($OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
            impl $Op<&Frac> for Frac {
                type Output = Frac;
                fn $op(self, rhs: &Frac) -> Self::Output {
                    self.$op(*rhs)
                }
            }

            impl $Op<Frac> for &Frac {
                type Output = Frac;
                fn $op(self, rhs: Frac) -> Self::Output {
                    (*self).$op(rhs)
                }
            }

            impl $Op<&Frac> for &Frac {
                type Output = Frac;
                fn $op(self, rhs: &Frac) -> Self::Output {
                    (*self).$op(*rhs)
                }
            }

            impl $OpAssign<Frac> for Frac {
                fn $op_assign(&mut self, rhs: Frac) {
                    *self = (*self).$op(rhs);
                }
            }

            impl $OpAssign<&Frac> for Frac {
                fn $op_assign(&mut self, rhs: &Frac) {
                    *self = (*self).$op(*rhs);
                }
            }
        };
    }

    forward_binop!(AddAssign add_assign, Add add);
    forward_binop!(SubAssign sub_assign, Sub sub);
    forward_binop!(MulAssign mul_assign, Mul mul);
    forward_binop!(DivAssign div_assign, Div div);

    impl Neg for Frac {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self(-self.0, self.1)
        }
    }

    impl From<S> for Frac {
        fn from(a: S) -> Self {
            Self::new(a, 1)
        }
    }

    impl From<(S, S)> for Frac {
        fn from((n, d): (S, S)) -> Self {
            Self::new(n, d)
        }
    }

    impl PartialEq for Frac {
        fn eq(&self, other: &Self) -> bool {
            self.0 * other.1 == other.0 * self.1
        }
    }

    impl Eq for Frac {}

    impl PartialOrd for Frac {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some((self.0 * other.1).cmp(&(other.0 * self.1)))
        }
    }

    impl Ord for Frac {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }
}

const INF: i64 = 1 << 30;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>();
        let v = input.value::<u32>();
        let w = input.value::<i64>();
        edges.push((u, v, w));
    }
    let neighbors_unweighted = jagged::CSR::from_pairs(n, edges.iter().map(|&(u, v, _)| (u, v)));

    let (n_scc, color) = gen_scc(&neighbors_unweighted);
    let mut scc_verts = vec![vec![]; n_scc];
    let mut scc_edges = vec![vec![]; n_scc];
    for u in 0..n {
        scc_verts[color[u] as usize].push(u as u32);
    }
    for &(u, v, w) in &edges {
        if color[u as usize] == color[v as usize] {
            scc_edges[color[u as usize] as usize].push((u, v, w));
        }
    }

    // Kahn's min mean cycle
    let mut min_mean_cycle = Frac::from(INF);
    let mut min_dist = vec![vec![INF; n]; n + 1];
    for (vs, es) in scc_verts.into_iter().zip(scc_edges) {
        let r = vs.len();

        for k in 0..=r {
            for &v in &vs {
                min_dist[k as usize][v as usize] = INF;
            }
        }
        min_dist[0][vs[0] as usize] = 0;

        for k in 1..=r {
            for &(u, v, w) in &es {
                min_dist[k][v as usize] =
                    min_dist[k][v as usize].min(min_dist[k - 1][u as usize] + w);
            }
        }

        for u in vs {
            if min_dist[r][u as usize] == INF {
                continue;
            }
            let mut row = Frac::from(-INF);
            for k in 0..r {
                if min_dist[k][u as usize] == INF {
                    continue;
                }

                row = row.max(
                    (
                        min_dist[r][u as usize] - min_dist[k][u as usize],
                        (r - k) as i64,
                    )
                        .into(),
                );
            }
            min_mean_cycle = min_mean_cycle.min(row);
        }
    }

    min_mean_cycle = min_mean_cycle.normalized();
    if min_mean_cycle == INF.into() {
        writeln!(output, "0 0").unwrap();
    } else {
        writeln!(
            output,
            "{} {}",
            min_mean_cycle.numer(),
            min_mean_cycle.denom()
        )
        .unwrap();
    }
}
