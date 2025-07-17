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

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<T>: IndexMut<usize, Output = [T]> {
        fn len(&self) -> usize;
    }

    impl<T, C> Jagged<T> for C
    where
        C: AsRef<[Vec<T>]> + IndexMut<usize, Output = [T]>,
    {
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
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
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
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
        pub fn from_pairs<I>(n: usize, pairs: I) -> Self
        where
            I: IntoIterator<Item = (u32, T)>,
            I::IntoIter: Clone,
        {
            let mut head = vec![0u32; n + 1];

            let pairs = pairs.into_iter();
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

            CSR { data, head }
        }
    }

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> Jagged<T> for CSR<T> {
        fn len(&self) -> usize {
            self.head.len() - 1
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    for _ in 0..n {
        let _x: u32 = input.value();
        let _y: u32 = input.value();
    }

    let mut edges: Vec<_> = (0..m)
        .map(|_| [input.value::<u32>() - 1, input.value::<u32>() - 1])
        .collect();
    let neighbors = jagged::CSR::from_pairs(n, edges.iter().flat_map(|&[u, v]| [(u, v), (v, u)]));

    const INF: u32 = 1 << 29;
    let mut toposort = vec![];
    let mut timer = 0;
    let mut degree: Vec<_> = (0..n).map(|u| neighbors[u].len() as u32).collect();
    for u in 0..n as u32 {
        if degree[u as usize] <= 5 {
            degree[u as usize] = INF;
            toposort.push(u);
        }
    }

    while let Some(&u) = toposort.get(timer) {
        timer += 1;

        for &v in &neighbors[u as usize] {
            degree[v as usize] -= 1;
            if degree[v as usize] <= 5 {
                degree[v as usize] = INF;
                toposort.push(v);
            }
        }
    }
    assert_eq!(toposort.len(), n);

    let mut inv_map = vec![!0u32; n];
    for u in 0..n as u32 {
        inv_map[toposort[u as usize] as usize] = u;
    }

    for [u, v] in &mut edges {
        *u = inv_map[*u as usize] as u32;
        *v = inv_map[*v as usize] as u32;
        if !(*u > *v) {
            std::mem::swap(u, v);
        }
    }
    let neighbors = jagged::CSR::from_pairs(n, edges.iter().flat_map(|&[u, v]| [(u, v), (v, u)]));

    let has_3cycle = || {
        let mut marker = vec![false; n];
        for u in 0..n as u32 {
            for &v in &neighbors[u as usize] {
                if v < u {
                    marker[v as usize] = true;
                }
            }

            for &v in &neighbors[u as usize] {
                if v < u {
                    for &w in &neighbors[v as usize] {
                        if w < v {
                            if marker[w as usize] {
                                return true;
                            }
                        }
                    }
                }
            }

            for &v in &neighbors[u as usize] {
                if v < u {
                    marker[v as usize] = false;
                }
            }
        }
        false
    };

    let has_4cycle = || {
        let mut marker = vec![false; n];
        for u in 0..n as u32 {
            for &v in &neighbors[u as usize] {
                if v < u {
                    for &w in &neighbors[v as usize] {
                        if w < u {
                            if marker[w as usize] {
                                return true;
                            }
                            marker[w as usize] = true;
                        }
                    }
                }
            }
            for &v in &neighbors[u as usize] {
                if v < u {
                    for &w in &neighbors[v as usize] {
                        marker[w as usize] = false;
                    }
                }
            }
        }
        false
    };

    let ans = (has_3cycle().then_some(3))
        .or_else(|| has_4cycle().then_some(4))
        .unwrap_or(5);
    writeln!(output, "{}", ans).unwrap();
}
