use std::io::Write;

use buffered_io::BufReadExt;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
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

const N: usize = 512;
type Mat = Vec<[u64; N / 64]>;
type V = [u64; N / 64];

fn m0() -> Mat {
    vec![[0u64; N / 64]; N]
}

fn mmul_t(ls: &Mat, rs: &Mat) -> Mat {
    let mut res = m0();
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0;
            for g in 0..N / 64 {
                acc |= ls[i][g] & rs[j][g];
            }
            res[i][j / 64] |= ((acc != 0) as u64) << (j % 64);
        }
    }
    res
}

fn mapp(ls: &Mat, v: &V) -> V {
    let mut res = [0u64; N / 64];
    for i in 0..N {
        let mut acc = 0;
        for g in 0..N / 64 {
            acc |= ls[i][g] & v[g];
        }
        res[i / 64] |= ((acc != 0) as u64) << (i % 64);
    }
    res
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let mut init = [!0u64; N / 64];
    for i in 0..n {
        init[i / 64] ^= 1 << (i % 64);
    }
    init[0 / 64] ^= 1 << 0;

    let mut p = m0();
    let mut q = m0();
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        p[v as usize][u as usize / 64] |= 1 << (u % 64);
        q[u as usize][v as usize / 64] |= 1 << (v % 64);
    }
    for i in n..N {
        p[i][i / 64] |= 1 << (i % 64);
        q[i][i / 64] |= 1 << (i % 64);
    }

    let e_bound = 18;
    let mut pow = vec![];
    for _ in 0..e_bound {
        pow.push(p.clone());

        let p_next = mmul_t(&p, &q);
        let q_next = mmul_t(&q, &p);
        p = p_next;
        q = q_next;
    }

    let mut ans = 0i64;

    let mut x = init;
    for e in (0..e_bound).rev() {
        let next = mapp(&pow[e], &x);
        if next.iter().all(|&x| x == !0u64) {
            continue;
        }

        x = next;
        ans += 1 << e;
    }
    ans += 1;

    if ans >= 1 << e_bound {
        ans = -1;
    }

    ans %= 1_000_000_007;
    writeln!(output, "{}", ans).unwrap();
}
