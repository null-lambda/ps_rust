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

const UNSET: u32 = !0;
fn cut_verts<'a>(neighbors: &'a impl jagged::Jagged<'a, u32>) -> Vec<bool> {
    let n = neighbors.len();

    // DFS tree structure
    let mut parent = vec![!0u32; n];
    let mut low = vec![0; n]; // Lowest destination of subtree's back edge
    let mut euler_in = vec![0; n];
    let mut timer = 1u32;

    // BCC structure
    let mut is_cut_vert = vec![false; n];
    let mut sub_bcc_count = vec![0u32; n];

    let mut current_edge = vec![0u32; n];
    for root in 0..n {
        if euler_in[root] != 0 {
            continue;
        }

        parent[root] = UNSET;
        let mut u = root as u32;
        loop {
            let p = parent[u as usize];
            let iv = &mut current_edge[u as usize];
            if *iv == 0 {
                euler_in[u as usize] = timer;
                low[u as usize] = timer + 1;
                timer += 1;
            }
            if (*iv as usize) == neighbors.get(u as usize).len() {
                if p == UNSET {
                    break;
                }
                low[p as usize] = low[p as usize].min(low[u as usize]);
                is_cut_vert[p as usize] |= low[u as usize] >= euler_in[p as usize];
                u = p;
                continue;
            }

            let v = neighbors.get(u as usize)[*iv as usize];
            *iv += 1;
            if v == p {
                continue;
            }
            if euler_in[v as usize] != 0 {
                // Back edge
                low[u as usize] = low[u as usize].min(euler_in[v as usize]);
                continue;
            }

            // Forward edge (a part of DFS spanning tree)
            parent[v as usize] = u;
            sub_bcc_count[u as usize] += 1;
            u = v;
        }

        is_cut_vert[root] = sub_bcc_count[root] >= 2;
    }

    is_cut_vert
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, v));
        edges.push((v, u));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let is_cut_vert = cut_verts(&neighbors);
    let n_cut_verts = is_cut_vert.iter().filter(|&&b| b).count();
    writeln!(output, "{}", n_cut_verts).unwrap();
    for u in 0..n {
        if is_cut_vert[u] {
            write!(output, "{} ", u + 1).unwrap();
        }
    }
    writeln!(output).unwrap();
}
