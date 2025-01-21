use std::{cmp::Reverse, io::Write};

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
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> &'a [T];
        fn get_mut(&'a mut self, u: usize) -> &'a mut [T];
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + AsMut<[Vec<T>]> + 'a,
        T: 'a,
    {
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> &'a [T] {
            &<Self as AsRef<[Vec<T>]>>::as_ref(self)[u]
        }
        fn get_mut(&'a mut self, u: usize) -> &'a mut [T] {
            &mut <Self as AsMut<[Vec<T>]>>::as_mut(self)[u]
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
        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> &'a [T] {
            &self.data[self.head[u] as usize..self.head[u + 1] as usize]
        }

        fn get_mut(&'a mut self, u: usize) -> &'a mut [T] {
            &mut self.data[self.head[u] as usize..self.head[u + 1] as usize]
        }
    }
}

fn on_lower_half(base: [i32; 2], p: [i32; 2]) -> bool {
    (p[1], p[0]) < (base[1], base[0])
}

fn signed_area(p: [i32; 2], q: [i32; 2], r: [i32; 2]) -> i32 {
    let dq = [q[0] - p[0], q[1] - p[1]];
    let dr = [r[0] - p[0], r[1] - p[1]];
    dq[0] * dr[1] - dq[1] * dr[0]
}

fn mex4(xs: impl IntoIterator<Item = u8>) -> Option<u8> {
    let mut seen = [false; 4];
    for x in xs {
        seen[x as usize] = true;
    }
    (0..4).find(|&x| !seen[x as usize])
}

fn is_connected<C>(n: usize, neighbors: impl Fn(usize) -> C, src: usize, dst: usize) -> bool
where
    C: IntoIterator<Item = usize>,
{
    let mut stack = vec![src];
    let mut visited = vec![false; n];
    visited[src] = true;
    if src == dst {
        return true;
    }
    while let Some(u) = stack.pop() {
        if u == dst {
            return true;
        }
        for v in neighbors(u) {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    false
}

fn swap_color<C>(
    n: usize,
    neighbors: impl Fn(usize) -> C,
    colors: &mut [u8],
    cs: [u8; 2],
    src: usize,
) where
    C: IntoIterator<Item = usize>,
{
    let mut swap = |u| {
        if colors[u] == cs[0] {
            colors[u] = cs[1];
            true
        } else if colors[u] == cs[1] {
            colors[u] = cs[0];
            true
        } else {
            false
        }
    };

    if !swap(src) {
        return;
    }
    let mut stack = vec![src];
    let mut visited = vec![false; n];
    visited[src] = true;
    while let Some(u) = stack.pop() {
        for v in neighbors(u) {
            if !visited[v] {
                visited[v] = true;
                if !swap(v) {
                    continue;
                }
                stack.push(v);
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let ps: Vec<[i32; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();
    let mut ps_ordered: Vec<([i32; 2], u32)> =
        ps.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
    ps_ordered.sort_unstable_by_key(|&([x, y], _)| Reverse([x, y]));

    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let mut neighbors = jagged::CSR::from_assoc_list(n, &edges);

    for u in 0..n {
        neighbors.get_mut(u).sort_unstable_by(|&(v, ()), &(w, ())| {
            let p = ps[u];
            let q = ps[v as usize];
            let r = ps[w as usize];
            (on_lower_half(p, q).cmp(&on_lower_half(p, r)))
                .then_with(|| 0.cmp(&signed_area(p, q, r)))
        });
    }

    const UNSET: u8 = !0;
    let mut color = vec![UNSET; n];
    for (_, u) in ps_ordered {
        let accessible_neighbors = |u: usize| {
            neighbors
                .get(u as usize)
                .iter()
                .map(|&(v, ())| v as usize)
                .filter(|&v| color[v] != UNSET)
        };

        color[u as usize] =
            if let Some(c) = mex4(accessible_neighbors(u as usize).map(|v| color[v])) {
                c
            } else {
                let mut vs = accessible_neighbors(u as usize);
                let vs: [_; 4] = std::array::from_fn(|_| vs.next().unwrap());
                let mut cs = [color[vs[0]], color[vs[2]]];
                let colored_neighbors =
                    |u: usize| accessible_neighbors(u).filter(|&v| cs.contains(&color[v]));
                if !is_connected(n, colored_neighbors, vs[0], vs[2]) {
                    let neighbors = |u: usize| neighbors.get(u).iter().map(|&(v, ())| v as usize);
                    swap_color(n, neighbors, &mut color, cs, vs[0]);
                } else {
                    cs = [color[vs[1]], color[vs[3]]];
                    let neighbors = |u: usize| neighbors.get(u).iter().map(|&(v, ())| v as usize);
                    swap_color(n, neighbors, &mut color, cs, vs[1]);
                }
                cs[0]
            }
    }

    for u in 0..n {
        writeln!(output, "{}", color[u] + 1).unwrap();
    }
}
