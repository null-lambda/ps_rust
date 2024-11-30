use std::{collections::VecDeque, io::Write};

use collections::DisjointMap;

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

mod collections {
    use std::cell::Cell;
    use std::mem::MaybeUninit;

    pub struct DisjointMap<T> {
        // represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        values: Vec<MaybeUninit<T>>,
    }

    impl<T> DisjointMap<T> {
        pub fn new(values: impl IntoIterator<Item = T>) -> Self {
            let node_weights: Vec<_> = values.into_iter().map(|c| MaybeUninit::new(c)).collect();
            let n = node_weights.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                values: node_weights,
            }
        }

        pub fn get_size(&self, u: usize) -> u32 {
            -self.parent_or_size[self.find_root(u)].get() as u32
        }

        pub fn get_mut(&mut self, u: usize) -> &mut T {
            let r = self.find_root(u);
            unsafe { self.values[r].assume_init_mut() }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if self.parent_or_size[u].get() < 0 {
                u
            } else {
                let root = self.find_root(self.parent_or_size[u].get() as usize);
                self.parent_or_size[u].set(root as i32);
                root
            }
        }

        // returns whether two set were different
        pub fn merge(
            &mut self,
            mut u: usize,
            mut v: usize,
            mut combine_values: impl FnMut(T, T) -> T,
        ) -> bool {
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            let size_u = -self.parent_or_size[u].get() as i32;
            let size_v = -self.parent_or_size[v].get() as i32;
            if size_u >= size_v {
                self.parent_or_size[v].set(u as i32);
                self.parent_or_size[u].set(-(size_u + size_v));
                unsafe {
                    self.values[u] = MaybeUninit::new(combine_values(
                        self.values[u].assume_init_read(),
                        self.values[v].assume_init_read(),
                    ))
                }
            } else {
                self.parent_or_size[u].set(v as i32);
                self.parent_or_size[v].set(-(size_u + size_v));
                unsafe {
                    self.values[v] = MaybeUninit::new(combine_values(
                        self.values[u].assume_init_read(),
                        self.values[v].assume_init_read(),
                    ))
                }
            }
            true
        }
    }

    impl<T> Drop for DisjointMap<T> {
        fn drop(&mut self) {
            for i in 0..self.parent_or_size.len() {
                if self.parent_or_size[i].get() < 0 {
                    unsafe {
                        self.values[i].assume_init_drop();
                    }
                }
            }
        }
    }
}

fn prefix_sum<T: std::ops::Add<Output = T> + Default + Clone>(
    iter: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = T> {
    let mut sum = T::default();
    std::iter::once(sum.clone()).chain(iter.into_iter().map(move |x| {
        sum = sum.clone() + x;
        sum.clone()
    }))
}

fn diff<T: std::ops::Sub<Output = T> + Default + Clone>(
    iter: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = T> {
    let mut last = T::default();
    iter.into_iter()
        .map(move |x| {
            let res = x.clone() - last.clone();
            last = x;
            res
        })
        .skip(1)
}

#[derive(Default, Debug)]
struct Line {
    prefix_sum: VecDeque<i64>,
    indices_base: i32,
}

impl Line {
    fn len(&self) -> usize {
        self.prefix_sum.len() - 1
    }

    fn join(self, other: Self, indices: &mut Vec<i32>) -> Self {
        let mut sa = self;
        let mut sb = other;
        let n = sa.len();
        let dx = *sa.prefix_sum.back().unwrap() - sb.prefix_sum[0];
        let di = n as i32 + sb.indices_base - sa.indices_base;
        if sa.len() >= sb.len() {
            for x in diff(sb.prefix_sum.iter().copied()) {
                indices[x as usize] += di;
            }
            sa.prefix_sum
                .extend(sb.prefix_sum.into_iter().skip(1).map(|x| x + dx));
            sa
        } else {
            sb.prefix_sum.pop_front();
            for x in diff(sa.prefix_sum.iter().copied()) {
                indices[x as usize] -= di;
            }
            for x in sa.prefix_sum.into_iter().rev() {
                sb.prefix_sum.push_front(x - dx);
            }
            sb.indices_base += n as i32;
            sb
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut lines = vec![];
    for _ in 0..n {
        let l: usize = input.value();
        let xs: Vec<u32> = (0..l).map(|_| input.value()).collect();
        lines.push(xs);
    }

    let x_max = lines.iter().map(|xs| xs.len()).sum::<usize>();
    let mut dmap = DisjointMap::new((0..=x_max).map(|_| Line::default()));
    let mut indices_delta = vec![0; x_max + 1];
    for xs in lines {
        let l = xs.len();
        for i in 1..l {
            dmap.merge(xs[0] as usize, xs[i] as usize, |s, _| s);
        }

        let line = dmap.get_mut(xs[0] as usize);
        line.prefix_sum = prefix_sum(xs.iter().map(|&x| x as i64)).collect();
        line.indices_base = 0;
        for (i, &x) in xs.iter().enumerate() {
            indices_delta[x as usize] = i as i32;
        }
    }

    let q: usize = input.value();
    for _ in 0..q {
        let cmd = input.token();
        let a = input.value();
        let b = input.value();
        match cmd {
            "1" => {
                if dmap.merge(a, b, |sa, sb| sa.join(sb, &mut indices_delta)) {
                    writeln!(output, "YES").unwrap();
                } else {
                    writeln!(output, "NO").unwrap();
                }
            }
            "2" => {
                if dmap.find_root(a) == dmap.find_root(b) {
                    let line = dmap.get_mut(a);
                    let mut ia = indices_delta[a] + line.indices_base;
                    let mut ib = indices_delta[b] + line.indices_base;
                    if ia > ib {
                        std::mem::swap(&mut ia, &mut ib);
                    }
                    let res = line.prefix_sum[ib as usize + 1] - line.prefix_sum[ia as usize];
                    writeln!(output, "{}", res).unwrap();
                } else {
                    writeln!(output, "-1").unwrap();
                }
            }
            _ => panic!(),
        }
    }
}
