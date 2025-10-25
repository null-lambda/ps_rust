use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

mod dset {
    use std::{cell::Cell, mem};

    #[derive(Clone)]
    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        link: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                link: vec![Cell::new(-1); n],
            }
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            let p = self.link[u].get();
            if p >= 0 {
                let (root, size) = self.find_root_with_size(p as usize);
                self.link[u].set(root as i32);
                (root, size)
            } else {
                (u, (-p) as u32)
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true iif two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.link[v].set(u as i32);
            self.link[u].set(-((size_u + size_v) as i32));
            true
        }
    }
}

// sum_a f_a where f_a(x) = max(0, x - a)
#[derive(Clone, Default)]
struct PWLinearConvex {
    // (mult, slope)
    bps: BinaryHeap<Reverse<(i64, i64)>>,
}

impl PWLinearConvex {
    fn singleton(shift: i64) -> Self {
        Self {
            bps: [Reverse((shift, 1))].into(),
        }
    }

    // apply max(-, singleton(shift))
    fn push(&mut self, shift: i64) {
        debug_assert!(self.bps.len() >= 1);

        let (x0, dm0) = self.bps.peek().unwrap().0;
        if x0 <= shift {
            return;
        }

        self.bps.pop();
        let mut x0 = x0;
        let mut y0 = 0;
        let mut m = dm0;
        while let Some(Reverse((x, dm))) = self.bps.pop() {
            y0 += (x - x0) * m;
            x0 = x;
            m += dm;

            let q = (m * x0 - y0 - shift) / (m - 1);
            let y1 = q - shift;
            let y2 = y0 + (q + 1 - x0) * m;
            debug_assert!(y0 + (q - x0) * m <= y1);
            debug_assert!(q + 1 - shift <= y2);
            if let Some(&Reverse((x_next, _))) = self.bps.peek() {
                if q >= x_next {
                    continue;
                }
            }

            debug_assert!(y2 - y1 >= 2);
            debug_assert!(m - (y2 - y1) >= 0);
            self.bps.push(Reverse((q, y2 - y1 - 1)));
            if m - (y2 - y1) != 0 {
                self.bps.push(Reverse((q + 1, m - (y2 - y1))));
            }
            break;
        }

        self.bps.push(Reverse((shift, 1)));
    }

    fn merge(mut self, mut other: Self) -> Self {
        if self.bps.len() < other.bps.len() {
            std::mem::swap(&mut self, &mut other);
        }
        self.bps.extend(other.bps);
        self
    }

    fn batch_eval(mut self, mut queries: Vec<(i64, u32)>) -> Vec<i64> {
        queries.sort_unstable();
        let mut ans = vec![0i64; queries.len()];

        let mut x0 = 0;
        let mut y0 = 0;
        let mut m = 0;
        for (l, i) in queries {
            while let Some(&Reverse((x, dm))) = self.bps.peek() {
                if l < x {
                    break;
                }

                y0 += (x - x0) * m;
                x0 = x;
                m += dm;
                self.bps.pop();
            }

            ans[i as usize] = y0 + (l - x0) * m;
        }

        ans
    }
}

#[derive(Clone, Default)]
struct Agg {
    ecut: i64,
    vsum: PWLinearConvex,
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let mut edges = vec![];
        for _ in 0..n - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            let w: i64 = input.value();
            edges.push((u, v, w));
        }
        edges.sort_unstable_by_key(|&(.., w)| Reverse(w));

        let mut conn = dset::DisjointSet::new(n);
        let mut agg = vec![Agg::default(); n];

        for &(u, v, w) in &edges {
            agg[u as usize].ecut += w;
            agg[v as usize].ecut += w;
        }
        for u in 0..n {
            agg[u as usize].vsum = PWLinearConvex::singleton(agg[u as usize].ecut);
        }

        for &(u, v, w) in &edges {
            let ru = conn.find_root(u as usize);
            let rv = conn.find_root(v as usize);

            conn.merge(u as usize, v as usize);

            let r = conn.find_root(u as usize);

            agg[r].ecut = agg[ru].ecut + agg[rv].ecut - 2 * w;
            agg[r].vsum =
                std::mem::take(&mut agg[ru].vsum).merge(std::mem::take(&mut agg[rv].vsum));
            let ecut = agg[r].ecut;
            agg[r].vsum.push(ecut);
        }

        let q: usize = input.value();
        let mut queries: Vec<_> = (0..q as u32).map(|i| (input.value::<i64>(), i)).collect();
        queries.sort_unstable();

        for a in std::mem::take(&mut agg[conn.find_root(0)])
            .vsum
            .batch_eval(queries)
        {
            writeln!(output, "{}", a).unwrap();
        }
    }
}
