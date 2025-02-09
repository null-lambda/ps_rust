use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap},
    hash::Hash,
    io::Write,
    ops::Range,
};

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

// Merge sort tree with dynamic insertion/removal
type Point<T> = (u32, T);
struct MergeSortTree<T> {
    n: usize,
    data: Vec<BTreeMap<T, u32>>,
}

impl<T: Ord + Clone> MergeSortTree<T> {
    fn new(x_max: u32) -> Self {
        let n = (x_max + 1).next_power_of_two();
        Self {
            n: (x_max + 1).next_power_of_two() as usize,
            data: (0..2 * n).map(|_| Default::default()).collect(),
        }
    }

    fn insert(&mut self, (x, y): Point<T>) {
        let mut u = x as usize + self.n;
        while u >= 1 {
            *self.data[u].entry(y.clone()).or_default() += 1;
            u >>= 1;
        }
    }

    fn remove(&mut self, (x, y): Point<T>) {
        let mut u = x as usize + self.n;
        while u >= 1 {
            if let Some(e) = self.data[u].get_mut(&y) {
                *e -= 1;
                if *e == 0 {
                    self.data[u].remove(&y);
                }
            }
            u >>= 1;
        }
    }

    fn query_rect(&mut self, x_range: Range<u32>, y_range: Range<T>) -> Vec<T> {
        let mut res = vec![];
        self.query_rect_rec(&mut res, &x_range, &y_range, 0..self.n as u32, 1);
        res
    }

    fn query_rect_rec(
        &mut self,
        res: &mut Vec<T>,
        x_range: &Range<u32>,
        y_range: &Range<T>,
        x_view: Range<u32>,
        u: usize,
    ) {
        if x_range.end <= x_view.start || x_view.end <= x_range.start {
            return;
        }
        if x_range.start <= x_view.start && x_view.end <= x_range.end {
            for (y, _) in self.data[u].range(y_range.clone()) {
                res.push(y.clone());
            }
            return;
        }

        let mid = x_view.start + x_view.end >> 1;
        self.query_rect_rec(res, x_range, y_range, x_view.start..mid, u << 1);
        self.query_rect_rec(res, x_range, y_range, mid..x_view.end, u << 1 | 1);
    }
}

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

type Idx = u32;

const INF_DIST: i64 = 1 << 60;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: i32 = input.value();
    let m: usize = input.value();
    let mut treatments = vec![];

    let mut dist = vec![INF_DIST; m];
    let mut queue = BinaryHeap::new();
    let mut is_src = vec![false; m];
    let mut is_dest = vec![false; m];
    for u in 0..m {
        let t: i32 = input.value();
        let s: i32 = input.value();
        let e = input.value::<i32>() + 1;
        let c: i64 = input.value();
        treatments.push((t, s, e, c));

        if s == 1 {
            is_src[u as usize] = true;
            dist[u] = c;
            queue.push((Reverse(c), u as Idx));
        }
        if e == n + 1 {
            is_dest[u as usize] = true;
        }
    }

    let mut xs = vec![];
    for &(t, s, e, _) in &treatments {
        xs.push(s - t);
        xs.push(e - t + 1);
    }
    let (x_map, x_inv) = compress_coord(xs);
    let x_bound = x_map.len() as u32;

    let mut ps = MergeSortTree::<(i32, Idx)>::new(x_bound);
    for u in 0..m {
        let (t, s, _, _) = treatments[u];
        ps.insert((x_inv[&(s - t)], (s + t, u as Idx)));
    }

    let mut ans = None;
    while let Some((Reverse(d), u)) = queue.pop() {
        if dist[u as usize] < d {
            continue;
        }
        if is_dest[u as usize] {
            ans = Some(d);
            break;
        }

        let (t, _, e, _) = treatments[u as usize];
        let vs = ps.query_rect(0..x_inv[&(e - t + 1)], (i32::MIN, 0)..(e + t + 1, 0));
        // println!("query {:?} => {:?}", (x_inv[&(e - t + 1)], (e + t + 1)), vs);

        for &(y, v) in &vs {
            let (t, s, _, c) = treatments[v as usize];
            let x = x_inv[&(s - t)];
            ps.remove((x, (y, v)));

            let dv_new = d + c;
            if dv_new < dist[v as usize] {
                dist[v as usize] = dv_new;
                queue.push((Reverse(dv_new), v));
            }
        }
    }

    if let Some(ans) = ans {
        writeln!(output, "{}", ans).unwrap();
    } else {
        writeln!(output, "-1").unwrap();
    }
}
