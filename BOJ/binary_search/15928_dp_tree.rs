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

fn xor_assign_tuple3<T1, T2, T3>(a: &mut (T1, T2, T3), (b1, b2, b3): &(T1, T2, T3))
where
    T1: std::ops::BitXorAssign + Copy,
    T2: std::ops::BitXorAssign + Copy,
    T3: std::ops::BitXorAssign + Copy,
{
    a.0 ^= *b1;
    a.1 ^= *b2;
    a.2 ^= *b3;
}

#[derive(Clone)]
struct NodeAgg {
    max_depth: i64,
}

impl NodeAgg {
    fn empty() -> Self {
        Self { max_depth: 0 }
    }

    fn pull_from(&mut self, child: &Self, weight: i32, diam: &mut i64) {
        *diam = (*diam).max(self.max_depth + child.max_depth + weight as i64);
        self.max_depth = self.max_depth.max(child.max_depth + weight as i64);
    }
}

fn ternary_search<F, K>(mut left: i32, mut right: i32, mut f: F) -> i32
where
    K: Ord,
    F: FnMut(&i32) -> K,
{
    while right - left > 3 {
        let m1 = left + (right - left) / 3;
        let m2 = right - (right - left) / 3;
        if f(&m1) <= f(&m2) {
            right = m2;
        } else {
            left = m1;
        }
    }
    (left..=right).min_by_key(|&x| f(&x)).unwrap()
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let d: i32 = input.value();
    let mut degree = vec![0; n];
    let mut xor_neighbors = vec![(0, 0, 0); n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let intercept: i32 = input.value();
        let slope: i32 = input.value();
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_assign_tuple3(&mut xor_neighbors[u as usize], &(v, intercept, slope));
        xor_assign_tuple3(&mut xor_neighbors[v as usize], &(u, intercept, slope));
    }
    degree[0] += 2;

    let mut topological_order = vec![];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let (p, intercept, slope) = xor_neighbors[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            xor_assign_tuple3(&mut xor_neighbors[p as usize], &(u, intercept, slope));
            topological_order.push((u, p, intercept, slope));

            u = p;
        }
    }

    let diam = |day: i32| {
        let weight = |intercept: i32, slope: i32| intercept + slope * day;

        let mut diam = 0;
        let mut dp = vec![NodeAgg::empty(); n];
        for &(u, p, intercept, slope) in &topological_order {
            let dp_u = dp[u as usize].clone();
            dp[p as usize].pull_from(&dp_u, weight(intercept, slope), &mut diam);
        }
        diam
    };
    let opt = ternary_search(0, d, |&day| (diam(day), day));
    writeln!(output, "{}", opt).unwrap();
    writeln!(output, "{}", diam(opt)).unwrap();
}
