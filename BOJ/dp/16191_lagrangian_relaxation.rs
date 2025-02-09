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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

type PairedSum = (i64, u32);

const NEG_INF: i64 = -(1 << 60);

#[derive(Clone)]
struct NodeAgg {
    exclusive: PairedSum,
    inclusive: PairedSum,
}

impl Default for NodeAgg {
    fn default() -> Self {
        Self {
            exclusive: (0, 0),
            inclusive: (NEG_INF, 0),
        }
    }
}

impl NodeAgg {
    fn pull_from(&mut self, child: &Self, weight: i64) {
        let add = |x: PairedSum, y: PairedSum| (x.0 + y.0, x.1 + y.1);
        let sub = |x: PairedSum, y: PairedSum| (x.0 - y.0, x.1 - y.1);
        self.exclusive = add(self.exclusive, child.exclusive.max(child.inclusive));
        self.inclusive = self.inclusive.max(sub(
            add(child.exclusive, (weight, 1)),
            child.exclusive.max(child.inclusive),
        ));
    }

    fn finalize(&mut self) {
        self.inclusive.0 += self.exclusive.0;
        self.inclusive.1 += self.exclusive.1;
    }

    fn collapse(&self) -> PairedSum {
        self.inclusive.max(self.exclusive)
    }
}

fn partition_point<P>(mut left: i64, mut right: i64, mut pred: P) -> i64
where
    P: FnMut(i64) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let k: usize = input.value();

    let mut degree = vec![0; n];
    let mut xor_neighbors = vec![(0, 0); n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w: i32 = input.value::<i32>();
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize].0 ^= v;
        xor_neighbors[v as usize].0 ^= u;
        xor_neighbors[u as usize].1 ^= w;
        xor_neighbors[v as usize].1 ^= w;
    }
    degree[0] += 2;

    let mut topological_order = vec![];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let (p, w) = xor_neighbors[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize].0 ^= u;
            xor_neighbors[p as usize].1 ^= w;
            topological_order.push((u, p, w));

            u = p;
        }
    }

    // Aliens trick
    let max_k_matchings = |slope: i64| {
        let mut dp = vec![NodeAgg::default(); n];
        for &(u, p, w) in &topological_order {
            let w_adjusted = w as i64 - slope;

            let mut dp_u = std::mem::take(&mut dp[u as usize]);
            dp_u.finalize();
            dp[p as usize].pull_from(&dp_u, w_adjusted);
        }
        dp[0].finalize();
        dp[0].collapse()
    };

    let slope_bound = 1_000_000 * 250_000 + 1;
    let opt_slope = partition_point(-slope_bound, slope_bound, |slope| {
        max_k_matchings(slope).1 >= k as u32
    }) - 1;
    let (mut max_sum, _k_max) = max_k_matchings(opt_slope);
    if max_k_matchings(opt_slope - 1).1 < k as u32 {
        writeln!(output, "Impossible").unwrap();
        return;
    }

    max_sum += opt_slope as i64 * k as i64;
    writeln!(output, "{}", max_sum).unwrap();
}
