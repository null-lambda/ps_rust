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

#[derive(Clone, Copy, Debug)]
struct NodeData {
    count: u64,
    sum: u64,
    sum_sq: u64,
}

impl NodeData {
    fn leaf() -> Self {
        NodeData {
            count: 1,
            sum: 0,
            sum_sq: 0,
        }
    }

    fn branch() -> Self {
        NodeData {
            count: 0,
            sum: 0,
            sum_sq: 0,
        }
    }

    fn pull_from(&mut self, mut other: Self, weight: u64, ans: &mut u64) {
        other.sum_sq += 2 * other.sum * weight + other.count * weight * weight;
        other.sum += other.count * weight;

        *ans += self.sum_sq * other.count + self.count * other.sum_sq + 2 * self.sum * other.sum;

        self.count += other.count;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![(0u32, 0u64); n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w = input.value::<u64>();
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize].0 ^= v;
        xor_neighbors[v as usize].0 ^= u;
        xor_neighbors[u as usize].1 ^= w;
        xor_neighbors[v as usize].1 ^= w;
    }
    let mut ans = 0;
    let mut dp = (0..n)
        .map(|u| {
            if degree[u] == 1 {
                NodeData::leaf()
            } else {
                NodeData::branch()
            }
        })
        .collect::<Vec<_>>();
    degree[0] += 2;

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let (p, w) = xor_neighbors[u as usize];
            degree[p as usize] -= 1;
            degree[u as usize] -= 1;
            xor_neighbors[p as usize].0 ^= u as u32;
            xor_neighbors[p as usize].1 ^= w;

            let dp_u = dp[u as usize];
            dp[p as usize].pull_from(dp_u, w, &mut ans);

            u = p;
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
