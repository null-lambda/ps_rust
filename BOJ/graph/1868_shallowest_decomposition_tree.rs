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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

fn least_significant_bit(n: u32) -> u32 {
    n & n.wrapping_neg()
}

#[derive(Default, Clone)]
struct EdgeAgg {
    banned_once: u32,
    banned_twice: u32,
}

struct NodeAgg {
    banned: u32,
}

impl EdgeAgg {
    fn pull_up(&mut self, child: &NodeAgg) {
        self.banned_twice |= self.banned_once & (child.banned + 1);
        self.banned_once |= child.banned + 1;
    }

    fn finalize(&mut self, ans: &mut u32) -> NodeAgg {
        let l = u32::BITS - u32::leading_zeros(self.banned_twice);
        let banned = self.banned_once | ((1 << l) - 1);
        *ans = (*ans).max(least_significant_bit(!banned));
        NodeAgg { banned }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }
    degree[0] += 2;

    // Shallowest decomposition tree
    // https://codeforces.com/blog/entry/125018
    // https://infossm.github.io/blog/2019/07/20/Optimal-Search-On-Tree/
    let mut ans = 0;
    let mut dp = vec![EdgeAgg::default(); n];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            xor_neighbors[p as usize] ^= u;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            let dp_u = dp[u as usize].finalize(&mut ans);
            dp[p as usize].pull_up(&dp_u);

            u = p;
        }
    }
    dp[0].finalize(&mut ans);
    let ans = u32::BITS - u32::leading_zeros(ans) - 1;

    writeln!(output, "{}", ans).unwrap();
}
