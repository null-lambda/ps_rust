use std::{collections::BTreeMap, io::Write};

mod simple_io {
    use std::string::*;

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

#[derive(Default, Clone, Debug)]
struct DistFreq {
    base: i64,
    delta: BTreeMap<i64, u64>,
}

impl DistFreq {
    fn singleton(c: u64, h: i64) -> Self {
        if h > 0 {
            Self {
                base: 0,
                delta: Default::default(),
            }
        } else {
            Self {
                base: 0,
                delta: [(0, c)].into(),
            }
        }
    }

    fn pull_up(&mut self, mut child: Self, weight: i64) {
        child.base += weight;
        if self.delta.len() < child.delta.len() {
            std::mem::swap(self, &mut child);
        }

        let d_base = child.base - self.base;
        for (dist, count) in child.delta {
            *self.delta.entry(dist + d_base).or_default() += count;
        }
    }

    fn finalize(&mut self, mut c: u64, h: i64, ans: &mut u64) {
        if h < 0 {
            debug_assert!(h == -1);
            return;
        }

        let mut to_remove = vec![];
        for (dist, count) in self.delta.range_mut(..=h - self.base).rev() {
            let delta_count = (*count).min(c);
            *count -= delta_count;
            c -= delta_count;
            *ans += delta_count;

            if *count == 0 {
                to_remove.push(*dist);
            } else {
                assert!(c == 0);
                break;
            }
        }

        for dist in to_remove {
            self.delta.remove(&dist);
        }
    }
}

fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
    debug_assert!(i < xs.len() && j < xs.len());
    if i == j {
        return None;
    }
    let ptr = xs.as_mut_ptr();
    Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();

    let cs: Vec<u64> = (0..n).map(|_| input.value()).collect();
    let hs: Vec<i64> = (0..n).map(|_| input.value()).collect();

    let root = 0;
    let mut degree = vec![0u32; n];
    let mut parent = vec![0u32; n];
    degree[root] = 2;
    for u in 1..n as u32 {
        let p = input.value::<u32>() - 1;
        degree[p as usize] += 1;
        degree[u as usize] += 1;
        parent[u as usize] = p;
    }

    let weight: Vec<i64> = std::iter::once(0)
        .chain((1..n).map(|_| input.value()))
        .collect();

    let mut ans = 0u64;
    let mut dp: Vec<_> = (0..n).map(|u| DistFreq::singleton(cs[u], hs[u])).collect();
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = parent[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            dp[u as usize].finalize(cs[u as usize], hs[u as usize], &mut ans);
            let (dp_u, dp_p) = get_two(&mut dp, u as usize, p as usize).unwrap();
            dp_p.pull_up(std::mem::take(dp_u), weight[u as usize]);

            u = p;
        }
    }
    dp[root].finalize(cs[root], hs[root], &mut ans);

    writeln!(output, "{}", ans).unwrap();
}
