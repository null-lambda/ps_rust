use std::{
    collections::{BinaryHeap, VecDeque},
    io::Write,
    mem::{self, MaybeUninit},
    usize,
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

struct Node {
    cs_lower: BinaryHeap<u32>,
    cs_sum: u32,
}

impl Node {
    fn truncate(&mut self, sum_ub: u32) {
        loop {
            if self.cs_sum <= sum_ub {
                break;
            }
            self.cs_sum -= self.cs_lower.pop().unwrap();
        }
    }

    fn union(&mut self, mut other: Self, sum_ub: u32) {
        if self.cs_lower.len() < other.cs_lower.len() {
            mem::swap(self, &mut other);
        }
        self.cs_sum += other.cs_sum;
        self.cs_lower.extend(other.cs_lower.into_iter());
        self.truncate(sum_ub);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: u32 = input.value();

    let mut parents = vec![0; n];
    let mut cs = vec![0; n];
    let mut ls = vec![0; n];
    let mut root = 0;
    for i in 0..n {
        match input.value::<usize>().checked_sub(1) {
            Some(p) => parents[i] = p,
            None => root = i,
        }
        cs[i] = input.value();
        ls[i] = input.value();
    }

    let mut dp: Vec<_> = (0..n)
        .map(|i| {
            let mut node = Node {
                cs_lower: [cs[i]].into(),
                cs_sum: cs[i],
            };
            node.truncate(m);
            MaybeUninit::new(node)
        })
        .collect();

    let mut indegree = vec![0; n];
    for i in 0..n {
        if i != root {
            indegree[parents[i]] += 1;
        }
    }

    let mut ans = 0;
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    while let Some(u) = queue.pop_front() {
        unsafe {
            let dp_u = dp[u].assume_init_read();
            ans = ans.max(ls[u] as u64 * dp_u.cs_lower.len() as u64);

            let parent = parents[u];
            if u == root {
                continue;
            }
            assert!(u != parent); // Safety guard

            dp[parent].assume_init_mut().union(dp_u, m);
            indegree[parent] -= 1;
            if indegree[parent] == 0 {
                queue.push_back(parent);
            }
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
