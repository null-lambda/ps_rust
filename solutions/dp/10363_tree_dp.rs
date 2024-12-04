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

mod graph {
    use std::{collections::VecDeque, iter};

    pub fn toposort<'a, C, F>(n: usize, mut children: F) -> impl 'a + Iterator<Item = usize>
    where
        F: 'a + FnMut(usize) -> C,
        C: IntoIterator<Item = usize>,
    {
        let mut indegree = vec![0u32; n];
        for node in 0..n {
            for child in children(node) {
                indegree[child] += 1;
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&node| indegree[node] == 0).collect();
        iter::from_fn(move || {
            let current = queue.pop_front()?;
            for child in children(current) {
                indegree[child] -= 1;
                if indegree[child] == 0 {
                    queue.push_back(child);
                }
            }
            Some(current)
        })
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for i_tc in 1..=input.value() {
        let n: usize = input.value();
        let weights: Vec<u32> = (0..n).map(|_| input.value()).collect();
        let parent: Vec<usize> = (0..n)
            .map(|_| input.value::<usize>().saturating_sub(1))
            .collect();
        let mut sub_grundies = vec![0u16; n];

        let root = 0;
        let mut ans = 0u32;
        for u in graph::toposort(n, |u| (u != root).then_some(parent[u])) {
            let grundy = sub_grundies[u].trailing_ones();
            if weights[u] % 2 == 1 {
                ans ^= grundy;
            }
            if u != root {
                sub_grundies[parent[u]] |= 1 << grundy;
            }
        }
        if ans != 0 {
            writeln!(output, "Case #{}: first", i_tc).unwrap();
        } else {
            writeln!(output, "Case #{}: second", i_tc).unwrap();
        }
    }
}
