use std::io::Write;

#[allow(dead_code)]
mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

const P: u64 = 1_000_000_007;

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: u64) -> u64 {
    pow(n, P - 2)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let n: usize = input.value();
    let m: usize = input.value();

    // Paul Burkhardt, David G. Harris. "Simple and efficient four-cycle counting on sparse graphs"
    // https://arxiv.org/abs/2303.06090

    // order by degree
    let mut degree = vec![0; n];
    let edges: Vec<_> = (0..m)
        .map(|_| {
            let u: usize = input.value();
            let v: usize = input.value();
            degree[u - 1] += 1;
            degree[v - 1] += 1;
            (u - 1, v - 1)
        })
        .collect();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by_key(|&i| degree[i]);
    let mut inv_indices = vec![0; n];
    for i in 0..n {
        inv_indices[indices[i]] = i;
    }
    drop(degree);

    let mut children = vec![vec![]; n];
    let mut neighbors = vec![vec![]; n];
    for &(u, v) in &edges {
        let (u, v) = (u as usize, v as usize);
        let (mut u, mut v) = (inv_indices[u], inv_indices[v]);
        if !(u < v) {
            std::mem::swap(&mut u, &mut v);
        }
        children[v].push(u);
        neighbors[v].push(u);
        neighbors[u].push(v);
    }

    let mut cnt = vec![0; n];
    for u in 0..n {
        for &v in &children[u] {
            for &w in &neighbors[v] {
                if w < u {
                    cnt[u] += 1;
                    cnt[v] += 1;
                    cnt[w] += 1;
                }
            }
        }
    }

    println!("{:?}", cnt);
}
