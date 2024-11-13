use std::collections::HashSet;
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
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn dfs_size(neighbors: &[Vec<u32>], size: &mut [u32], u: usize, parent: usize) {
    size[u] = 1;
    for &v in &neighbors[u] {
        if v as usize != parent {
            dfs_size(neighbors, size, v as usize, u);
            size[u] += size[v as usize];
        }
    }
}

fn dfs_centroid(
    neighbors: &[Vec<u32>],
    size: &[u32],
    u: usize,
    parent: usize,
    centroid: &mut Vec<usize>,
) {
    let n = size.len();
    let mut is_centroid = true;
    for &v in &neighbors[u] {
        if v as usize != parent {
            dfs_centroid(neighbors, size, v as usize, u, centroid);
            if size[v as usize] > n as u32 / 2 {
                is_centroid = false;
            }
        }
    }
    if n as u32 - size[u] > n as u32 / 2 {
        is_centroid = false;
    }
    if is_centroid {
        centroid.push(u);
    }
}

fn hash_rooted_tree(neighbors: &[Vec<u32>], root: usize) -> u64 {
    fn rec(neighbors: &[Vec<u32>], u: u32, parent: u32) -> (u64, u32) {
        let mut sub_codes = neighbors[u as usize]
            .iter()
            .filter(|&&v| v != parent)
            .map(|&v| rec(neighbors, v, u))
            .collect::<Vec<_>>();
        sub_codes.sort_unstable();
        let mut res = 0;
        for (c, c_len) in sub_codes {
            res = res << 1 | 1;
            res = res << c_len | c;
            res = res << 1 | 0;
        }
        (res, u64::BITS - res.leading_zeros())
    }

    rec(neighbors, root as u32, root as u32).0
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let mut hashes: HashSet<u64> = HashSet::new();
    for _ in 0..input.value() {
        let n = input.value();
        let mut neighbors = vec![vec![]; n];
        for _ in 0..n - 1 {
            let u = input.value::<usize>();
            let v = input.value::<usize>();
            neighbors[u].push(v as u32);
            neighbors[v].push(u as u32);
        }

        let mut size = vec![0; n];
        dfs_size(&neighbors, &mut size, 0, n);
        let mut centroids = vec![];
        dfs_centroid(&neighbors, &size, 0, n, &mut centroids);

        let hash = centroids
            .iter()
            .map(|&c| hash_rooted_tree(&neighbors, c))
            .min()
            .unwrap();
        hashes.insert(hash);
    }

    let ans = hashes.len();
    writeln!(output, "{}", ans).unwrap();
}
