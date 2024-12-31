use std::io::Write;

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

pub mod centroid {
    // Centroid Decomposition

    fn reroot_on_edge(size: &mut [u32], u: usize, p: usize) {
        size[p] -= size[u];
        size[u] += size[p];
    }

    fn find_centroid<E>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &[bool],
        n_half: u32,
        path: &mut Vec<u32>,
        u: usize,
        p: usize,
    ) -> usize {
        path.push(u as u32);
        for &(v, _) in &neighbors[u] {
            if v as usize == p || visited[v as usize] {
                continue;
            }
            if size[v as usize] > n_half {
                reroot_on_edge(size, v as usize, u);
                return find_centroid(neighbors, size, visited, n_half, path, v as usize, u);
            }
        }
        u
    }

    fn update_size<E>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &[bool],
        u: usize,
        p: usize,
    ) {
        size[u] = 1;
        for &(v, _) in &neighbors[u] {
            if v as usize == p || visited[v as usize] {
                continue;
            }
            update_size(neighbors, size, visited, v as usize, u);
            size[u] += size[v as usize];
        }
    }

    pub fn init_size<E>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &mut [bool],
        init: usize,
    ) {
        update_size(neighbors, size, visited, init, init); // TODO
    }

    pub fn dnc<E, F>(
        neighbors: &[Vec<(u32, E)>],
        size: &mut [u32],
        visited: &mut [bool],
        rooted_solver: &mut F,
        init: usize,
    ) where
        F: FnMut(&[Vec<(u32, E)>], &[u32], &[bool], usize),
    {
        println!("init: {:?}", size);

        update_size(neighbors, size, visited, init, init);
        let mut path = vec![];
        let root = find_centroid(
            neighbors,
            size,
            visited,
            size[init] / 2,
            &mut path,
            init,
            init,
        );

        visited[root] = true;
        rooted_solver(neighbors, size, visited, root);

        for &(v, _) in &neighbors[root] {
            if visited[v as usize] {
                continue;
            }
            dnc(neighbors, size, visited, rooted_solver, v as usize);
        }

        loop {
            match &path[..] {
                [.., p, u] => reroot_on_edge(size, *p as usize, *u as usize),
                _ => break,
            }
            path.pop();
        }
        path.clear();
    }
}

const INF: u32 = u32::MAX / 3;

fn update_ans(
    neighbors: &[Vec<(u32, u32)>],
    visited: &[bool],
    k: u32,
    dp: &mut [u32],
    ans: &mut u32,
    u: usize,
    p: usize,
    depth: u32,
    dist: u32,
) {
    if dist > k {
        return;
    }
    *ans = (*ans).min(depth + dp[(k - dist) as usize]);
    for &(v, w) in &neighbors[u] {
        if v as usize == p || visited[v as usize] {
            continue;
        }
        update_ans(
            neighbors,
            visited,
            k,
            dp,
            ans,
            v as usize,
            u,
            depth + 1,
            dist + w as u32,
        );
    }
}

fn update_dp(
    neighbors: &[Vec<(u32, u32)>],
    visited: &[bool],
    k: u32,
    dp: &mut [u32],
    u: usize,
    p: usize,
    depth: u32,
    dist: u32,
) {
    if dist > k {
        return;
    }
    dp[dist as usize] = dp[dist as usize].min(depth);
    for &(v, w) in &neighbors[u] {
        if v as usize == p || visited[v as usize] {
            continue;
        }
        update_dp(
            neighbors,
            visited,
            k,
            dp,
            v as usize,
            u,
            depth + 1,
            dist + w as u32,
        );
    }
}

fn clear_dp(
    neighbors: &[Vec<(u32, u32)>],
    visited: &[bool],
    k: u32,
    dp: &mut [u32],
    u: usize,
    p: usize,
    dist: u32,
) {
    if dist > k {
        return;
    }
    dp[dist as usize] = INF;
    for &(v, w) in &neighbors[u] {
        if v as usize == p || visited[v as usize] {
            continue;
        }
        clear_dp(neighbors, visited, k, dp, v as usize, u, dist + w);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: u32 = input.value();

    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u: u32 = input.value();
        let v: u32 = input.value();
        let w: u32 = input.value();
        neighbors[u as usize].push((v, w));
        neighbors[v as usize].push((u, w));
    }

    let mut dp = vec![INF; k as usize + 1];
    let mut ans = INF;

    let mut size = vec![0; n];
    centroid::init_size(&neighbors, &mut size, &mut vec![false; n], 0);
    centroid::dnc(
        &neighbors,
        &mut size,
        &mut vec![false; n],
        &mut |neighbors, _size, visited, root| {
            dp[0] = 0;
            for &(v, w) in &neighbors[root] {
                if visited[v as usize] {
                    continue;
                }
                update_ans(
                    neighbors, visited, k, &mut dp, &mut ans, v as usize, v as usize, 1, w,
                );
                update_dp(neighbors, visited, k, &mut dp, v as usize, v as usize, 1, w);
            }
            clear_dp(neighbors, visited, k, &mut dp, root, root, 0);
        },
        0,
    );
    if ans == INF {
        writeln!(output, "-1").unwrap();
    } else {
        writeln!(output, "{}", ans).unwrap();
    }
}
