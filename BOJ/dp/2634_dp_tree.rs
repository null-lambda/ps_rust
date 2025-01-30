use std::{cmp::Reverse, io::Write, mem::MaybeUninit};

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

fn matchable(mut ds: Vec<u32>, max_len: u32) -> bool {
    assert!(ds.len() % 2 == 0);
    ds.sort_unstable();

    while let Some(x) = ds.pop() {
        let Some(i) = (0..ds.len()).rposition(|i| x + ds[i] <= max_len) else {
            return false;
        };
        ds.remove(i);
    }

    true
}

fn match_paths(mut ds: Vec<Vec<u32>>, root: u32, max_len: u32) -> Option<Vec<Vec<u32>>> {
    assert!(ds.len() % 2 == 0);
    ds.sort_unstable_by_key(|path| path.len());
    let mut joined = vec![];
    // print!("ds = {:?}    ", ds);
    while let Some(mut path) = ds.pop() {
        let Some(i) = (0..ds.len()).rposition(|i| path.len() + ds[i].len() <= max_len as usize)
        else {
            return None;
        };
        let other = ds.remove(i);
        path.push(root);
        path.extend(other.into_iter().rev());
        joined.push(path);
    }
    // println!("joined: {:?}", joined);
    Some(joined)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];
    for _ in 1..n {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }

    if !degree.iter().all(|&d| d % 2 == 0 || d == 1) {
        writeln!(output, "0").unwrap();
        return;
    }

    degree[0] += 2;
    let mut topological_order = vec![];
    for mut u in 0..n {
        while degree[u] == 1 {
            let p = xor_neighbors[u];
            degree[p as usize] -= 1;
            degree[u] -= 1;
            xor_neighbors[p as usize] ^= u as u32;
            topological_order.push((u as u32, p));
            u = p as usize;
        }
    }

    let satistiable = |max_len: u32| {
        let mut sub_depths = vec![vec![]; n];

        'outer: for &(u, p) in &topological_order {
            if sub_depths[u as usize].is_empty() {
                sub_depths[p as usize].push(1);
            } else {
                let mut ds = std::mem::take(&mut sub_depths[u as usize]);
                ds.sort_unstable();

                for i in 0..ds.len() {
                    ds.swap(0, i);

                    if ds[0] <= max_len && matchable(ds[1..].to_vec(), max_len) {
                        sub_depths[p as usize].push(ds[0] + 1);
                        continue 'outer;
                    }

                    ds.swap(0, i);
                }

                return false;
            }
        }

        let mut ds = std::mem::take(&mut sub_depths[0]);
        ds.sort_unstable();
        if ds.len() % 2 == 1 {
            ds[0] <= max_len && matchable(ds[1..].to_vec(), max_len)
        } else {
            matchable(ds, max_len)
        }
    };

    let construct_path = |max_len: u32| {
        let mut sub_paths = vec![vec![]; n];
        let mut paths: Vec<Vec<u32>> = vec![];
        'outer: for &(u, p) in &topological_order {
            if sub_paths[u as usize].is_empty() {
                // println!("sub_paths: {:?}", [u]);
                sub_paths[p as usize].push(vec![u]);
            } else {
                let mut ds = std::mem::take(&mut sub_paths[u as usize]);
                // println!("paths: {paths:?}, sub_paths: {ds:?}");
                ds.sort_unstable_by_key(|path| path.len());

                for i in 0..ds.len() {
                    ds.swap(0, i);

                    if let Some(joined) = match_paths(ds[1..].to_vec(), u, max_len) {
                        ds[0].push(u);
                        sub_paths[p as usize].push(ds[0].clone());
                        paths.extend(joined);
                        continue 'outer;
                    }

                    ds.swap(0, i);
                }

                panic!()
            }
        }

        let mut ds = std::mem::take(&mut sub_paths[0]);
        ds.sort_unstable();
        if ds.len() % 2 == 1 {
            for i in 0..ds.len() {
                ds.swap(0, i);

                if let Some(joined) = match_paths(ds[1..].to_vec(), 0, max_len) {
                    ds[0].push(0);
                    paths.push(ds[0].clone());
                    paths.extend(joined);
                    break;
                }

                ds.swap(0, i);
            }
        } else {
            paths.extend(match_paths(ds, 0, max_len).unwrap());
        }
        paths
    };

    let (mut left, mut right) = (1, n - 1);
    while left < right {
        let mid = (left + right) / 2;
        if !satistiable(mid as u32) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    let ans = left;
    writeln!(output, "{}", ans).unwrap();

    let ps = construct_path(ans as u32);
    writeln!(output, "{}", ps.len()).unwrap();
    for path in ps {
        // write!(output, "{} ", path.len()).unwrap();
        for u in path {
            write!(output, "{} ", u + 1).unwrap();
        }
        writeln!(output).unwrap();
    }
}
