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

const UNSET: u32 = !0;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let head0 = input.value::<usize>() - 1;
    let tail0 = input.value::<usize>() - 1;
    assert!(n >= 2);

    let mut degree = vec![0; n];
    let mut xor_neighbors = vec![0; n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }

    let mut parent;
    let mut topological_order = vec![];
    let (head, tail);
    {
        let mut leaf_in_subtree: Vec<_> = (0..n as u32).collect();

        degree[head0] += 2;

        for mut u in 0..n as u32 {
            while degree[u as usize] == 1 {
                let p = xor_neighbors[u as usize];
                xor_neighbors[p as usize] ^= u;
                degree[u as usize] -= 1;
                degree[p as usize] -= 1;
                topological_order.push(u);

                leaf_in_subtree[p as usize] = leaf_in_subtree[u as usize];

                u = p;
            }
        }
        topological_order.push(head0 as u32);

        parent = xor_neighbors;
        tail = leaf_in_subtree[head0] as usize;
    }

    let level = |mut u: usize| {
        let mut res = 0;
        while u != head0 {
            u = parent[u] as usize;
            res += 1;
        }
        res
    };
    let nth_parent = |mut u: usize, mut n: usize| {
        while n > 0 {
            u = parent[u] as usize;
            n -= 1;
        }
        u
    };

    let len = level(tail0);
    head = nth_parent(tail, len) as usize;

    let mut u = tail as u32;
    let mut roots = vec![tail as u32];
    while u as usize != head0 {
        let p = std::mem::replace(&mut parent[u as usize], u);
        u = p;
    }

    let mut max_depth = vec![0u32; n];
    let mut topological_order = vec![];
    {
        let mut degree = vec![1u32; n];
        for u in 0..n as u32 {
            let p = parent[u as usize];
            if p != u {
                degree[p as usize] += 1;
            } else {
                degree[u as usize] += 2;
            }
        }

        for mut u in 0..n as u32 {
            while degree[u as usize] == 1 {
                let p = parent[u as usize];
                degree[u as usize] -= 1;
                degree[p as usize] -= 1;

                max_depth[p as usize] = max_depth[p as usize].max(max_depth[u as usize] + 1);

                topological_order.push((u, ()));
                u = p;
            }
        }

        for u in 0..n {
            if degree[u] > 0 {
                topological_order.push((u as u32, ()));
            }
        }
    }

    let mut level = vec![0u32; n];
    let mut root = vec![UNSET; n];
    for &(u, ()) in topological_order.iter().rev() {
        let p = parent[u as usize];
        if p == u {
            root[u as usize] = u;
        } else {
            root[u as usize] = root[p as usize];
            level[u as usize] = level[p as usize] + 1;
        }
    }

    debug::with(|| {
        println!("order: {:?}", topological_order);
        println!("parent: {:?}", parent);
        println!("head: {}, tail: {}", head, tail);
        println!("{:?}", root);
    });

    let (nth_parent, lca) = {
        let mut parent_sparse = parent;
        let n_log2 = (usize::BITS - usize::leading_zeros(n.next_power_of_two())) as usize;

        for _ in 0..n_log2 {
            for i in 0..n {
                let prev = &parent_sparse[parent_sparse.len() - n..];
                parent_sparse.push(prev[prev[i] as usize]);
            }
        }
        let parent_sparse = move |exp: usize, u: usize| parent_sparse[n * exp..][u] as usize;

        let root = &root;
        let level = &level;
        move |mut u: usize, mut v: usize| -> Option<usize> {
            if root[u] != root[v] {
                return None;
            }

            if level[u] < level[v] {
                std::mem::swap(&mut u, &mut v);
            }
            let d = level[u] - level[v];

            for exp in (0..n_log2).rev() {
                if (d >> exp) & 1 == 1 {
                    u = parent_sparse(exp, u);
                }
            }

            if u == v {
                return Some(u);
            }
            for exp in (0..n_log2).rev() {
                let u_next = parent_sparse(exp, u);
                let v_next = parent_sparse(exp, v);
                if u_next != v_next {
                    u = u_next;
                    v = v_next;
                }
            }

            Some(parent_sparse(0, u))
        }
    };

    debug::with(|| {
        // for u in 0..n {
        //     for v in 0..n {
        // dbg!(u, v, lca(u, v));
        // }
        // }
    });

    // Step 3.
    for _ in 0..input.value() {
        let h = input.value::<usize>() - 1;
        let t = input.value::<usize>() - 1;

        let ans = if let Some(j) = lca(h, t) {
            max_depth[h] >= level[t] - level[j] || max_depth[t] >= level[h] - level[j]
        } else {
            max_depth[h] >= level[t] || max_depth[t] >= level[h]
        };
        writeln!(output, "{}", if ans { "YES" } else { "NO" }).unwrap();
    }
}
