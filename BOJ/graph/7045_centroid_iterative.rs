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

// pub mod tree_iso {
//     // AHU algorithm for testing whether two rooted trees are isomorphic
//     pub fn ahu_label_compression(
//         n_verts: usize,
//         edges: impl IntoIterator<Item = (u32, u32)>,
//         rooted: bool,
//     ) -> (u32, impl Iterator<Item = Box<[u32]>>) {
//         assert!(n_verts > 0);

//         let mut root = 0;
//         let mut degree = vec![0u32; n_verts];
//         let mut xor_neighbors = vec![0u32; n_verts];
//         for (u, v) in edges {
//             debug_assert!(u < n_verts as u32);
//             debug_assert!(v < n_verts as u32);
//             degree[u as usize] += 1;
//             degree[v as usize] += 1;
//             xor_neighbors[u as usize] ^= v;
//             xor_neighbors[v as usize] ^= u;
//         }
//         degree[root] += 2;

//         let mut size = vec![1u32; n_verts];
//         let mut topological_order = vec![];
//         for mut u in 0..n_verts as u32 {
//             while degree[u as usize] == 1 {
//                 let p = xor_neighbors[u as usize];
//                 xor_neighbors[p as usize] ^= u;
//                 degree[u as usize] -= 1;
//                 degree[p as usize] -= 1;
//                 topological_order.push(u);
//                 u = p;
//             }
//         }
//         let parent = xor_neighbors;

//         // If not rooted, then reroot to the centroid
//         if !rooted {
//             let threshold = (n_verts as u32 + 1) / 2;
//             for u in topological_order.into_iter().rev() {
//                 let p = parent[u as usize] as usize;
//                 if p as usize == root && size[u as usize] >= threshold {
//                     size[p as usize] -= size[u as usize];
//                     size[u as usize] += size[p as usize];
//                     root = u as usize;
//                 }
//             }
//         }

//         (todo!(), None.into_iter())
//     }

//     // pub fn ahu_label_compression_rooted()
// }

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n_verts: usize = input.value();
    let edges = (0..n_verts - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1));

    let mut root = 0;
    let mut degree = vec![0u32; n_verts];
    let mut xor_neighbors = vec![0u32; n_verts];
    for (u, v) in edges {
        debug_assert!(u < n_verts as u32);
        debug_assert!(v < n_verts as u32);
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }
    degree[root] += 2;

    let mut size = vec![1u32; n_verts];
    let mut topological_order = vec![];
    for mut u in 0..n_verts as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            xor_neighbors[p as usize] ^= u;
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            topological_order.push(u);
            size[p as usize] += size[u as usize];
            u = p;
        }
    }
    let parent = xor_neighbors;

    let threshold = (n_verts as u32 + 1) / 2;
    for u in topological_order.into_iter().rev() {
        let p = parent[u as usize] as usize;
        if p as usize != root {
            continue;
        }
        if size[u as usize] >= threshold {
            size[p as usize] -= size[u as usize];
            size[u as usize] += size[p as usize];
            root = u as usize;
        }
    }
    let mut roots = vec![root];

    let p = parent[root];
    if p != root as u32 && size[p as usize] >= threshold {
        roots.push(p as usize);
    }
    roots.sort_unstable();

    for &r in &roots {
        writeln!(output, "{}", r + 1).unwrap();
    }
}
