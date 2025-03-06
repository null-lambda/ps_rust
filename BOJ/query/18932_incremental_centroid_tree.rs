use std::{cell::Cell, io::Write};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }

    #[cfg(debug_assertions)]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(T);

    #[cfg(not(debug_assertions))]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(std::marker::PhantomData<T>);

    impl<T> Label<T> {
        #[inline]
        pub fn new_with(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(value())
            }
            #[cfg(not(debug_assertions))]
            {
                Self(Default::default())
            }
        }

        pub fn with(&mut self, #[allow(unused_variables)] f: impl FnOnce(&mut T)) {
            #[cfg(debug_assertions)]
            f(&mut self.0)
        }
    }

    impl<T: std::fmt::Debug> std::fmt::Debug for Label<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            #[cfg(debug_assertions)]
            {
                write!(f, "{:?}", self.0)
            }
            #[cfg(not(debug_assertions))]
            {
                write!(f, "()")
            }
        }
    }
}

const UNSET: u32 = !0;

fn expand_vec_with<T>(xs: &mut Vec<T>, n: usize, f: impl FnMut() -> T) {
    if xs.len() < n {
        xs.resize_with(n, f);
    }
}

#[derive(Clone, Debug)]
struct CentroidNode {
    parent: u32,
    level: u32,
    max_level: u32,
    old_max_level: u32,
    size: u32,

    depth_freq: Vec<u32>,
    depth_freq_subtree: Vec<Vec<u32>>,
}

impl CentroidNode {
    fn singleton(u: usize) -> Self {
        Self {
            parent: UNSET,
            level: 0,
            max_level: 0,
            old_max_level: 0,
            size: 1,

            depth_freq: vec![1],
            depth_freq_subtree: vec![],
        }
    }
}

#[derive(Clone, Debug)]
struct IncrementalCentroidTree {
    neighbors: Vec<Vec<u32>>,

    nodes: Vec<CentroidNode>,

    depth_in_layer: Vec<Vec<u32>>,
    branch_in_layer: Vec<Vec<u32>>,

    n_rebuilds: u32,
    acc_rebuild_size: u64,
    last_query_depth: Cell<u32>,
}

impl IncrementalCentroidTree {
    fn new(n_verts: usize) -> Self {
        Self {
            neighbors: vec![vec![]; n_verts],

            nodes: (0..n_verts).map(CentroidNode::singleton).collect(),

            depth_in_layer: vec![vec![0; n_verts]],
            branch_in_layer: vec![vec![UNSET; n_verts]],

            n_rebuilds: 0,
            acc_rebuild_size: 0,
            last_query_depth: 0.into(),
        }
    }

    fn n_verts(&self) -> usize {
        self.neighbors.len()
    }

    fn expand_layers(&mut self, l_max: usize) {
        let n = self.n_verts();
        expand_vec_with(&mut self.depth_in_layer, l_max + 1, || vec![UNSET; n]);
        expand_vec_with(&mut self.branch_in_layer, l_max + 1, || vec![UNSET; n]);
    }

    fn find_root_centroid(&self, mut u: usize) -> usize {
        loop {
            let p = self.nodes[u].parent as usize;
            if p == UNSET as usize {
                return u;
            }
            u = p;
        }
    }

    fn bfs(
        &mut self,
        u: usize,
        p: usize,
        mut visitor: impl FnMut(&mut Self, usize, usize),
        mut filter: impl FnMut(&mut Self, usize) -> bool,
    ) -> Vec<(u32, u32)> {
        if !filter(self, u) {
            return vec![];
        }
        let mut bfs = vec![(u as u32, p as u32)];
        let mut timer = 0;
        while let Some(&(u, p)) = bfs.get(timer) {
            timer += 1;
            visitor(self, u as usize, p as usize);
            for iv in 0..self.neighbors[u as usize].len() {
                let v = self.neighbors[u as usize][iv];
                if v == p || !filter(self, v as usize) {
                    continue;
                }
                bfs.push((v, u));
            }
        }

        bfs
    }

    fn weak_dfs(
        &mut self,
        u: usize,
        p: usize,
        mut visitor: impl FnMut(&mut Self, usize, usize),
        mut filter: impl FnMut(&mut Self, usize) -> bool,
    ) {
        if !filter(self, u) {
            return;
        }
        let mut stack = vec![(u as u32, p as u32)];
        while let Some((u, p)) = stack.pop() {
            visitor(self, u as usize, p as usize);
            for iv in 0..self.neighbors[u as usize].len() {
                let v = self.neighbors[u as usize][iv];
                if v == p || !filter(self, v as usize) {
                    continue;
                }
                stack.push((v, u));
            }
        }
    }

    fn init_centroid_decomp(&mut self, u: usize) {
        let l = self.nodes[u].level;
        let bfs = self.bfs(
            u,
            UNSET as usize,
            |this, b, _| {
                // this.nodes[b].parent = UNSET;
                this.nodes[b].level = UNSET;

                this.nodes[b].depth_freq.clear();
                this.nodes[b].depth_freq.push(1);

                this.nodes[b]
                    .depth_freq_subtree
                    .resize_with(this.neighbors[b].len(), || vec![]);
                for g in &mut this.nodes[b].depth_freq_subtree {
                    g.clear();
                }

                this.nodes[b as usize].size = 1;

                // debug::with(|| println!("init_centroid_decomp {b} - {:?}", this.nodes[b]));
            },
            |this, b| this.nodes[b].level >= l,
        );

        for &(u, p) in bfs.iter().skip(1).rev() {
            self.nodes[p as usize].size += self.nodes[u as usize].size;
        }
    }

    fn reroot_to_centroid(&mut self, u: &mut usize) {
        // let threshold = (self.nodes[*u].size + 1) / 2;
        let threshold = self.nodes[*u].size / 2 + 1;
        let mut p = UNSET as usize;
        'outer: loop {
            for &v in &self.neighbors[*u] {
                if v as usize == p || self.nodes[v as usize].level != UNSET {
                    continue;
                }

                if self.nodes[v as usize].size >= threshold {
                    // debug::with(|| println!("reroot {u} -> {v}"));
                    self.nodes[*u].size -= self.nodes[v as usize].size;
                    self.nodes[v as usize].size += self.nodes[*u].size;

                    p = *u;
                    *u = v as usize;
                    continue 'outer;
                }
            }
            break;
        }
    }

    fn rebuild(&mut self, mut u: usize, p: usize) -> usize {
        if self.nodes[u].size >= 64 {
            self.reroot_to_centroid(&mut u);
        }
        self.nodes[u].parent = p as u32;
        self.nodes[u].level = if p == UNSET as usize {
            0
        } else {
            self.nodes[p as usize].level + 1
        };
        self.nodes[u].max_level = self.nodes[u].level;

        let l = self.nodes[u].level as usize;
        self.expand_layers(l);

        self.depth_in_layer[l][u] = 0;
        self.branch_in_layer[l][u] = UNSET;
        for iv in 0..self.neighbors[u].len() {
            let v = self.neighbors[u][iv] as usize;
            if v == p || self.nodes[v].level != UNSET {
                continue;
            }

            self.bfs(
                v,
                u,
                |this, b, a| {
                    this.depth_in_layer[l][b] = this.depth_in_layer[l][a] + 1;
                    this.branch_in_layer[l][b] = iv as u32;
                    let d = this.depth_in_layer[l][b] as usize;

                    expand_vec_with(&mut this.nodes[u].depth_freq, d + 1, || 0);
                    expand_vec_with(&mut this.nodes[u].depth_freq_subtree[iv], d + 1, || 0);
                    this.nodes[u].depth_freq[d] += 1;
                    this.nodes[u].depth_freq_subtree[iv][d] += 1;
                },
                |this, b| this.nodes[b].level == UNSET,
            );

            let rv = self.rebuild(v, u);
            self.nodes[u].max_level = self.nodes[u].max_level.max(self.nodes[rv].max_level);
        }
        self.nodes[u].old_max_level = self.nodes[u].max_level;

        self.n_rebuilds += 1;
        self.acc_rebuild_size += self.nodes[u].size as u64;
        u
    }

    fn link_naive(&mut self, mut u: usize, mut v: usize) {
        {
            let ru = self.find_root_centroid(u);
            let rv = self.find_root_centroid(v);
            if self.nodes[ru].size < self.nodes[rv].size {
                std::mem::swap(&mut u, &mut v);
            }
        }

        // debug::with(|| println!("link_naive {} -> {}", u, v));

        self.neighbors[u].push(v as u32);
        self.neighbors[v].push(u as u32);

        self.bfs(
            v,
            UNSET as usize,
            |this, u, _| this.nodes[u].level = UNSET,
            |_, _| true,
        );

        self.init_centroid_decomp(u);
        self.rebuild(u, UNSET as usize);

        self.debug_print(|| "link_naive".into(), true);
    }

    fn link(&mut self, mut u: usize, mut v: usize) {
        let mut ru = self.find_root_centroid(u);
        let mut rv = self.find_root_centroid(v);
        if self.nodes[ru].size < self.nodes[rv].size {
            std::mem::swap(&mut u, &mut v);
            std::mem::swap(&mut ru, &mut rv);
        }

        // debug::with(|| println!("link {} -> {}", u, v));

        let size_v = self.nodes[rv].size;
        self.nodes[rv].parent = u as u32;

        let mut path_u = vec![];
        let mut c = u;
        loop {
            path_u.push(c);
            c = self.nodes[c].parent as usize;
            if c == UNSET as usize {
                break;
            }
        }
        let mut rebuild_at = None;
        for &b in path_u.iter().rev() {
            if (self.nodes[b].old_max_level * 16).max(64) < self.nodes[rv].max_level {
                rebuild_at = Some(b);
                break;
            }
            self.nodes[b].size += size_v;
        }

        let lu = self.nodes[u].level as usize;
        let bfs_v = self.bfs(v, u, |_, _, _| {}, |_, _| true);

        self.neighbors[u].push(v as u32);
        self.neighbors[v].push(u as u32);
        self.nodes[u].depth_freq_subtree.push(vec![]);
        self.nodes[v].depth_freq_subtree.push(vec![]);

        let c0 = if let Some(rebuild_at) = rebuild_at {
            for &(b, _) in &bfs_v {
                let b = b as usize;
                // debug::with(|| println!("reset u {b}"));
                self.nodes[b].level = UNSET;
                self.branch_in_layer[lu][b] = UNSET;
            }

            self.init_centroid_decomp(rebuild_at);
            self.rebuild(rebuild_at, self.nodes[rebuild_at].parent as usize)
        } else {
            let l_base = self.nodes[u].level as usize;
            for &(b, _) in &bfs_v {
                let b = b as usize;
                // debug::with(|| println!("update u {b}"));

                let l_prev = self.nodes[b].level as usize;
                self.nodes[b].level += l_base as u32 + 1;
                self.nodes[b].max_level += l_base as u32 + 1;
                let l = self.nodes[b].level as usize;
                self.expand_layers(l);

                for l_sub in (0..=l_prev).rev() {
                    self.depth_in_layer[l_sub + l_base + 1][b] = self.depth_in_layer[l_sub][b];
                    self.branch_in_layer[l_sub + l_base + 1][b] = self.branch_in_layer[l_sub][b];
                }
            }

            // for &(b, _) in &bfs_v {
            //     let b = b as usize;
            //     debug::with(|| println!("reset u {b}"));
            //     self.nodes[b].level = UNSET;
            //     self.branch_in_layer[lu][b] = UNSET;
            // }
            // self.init_centroid_decomp(rv);
            // self.rebuild(rv, u);

            rv
        };

        // Update ancestors
        let mut p = self.nodes[c0].parent as usize;
        if p != UNSET as usize {
            let lp0 = self.nodes[p].level as usize;
            let iv = if rebuild_at.is_none() || rebuild_at == Some(rv) {
                self.neighbors[u].len() - 1
            } else {
                self.branch_in_layer[lp0][u] as usize
            };

            // debug::with(|| {
            //     println!("lp0 {lp0} c0 {c0} iv {iv} //  p {p}, c0 {c0} // rebuild_at {rebuild_at:?} // bfs_v {bfs_v:?}")
            // });

            // self.depth_in_layer[lp0][p] = 0;
            for &(b, a) in &bfs_v {
                let (b, a) = (b as usize, a as usize);
                self.branch_in_layer[lp0][b] = iv as u32;
                self.depth_in_layer[lp0][b] = self.depth_in_layer[lp0][a] + 1;

                let d = self.depth_in_layer[lp0][b] as usize;
                expand_vec_with(&mut self.nodes[p].depth_freq, d + 1, || 0);
                expand_vec_with(&mut self.nodes[p].depth_freq_subtree[iv], d + 1, || 0);
                self.nodes[p].depth_freq[d] += 1;
                self.nodes[p].depth_freq_subtree[iv][d] += 1;
            }

            self.debug_print(
                || format!("link {u} {v}, c0 {c0} on path {path_u:?}, mid-update, bfs_v {bfs_v:?}"),
                false,
            );

            let ml = self.nodes[c0].max_level;
            loop {
                self.nodes[p].max_level = self.nodes[p].max_level.max(ml);

                let c = p;
                p = self.nodes[p].parent as usize;
                if p == UNSET as usize {
                    break;
                }

                let l = self.nodes[p].level as usize;
                let iv = self.branch_in_layer[l][c] as usize;

                for &(b, a) in &bfs_v {
                    let (b, a) = (b as usize, a as usize);
                    self.branch_in_layer[l][b] = iv as u32;
                    self.depth_in_layer[l][b] = self.depth_in_layer[l][a] + 1;
                    // self.depth_in_layer[l][b] =
                    //     self.depth_in_layer[l][u] + self.depth_in_layer[lp0 as usize][b];

                    let d = self.depth_in_layer[l][b] as usize;
                    expand_vec_with(&mut self.nodes[p].depth_freq, d + 1, || 0);
                    expand_vec_with(&mut self.nodes[p].depth_freq_subtree[iv], d + 1, || 0);
                    self.nodes[p].depth_freq[d] += 1;
                    self.nodes[p].depth_freq_subtree[iv][d] += 1;
                }
            }
        }

        self.debug_print(
            || format!("link {u} {v}, rebuilt at {rebuild_at:?} on path {path_u:?}"),
            true,
        );
    }

    fn query(&self, u: usize, k: usize) -> u32 {
        let mut res = 0;
        let mut c = u;
        let mut l = self.nodes[c].level as usize;

        self.last_query_depth.set(0);
        loop {
            let d = self.depth_in_layer[l][u] as usize;
            if k >= d {
                let d_twin = k - d;
                res += self.nodes[c].depth_freq.get(d_twin).unwrap_or(&0);

                if u != c {
                    let iu = self.branch_in_layer[l][u] as usize;
                    res -= self.nodes[c].depth_freq_subtree[iu]
                        .get(d_twin as usize)
                        .unwrap_or(&0);
                }
            }

            c = self.nodes[c].parent as usize;
            debug_assert!((l == 0) == (c == UNSET as usize));
            if c == UNSET as usize {
                break res;
            }
            l -= 1;

            self.last_query_depth.set(self.last_query_depth.get() + 1);
        }
    }

    fn debug_print(&self, msg: impl FnOnce() -> String, query: bool) {
        debug::with(|| {
            // println!();
            // println!("## Debug print - {}", msg());

            // println!(
            //     "neighbors: {:?}",
            //     self.neighbors.iter().enumerate().collect::<Vec<_>>()
            // );

            // for u in 0..self.n_verts() {
            //     println!("nodes[{u}] = {:?}", self.nodes[u]);
            // }
            // println!();

            // let f = |u: u32| {
            //     if u == UNSET {
            //         format!("!")
            //     } else {
            //         format!("{}", u)
            //     }
            // };
            // for l in 0..self.depth_in_layer.len() {
            //     print!("depth[layer = {l}]: ");
            //     for u in 0..self.n_verts() {
            //         print!("{} ", f(self.depth_in_layer[l][u]));
            //     }
            //     println!();
            // }
            // println!();

            // for l in 0..self.depth_in_layer.len() {
            //     print!("branch[layer = {l}]: ");
            //     for u in 0..self.n_verts() {
            //         print!("{} ", f(self.branch_in_layer[l][u]));
            //     }
            //     println!();
            // }
            // println!();

            // for u in 0..self.n_verts() {
            //     assert_eq!(
            //         self.neighbors[u].len(),
            //         self.nodes[u].depth_freq_subtree.len()
            //     );
            // }

            // if query {
            //     for u in 0..self.n_verts() {
            //         print!("freq[{u}]: ");
            //         for k in 0..self.n_verts() {
            //             print!("{} ", self.query(u, k));
            //         }
            //         println!();
            //     }
            // }

            // println!();
            // println!();
        });
    }
}

#[test]
fn randomized_test() {
    use rand::prelude::*;
    use std::time::{Duration, Instant};

    let mut rng_seed = rand::rng();
    let seed = rng_seed.next_u64();

    let seed = 7805547733949099381;
    println!("## Seed: ");
    println!("{:?}", seed);
    println!();

    let mut rng = StdRng::seed_from_u64(seed);
    // let n = 10_000;
    // let q = 200_000;

    let n = 100_000;
    let q = 200_000;

    // let n = 30;
    // let q = 30;

    // let n = 16;
    // let q = 10;

    // let n = 8;
    // let q = 2;

    let mut ans = 0;
    let mut queries = vec![];
    {
        for u in 1..n {
            let p = rng.random_range(0..u);
            // let p = u - 1;
            // let p = 0;
            // let p = (u - 1) / 2;
            queries.push(("1", u, p));
        }
        for _ in 0..q {
            let u = rng.random_range(0..n);
            let k = rng.random_range(0..n + 1);
            queries.push(("2", u, k));
        }
        queries.shuffle(&mut rng);
        // println!("queries: {:?}", queries);
    }

    let mut reference = vec![];
    {
        let mut ct = IncrementalCentroidTree::new(n);
        for &(cmd, u, v) in &queries {
            match cmd {
                // "1" => ct.link_naive(u, v),
                "1" => ct.link(u, v),
                _ => {
                    ans = ct.query(u, v);
                    reference.push(ans);
                }
            }
        }

        debug::with(|| {
            for u in 0..n {
                print!("ds[{u}]: ");
                for k in 0..n {
                    print!("{} ", ct.query(u, k));
                }
                println!();
            }
            println!();
        });
    }

    let mut seq = vec![];
    let mut d_sum = 0u64;
    let mut d_count = 0u64;

    let mut t_link: Duration = Default::default();
    let mut t_query: Duration = Default::default();
    {
        println!();
        println!();
        println!();
        println!("# adaptive rebuild");
        let mut ct = IncrementalCentroidTree::new(n);
        for &(cmd, u, v) in &queries {
            let start = Instant::now();
            match cmd {
                "1" => {
                    ct.link(u, v);
                    t_link += start.elapsed();
                }
                _ => {
                    ans = ct.query(u, v);
                    seq.push(ans);

                    d_sum += ct.last_query_depth.get() as u64;
                    d_count += 1;

                    t_query += start.elapsed();
                }
            }

            debug::with(|| {
                // for u in 0..n {
                //     print!("ds[{u}]: ");
                //     for k in 0..n {
                //         print!("{} ", ct.query(u, k));
                //     }
                //     println!();
                // }
                // println!();
            });
        }

        println!();
        println!("## Stats");
        println!("mean t_link: {:?}", t_link / n as u32);
        println!("mean t_query: {:?}", t_query / q as u32);
        println!("n_rebuilds: {:?}", ct.n_rebuilds);
        println!(
            "mean rebuild size: {:.2}",
            ct.acc_rebuild_size as f32 / ct.n_rebuilds as f32
        );
        println!("mean query depth: {:.2}", d_sum as f32 / d_count as f32);
    }
    assert_eq!(reference, seq);

    // std::fs::write("./to.txt", std::format!("{:?}", reference)).unwrap();
    std::hint::black_box(ans);
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut ct = IncrementalCentroidTree::new(n);
    let queries = (0..q)
        .map(|_| (input.token(), input.u32(), input.u32()))
        .collect::<Vec<_>>();

    let mut ans = 0;
    for (cmd, a, b) in queries {
        let u = (a + ans) as usize % n;
        let v = (b + ans) as usize % n;
        match cmd {
            "1" => ct.link(u, v),
            _ => {
                ans = ct.query(u, v);
                writeln!(output, "{}", ans).unwrap();
            }
        }
    }
}
