use std::io::Write;

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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> &'a [T];
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> &'a [T] {
            &self.as_ref()[u]
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len())
                .map(|i| self.get(i).iter().collect())
                .collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> &'a [T] {
            &self.data[self.head[u] as usize..self.head[u + 1] as usize]
        }
    }
}

pub mod graph {
    // General graph matching with blossom algorithm, O(V^3)
    // Adapted from:
    // https://blog.kyouko.moe/20?category=767011
    // https://koosaga.com/258

    use super::jagged;

    pub const UNSET: u32 = !0;
    pub struct MatchingState<'a, J: jagged::Jagged<'a, u32>> {
        neighbors: &'a J,
        parent: Vec<u32>,
        colors: Vec<u32>,
        n_color: u32,

        assignment: Vec<u32>,

        visited: Vec<Option<bool>>,
        orig: Vec<u32>,
        queue: Vec<u32>,
        queue_timer: u32,
    }

    impl<'a, J: jagged::Jagged<'a, u32>> MatchingState<'a, J> {
        fn new(neighbors: &'a J) -> Self {
            let n = neighbors.len();
            Self {
                neighbors,
                parent: vec![UNSET; n],
                colors: vec![UNSET; n],
                n_color: 0,

                assignment: vec![UNSET; n],

                visited: vec![None; n],
                orig: vec![],
                queue: vec![],
                queue_timer: 0,
            }
        }

        fn augment(&mut self, u: u32, mut v: u32) {
            let mut pv;
            let mut nv;
            loop {
                pv = self.parent[v as usize];
                nv = self.assignment[pv as usize];
                self.assignment[v as usize] = pv;
                self.assignment[pv as usize] = v;
                v = nv;
                if u == pv {
                    break;
                }
            }
        }

        fn lca(&mut self, mut v: u32, mut w: u32) -> u32 {
            loop {
                self.n_color += 1;
                loop {
                    if v != UNSET {
                        if self.colors[v as usize] == self.n_color {
                            return v;
                        }
                        self.colors[v as usize] = self.n_color;

                        v = self.assignment[v as usize];
                        if v != UNSET {
                            v = self.parent[v as usize];
                            if v != UNSET {
                                v = self.orig[v as usize];
                            }
                        }
                    }
                    std::mem::swap(&mut v, &mut w);
                }
            }
        }

        fn blossom(&mut self, mut v: u32, mut w: u32, a: u32) {
            while self.orig[v as usize] != a {
                self.parent[v as usize] = w;
                w = self.assignment[v as usize];
                if self.visited[w as usize] == Some(true) {
                    self.visited[w as usize] = Some(false);
                    self.queue.push(w);
                }
                self.orig[v as usize] = a;
                self.orig[w as usize] = a;
                v = self.parent[w as usize];
            }
        }

        fn run(&mut self) -> u32 {
            let n = self.neighbors.len();
            let mut n_matchings = 0;
            'outer: for init in 0..n as u32 {
                if self.assignment[init as usize] != UNSET {
                    continue;
                }

                self.visited.fill(None);
                self.visited[init as usize] = Some(false);
                self.orig = (0..n as u32).collect();
                self.queue = [init].into();
                self.queue_timer = 0;

                while let Some(&u) = self.queue.get(self.queue_timer as usize) {
                    self.queue_timer += 1;
                    for &v in self.neighbors.get(u as usize) {
                        if self.visited[v as usize].is_none() {
                            self.parent[v as usize] = u;
                            self.visited[v as usize] = Some(true);
                            let nv = self.assignment[v as usize];
                            if nv == UNSET {
                                self.augment(init, v);
                                n_matchings += 1;
                                continue 'outer;
                            }
                            self.queue.push(nv);
                            self.visited[nv as usize] = Some(false);
                        } else if self.visited[v as usize] == Some(false)
                            && self.orig[u as usize] != self.orig[v as usize]
                        {
                            let a = self.lca(self.orig[u as usize], self.orig[v as usize]);
                            self.blossom(v, u, a);
                            self.blossom(u, v, a);
                        }
                    }
                }
            }
            n_matchings
        }
    }

    pub fn max_matching<'a, J: jagged::Jagged<'a, u32>>(neighbors: &'a J) -> (u32, Vec<u32>) {
        let mut state = MatchingState::new(neighbors);
        let n_matchings = state.run();
        (n_matchings, state.assignment)
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges = vec![];
    for _ in 0..m {
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        edges.push((u, v));
        edges.push((v, u));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let (max_matching, assignment) = graph::max_matching(&neighbors);

    if cfg!(debug_assertions) {
        for u in 0..n as u32 {
            let v = assignment[u as usize];
            if v != graph::UNSET && u < v {
                println!("{} {}", u + 1, v + 1);
            }
        }
    }
    writeln!(output, "{}", max_matching).unwrap();
}
