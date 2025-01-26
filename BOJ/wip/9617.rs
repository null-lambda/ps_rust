use std::{cmp::Reverse, collections::HashMap, io::Write};

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

#[derive(Clone, Default, Debug)]
struct HashMultiSet<K> {
    freq: HashMap<K, u32>,
}

impl<K: Eq + std::hash::Hash> HashMultiSet<K> {
    fn new() -> Self {
        HashMultiSet {
            freq: HashMap::new(),
        }
    }

    fn len_unique(&self) -> usize {
        self.freq.len()
    }

    fn len(&self) -> usize {
        self.freq.values().sum::<u32>() as usize
    }

    fn insert(&mut self, key: K) {
        *self.freq.entry(key).or_insert(0) += 1;
    }

    fn remove(&mut self, key: &K) -> bool {
        if let Some(f) = self.freq.get_mut(key) {
            *f -= 1;
            if *f == 0 {
                self.freq.remove(key);
            }
            true
        } else {
            false
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.freq.contains_key(key)
    }
}

fn test_digit(n_verts: usize, edges: &[(u32, u32, i32)], digit: u8) -> Option<i32> {
    let res = match (digit, edges, n_verts) {
        (1, &[(0, 1, w)], 2) if w % 2 == 0 => Some(w / 2 - 1),
        (2 | 5, &[(0, 1, w)], 2) if w % 5 == 0 => Some(w / 5 - 1),
        (7, &[(0, 1, w)], 2) if w % 3 == 0 => Some(w / 3 - 1),
        (3, &[(0, 1, w1), (0, 2, w2), (0, 3, w3)], 4) if w1 == w2 && w2 == w3 * 2 => Some(w3 - 1),
        (4, &[(0, 1, w1), (0, 2, w2), (0, 3, w3)], 4) if w1 == w2 * 2 && w2 == w3 => Some(w3 - 1),
        (6 | 9, &[(0, 1, a), (0, 1, b), (0, 2, c)], 3) if a + b == 2 * c && c % 2 == 0 => {
            Some(c / 2 - 1)
        }
        (8, &[(0, 1, a1), (0, 1, a2), (0, 1, w)], 2) if a1 == w * 3 && a2 == w * 3 => Some(w - 1),
        _ => None,
    };
    res.filter(|&rank| rank >= 1)
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    for i_tc in 1..=input.value() {
        let n: usize = input.value();
        let m: usize = input.value();
        let mut neighbors = vec![HashMultiSet::new(); n];
        for _ in 0..m {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            neighbors[u as usize].insert((v, 1));
            neighbors[v as usize].insert((u, 1));
        }

        let mut erased = vec![false; n];
        let mut queue: Vec<_> = (0..n as u32)
            .filter(|&u| neighbors[u as usize].len_unique() == 2)
            .collect();
        let mut timer = 0;
        while let Some(&u) = queue.get(timer) {
            timer += 1;
            if neighbors[u as usize].len_unique() != 2 || neighbors[u as usize].len() != 2 {
                continue;
            }

            let mut neighbors_u = neighbors[u as usize].freq.iter();
            let (&(v1, w1), _) = neighbors_u.next().unwrap();
            let (&(v2, w2), _) = neighbors_u.next().unwrap();
            if v1 == v2 {
                continue;
            }

            drop(neighbors_u);
            neighbors[u as usize] = Default::default();
            erased[u as usize] = true;

            neighbors[v1 as usize].remove(&(u, w1));
            neighbors[v2 as usize].remove(&(u, w2));
            neighbors[v1 as usize].insert((v2, w1 + w2));
            neighbors[v2 as usize].insert((v1, w1 + w2));
            for v in [v1, v2] {
                if neighbors[v as usize].len_unique() == 2 && neighbors[v as usize].len() == 2 {
                    queue.push(v);
                }
            }
        }

        let mut verts: Vec<_> = (0..n as u32).filter(|&u| !erased[u as usize]).collect();
        verts.sort_by_cached_key(|&u| {
            let mut weights_u = vec![];
            for (&(_, w), &f) in &neighbors[u as usize].freq {
                for _ in 0..f {
                    weights_u.push(w);
                }
            }
            weights_u.sort_unstable_by_key(|&w| Reverse(w));

            Reverse((
                neighbors[u as usize].len(),
                neighbors[u as usize].len_unique(),
                weights_u,
            ))
        });

        const UNSET: u32 = !0;
        let mut vert_map = vec![UNSET; n];
        let mut n_compressed = 0;
        for u in verts {
            vert_map[u as usize] = n_compressed;
            n_compressed += 1;
        }

        let mut edges = vec![];
        for u in 0..n {
            for (&(v, w), &f) in &neighbors[u].freq {
                assert!(u != v as usize);
                let nu = vert_map[u];
                let nv = vert_map[v as usize];
                if nu < nv {
                    for _ in 0..f {
                        edges.push((vert_map[u as usize], vert_map[v as usize], w));
                    }
                }
            }
        }
        edges.sort_unstable_by_key(|&(u, v, w)| (u, v, Reverse(w)));

        let matches: Vec<_> = (0..=9)
            .flat_map(|digit| {
                test_digit(n_compressed as usize, &edges, digit).map(|rank| (digit, rank))
            })
            .collect();
        writeln!(output, "Case {}: {}", i_tc, matches.len()).unwrap();
        for (digit, rank) in matches {
            writeln!(output, "{} {}", digit, rank).unwrap();
        }
        writeln!(output).unwrap();
    }
}

/*

5
16 15
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
10 11
11 12
12 13
13 14
14 15
15 16

4 3
1 2
1 3
1 4

9 8
1 2
2 3
3 4
4 5
5 6
6 7
5 8
8 9


18 18
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
 10 11
 11 12
 12 13
 13 14
 14 15
 15 16
 16 17
 17 18
 18 7


13 14
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
 10 11
 11 12
 12 1
 5 13
 13 11

*/
