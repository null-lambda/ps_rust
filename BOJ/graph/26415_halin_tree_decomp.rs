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

pub mod tree_decomp {
    use std::ops::{Index, IndexMut};

    pub const UNSET: u32 = u32::MAX;

    // Small-sized array-vec with none-like value
    #[derive(Clone, Copy)]
    pub struct Bag<const N: usize>(pub [u32; N]);

    impl<const N: usize> Default for Bag<N> {
        fn default() -> Self {
            Self([UNSET; N])
        }
    }

    impl<const N: usize> Bag<N> {
        pub fn get_raw(&self, idx: usize) -> &u32 {
            &self.0[idx]
        }

        pub fn get_raw_mut(&mut self, idx: usize) -> &mut u32 {
            &mut self.0[idx]
        }

        pub fn get(&self, idx: usize) -> Option<&u32> {
            let res = self.get_raw(idx);
            (res != &UNSET).then(|| res)
        }

        pub fn get_mut(&mut self, idx: usize) -> Option<&mut u32> {
            let res = self.get_raw_mut(idx);
            (res != &UNSET).then(|| res)
        }

        pub fn len(&self) -> usize {
            self.iter().count()
        }

        pub fn iter(&self) -> Iter<'_, N> {
            Iter {
                inner: self.0.iter(),
            }
        }

        pub fn iter_mut(&mut self) -> IterMut<'_, N> {
            IterMut {
                inner: self.0.iter_mut(),
            }
        }

        pub fn try_push(&mut self, value: u32) -> Result<usize, ()> {
            for i in 0..N {
                if self.0[i] == UNSET {
                    self.0[i] = value;
                    return Ok(i);
                }
            }
            Err(())
        }

        pub fn push(&mut self, value: u32) {
            self.try_push(value).unwrap();
        }

        pub fn sort(&mut self) {
            self.0.sort_unstable();
        }

        pub fn dedup(&mut self) {
            let mut end = 0;
            let mut prev = UNSET;
            for i in 0..N {
                let x = self.0[i];
                if x == UNSET {
                    break;
                }
                if x != prev {
                    prev = x;
                    self.0[end] = x;
                    end += 1;
                }
            }

            for x in &mut self.0[end..N] {
                *x = UNSET;
            }
        }
    }

    impl<const N: usize> Index<usize> for Bag<N> {
        type Output = u32;

        fn index(&self, index: usize) -> &Self::Output {
            self.get(index).unwrap()
        }
    }

    impl<const N: usize> IndexMut<usize> for Bag<N> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.get_mut(index).unwrap()
        }
    }

    impl<const N: usize, const M: usize> From<[u32; M]> for Bag<N> {
        fn from(value: [u32; M]) -> Self {
            assert!(M <= N);
            let mut this = Self::default();
            for i in 0..M {
                this.0[i] = value[i];
            }
            this
        }
    }

    pub struct IntoIter<const N: usize> {
        inner: std::array::IntoIter<u32, N>,
    }

    pub struct Iter<'a, const N: usize> {
        inner: std::slice::Iter<'a, u32>,
    }

    pub struct IterMut<'a, const N: usize> {
        inner: std::slice::IterMut<'a, u32>,
    }

    impl<const N: usize> Iterator for IntoIter<N> {
        type Item = u32;
        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().filter(|&u| u != UNSET)
        }
    }

    impl<'a, const N: usize> Iterator for Iter<'a, N> {
        type Item = &'a u32;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().filter(|&&u| u != UNSET)
        }
    }

    impl<'a, const N: usize> Iterator for IterMut<'a, N> {
        type Item = &'a mut u32;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().filter(|&&mut u| u != UNSET)
        }
    }

    impl<const N: usize> IntoIterator for Bag<N> {
        type Item = u32;
        type IntoIter = IntoIter<N>;
        fn into_iter(self) -> Self::IntoIter {
            IntoIter {
                inner: self.0.into_iter(),
            }
        }
    }

    impl<'a, const N: usize> IntoIterator for &'a Bag<N> {
        type Item = &'a u32;
        type IntoIter = Iter<'a, N>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }
    impl<'a, const N: usize> IntoIterator for &'a mut Bag<N> {
        type Item = &'a mut u32;
        type IntoIter = IterMut<'a, N>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter_mut()
        }
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let mut children = vec![vec![]; n];
    let mut parent = vec![!0; n];
    for u in 1..n {
        let p = input.u32() - 1;
        children[p as usize].push(u as u32);
        parent[u] = p;
    }

    let mut bfs = vec![0u32];
    let mut timer = 0;
    while let Some(&u) = bfs.get(timer) {
        timer += 1;
        for &v in &children[u as usize] {
            bfs.push(v);
        }
    }

    let mut leaves = vec![];
    for u in 0..n {
        if children[u].is_empty() {
            leaves.push(u as u32);
        }
    }

    const UNSET: u32 = !0;
    type Bag = tree_decomp::Bag<4>;

    let mut next_leaf = vec![UNSET; n];
    for (&u, &v) in leaves.iter().zip(leaves.iter().skip(1)) {
        next_leaf[u as usize] = v;
    }
    let ve = *leaves.last().unwrap();
    next_leaf[ve as usize] = ve;

    let mut bags = vec![];
    let mut bag_edges = vec![];
    let mut top_bag = vec![UNSET; n];
    for &u in bfs.iter().rev() {
        if children[u as usize].is_empty() {
            let r = next_leaf[u as usize];
            debug_assert!(r != UNSET);
            bags.push(Bag::from([u, r]));
        } else {
            for &v in &children[u as usize] {
                let bv = &bags[top_bag[v as usize] as usize];
                if children[v as usize].is_empty() {
                    debug_assert!(bv.len() == 2);
                    let mut bu = bv.clone();
                    bu.push(u);
                    bags.push(bu);

                    let i = bags.len() as u32 - 1;
                    bag_edges.push((top_bag[v as usize], i));
                    top_bag[v as usize] = i;
                } else {
                    debug_assert!(bv.len() == 3);
                    let mut bw = bv.clone();
                    let mut bu = bv.clone();

                    bw.push(u);
                    bu[2] = u;
                    bags.push(bw);
                    bags.push(bu);

                    let i = bags.len() as u32 - 1;
                    bag_edges.push((top_bag[v as usize], i - 1));
                    bag_edges.push((i - 1, i));
                    top_bag[v as usize] = i;
                }
            }

            let v0 = children[u as usize][0];
            for &v1 in &children[u as usize][1..] {
                let bv0 = &bags[top_bag[v0 as usize] as usize];
                let bv1 = &bags[top_bag[v1 as usize] as usize];

                let mut bw = bv0.clone();
                bw.push(bv1[1]);

                let mut br = bv0.clone();
                br[1] = bv1[1];

                bags.push(bw);
                bags.push(br);

                let i = bags.len() as u32 - 1;
                bag_edges.push((top_bag[v0 as usize], i - 1));
                bag_edges.push((top_bag[v1 as usize], i - 1));
                bag_edges.push((i - 1, i));
                top_bag[v0 as usize] = i;
            }
        }
        top_bag[u as usize] = bags.len() as u32 - 1;
    }

    writeln!(output, "{}", bags.len()).ok();
    for mut gs in bags {
        gs.sort();
        gs.dedup();
        write!(output, "{} ", gs.len()).ok();
        for u in gs {
            write!(output, "{} ", u + 1).ok();
        }
        writeln!(output).ok();
    }

    for (b, c) in bag_edges {
        writeln!(output, "{} {}", b + 1, c + 1).ok();
    }
}
