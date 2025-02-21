use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

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

pub mod bitset {
    type B = u64;
    const BLOCK_SIZE: usize = B::BITS as usize;

    pub struct BitVec {
        blocks: Vec<B>,
    }

    impl BitVec {
        pub fn zeros(n: usize) -> Self {
            Self {
                blocks: vec![0; (n + BLOCK_SIZE - 1) / BLOCK_SIZE],
            }
        }

        pub fn set(&mut self, i: usize, value: bool) {
            let (b, s) = (i / BLOCK_SIZE, i % BLOCK_SIZE);
            self.blocks[b] ^= (-(value as i64) as u64 ^ self.blocks[b]) & (1 << s);
        }

        pub fn get(&self, i: usize) -> bool {
            self.blocks[i / BLOCK_SIZE] & (1 << (i % BLOCK_SIZE)) != 0
        }
    }
}

const INF: i32 = 1 << 30;

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let ws = (0..n * n).map(|_| input.i32()).collect::<Vec<_>>();

    // Prim's algorithm
    let mut pq = BinaryHeap::new();
    let mut visited = bitset::BitVec::zeros(n);
    let mut min_edge = vec![INF; n];

    let mut mst_len = 0;
    pq.push((Reverse(0), 0u32));

    while let Some((Reverse(w), u)) = pq.pop() {
        if visited.get(u as usize) {
            continue;
        }
        visited.set(u as usize, true);
        mst_len += w as i64;
        for v in 0..n {
            let w = ws[u as usize * n + v];
            if !visited.get(v) && w < min_edge[v] {
                min_edge[v] = w;
                pq.push((Reverse(w), v as u32));
            }
        }
    }
    writeln!(output, "{}", mst_len).unwrap();
}
