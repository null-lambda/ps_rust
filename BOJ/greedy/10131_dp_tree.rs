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

#[derive(Clone, Default)]
struct NodeAgg {
    size_m2: u32,
    delay: i64,
    subtrees: Vec<(u32, i64)>,
}

impl NodeAgg {
    fn singleton(delay: i64) -> Self {
        Self {
            size_m2: 0,
            delay,
            subtrees: vec![],
        }
    }

    fn pull_from(&mut self, child: &Self) {
        self.subtrees.push((child.size_m2, child.delay + 1));
    }

    fn finalize(&mut self) {
        self.subtrees
            .sort_unstable_by_key(|&(s, d)| s as i64 - d as i64);
        self.size_m2 = 0;
        for &(s, d) in &self.subtrees {
            self.delay = self.delay.max(self.size_m2 as i64 + d);
            self.size_m2 += s;
        }
        self.size_m2 += 2;

        // println!("size_m2 = {}, delay = {}", self.size_m2, self.delay);
    }

    fn collapse(&self) -> i64 {
        self.delay
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let xs = (0..n).map(|_| input.value());
    let mut dp: Vec<_> = xs.into_iter().map(NodeAgg::singleton).collect();
    let base = dp[0].delay;

    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];
    for _ in 0..n - 1 {
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize] ^= v;
        xor_neighbors[v as usize] ^= u;
    }
    degree[0] += 2;

    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize] ^= u;

            let mut dp_u = std::mem::take(&mut dp[u as usize]);
            dp_u.finalize();
            dp[p as usize].pull_from(&dp_u);

            u = p;
        }
    }
    dp[0].finalize();
    let ans = (base + 2 * (n as i64 - 1)).max(dp[0].collapse());
    writeln!(output, "{}", ans).unwrap();
}
