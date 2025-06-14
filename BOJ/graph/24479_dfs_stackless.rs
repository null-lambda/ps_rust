use std::{cmp::Reverse, io::Write};

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

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let r = input.u32() as usize - 1;

    // CSR representation of an adjacency list
    let mut edges = Vec::with_capacity(m);
    let mut heads = vec![0u32; n + 1];
    for _ in 0..m {
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        heads[u as usize] += 1;
        heads[v as usize] += 1;
        edges.push([u, v]);
    }

    for u in 0..n {
        heads[u + 1] += heads[u];
    }

    let mut links = vec![0u32; m * 2];
    for [u, v] in edges {
        heads[u as usize] -= 1;
        links[heads[u as usize] as usize] = v as u32;

        heads[v as usize] -= 1;
        links[heads[v as usize] as usize] = u as u32;
    }
    for u in 0..n {
        links[heads[u] as usize..heads[u + 1] as usize].sort_unstable_by_key(|&v| Reverse(v));
    }
    let neighbors = |u: usize| &links[heads[u] as usize..heads[u + 1] as usize];

    // Stackless DFS
    const UNSET: u32 = !0;

    let mut current_edge: Vec<_> = (0..n).map(|u| neighbors(u).len() as u32).collect();
    let mut parent = vec![UNSET; n];
    let mut u = r;

    parent[u] = u as u32;
    let mut t_in = vec![0u32; n];
    let mut timer = 1;
    t_in[u] = timer;

    loop {
        let p = parent[u] as usize;
        let iv = &mut current_edge[u];

        if *iv as usize == 0 {
            if p == u as usize {
                break;
            }
            u = p;
            continue;
        }

        *iv -= 1;
        let v = neighbors(u)[*iv as usize] as usize;
        if v == p {
            continue;
        }

        if parent[v] == UNSET {
            timer += 1;
            t_in[v] = timer;
            parent[v] = u as u32;
            u = v;
        }
    }

    for t in t_in {
        writeln!(output, "{}", t).unwrap();
    }
}
