use std::{collections::HashSet, io::Write};

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

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut neighbors = vec![HashSet::new(); n];
    for _ in 0..m {
        let u = input.u32() - 1;
        let v = input.u32() - 1;
        neighbors[u as usize].insert(v);
        neighbors[v as usize].insert(u);
    }

    let mut queue: Vec<_> = (0..n as u32)
        .filter(|&u| neighbors[u as usize].len() == 2)
        .collect();
    let mut timer = 0;
    while let Some(&u) = queue.get(timer) {
        timer += 1;
        if neighbors[u as usize].len() != 2 {
            continue;
        }

        let mut neighbors_u = neighbors[u as usize].drain();
        let v1 = neighbors_u.next().unwrap();
        let v2 = neighbors_u.next().unwrap();
        drop(neighbors_u);
        neighbors[v1 as usize].remove(&u);
        neighbors[v2 as usize].remove(&u);
        neighbors[v1 as usize].insert(v2);
        neighbors[v2 as usize].insert(v1);
        if neighbors[v1 as usize].len() == 2 {
            queue.push(v1);
        }
        if neighbors[v2 as usize].len() == 2 {
            queue.push(v2);
        }
    }

    let edges: HashSet<_> = neighbors
        .into_iter()
        .enumerate()
        .flat_map(|(u, neighbors)| neighbors.into_iter().map(move |v| (u as u32, v)))
        .map(|(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();
    let ans = edges.len() == 1;

    writeln!(output, "{}", if ans { "Yes" } else { "No" }).unwrap();
}
