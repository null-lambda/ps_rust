use std::{io::Write, vec};

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

const UNSET: u32 = u32::MAX;

// Build a max cartesian tree from inorder traversal
fn max_cartesian_tree<T>(
    n: usize,
    iter: impl IntoIterator<Item = (usize, T)>,
) -> (Vec<u32>, Vec<[u32; 2]>, usize)
where
    T: Ord,
{
    let mut parent = vec![UNSET; n];
    let mut children = vec![[UNSET; 2]; n];

    // Monotone stack
    let mut stack = vec![];

    for (u, h) in iter {
        let u = u as u32;

        let mut c = None;
        while let Some((prev, _)) = stack.last() {
            if prev > &h {
                break;
            }
            c = stack.pop();
        }
        if let Some(&(_, p)) = stack.last() {
            parent[u as usize] = p;
            children[p as usize][1] = u;
        }
        if let Some((_, c)) = c {
            parent[c as usize] = u;
            children[u as usize][0] = c;
        }
        stack.push((h, u));
    }
    let root = stack[0].1 as usize;

    (parent, children, root)
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let hs: Vec<i32> = (0..n).map(|_| input.u32() as i32).collect();
    let (parent, children, root) = max_cartesian_tree(n, hs.iter().copied().enumerate());

    let mut bfs = vec![root as u32];
    let mut timer = 0;
    while let Some(&u) = bfs.get(timer) {
        timer += 1;
        for c in children[u as usize] {
            if c != UNSET {
                bfs.push(c);
            }
        }
    }

    let mut size = vec![1i32; n];
    for &u in bfs[1..].iter().rev() {
        size[parent[u as usize] as usize] += size[u as usize];
    }

    let mut virtual_opponent = vec![0; n];

    for &u in &bfs {
        let cs = children[u as usize];
        let cs_size: [_; 2] = std::array::from_fn(|b| {
            if cs[b] == UNSET {
                0
            } else {
                size[cs[b] as usize]
            }
        });

        for b in 0..2 {
            if cs[b] != UNSET {
                virtual_opponent[cs[b] as usize] =
                    hs[u as usize].max(virtual_opponent[u as usize] - cs_size[b ^ 1] - 1);
            }
        }

        for b in 0..2 {
            if cs[b] != UNSET {
                virtual_opponent[u as usize] -= cs_size[b];
            }
        }
    }

    if n >= 2 {
        for i in 0..n {
            if i > 0 && hs[i - 1] < hs[i as usize] || i + 1 < n && hs[i + 1] < hs[i as usize] {
                continue;
            }
            virtual_opponent[i] = hs[i];
        }
    }

    let mut ans = vec![];
    for u in 0..n {
        if hs[u] > virtual_opponent[u] {
            ans.push(u as u32);
        }
    }

    if ans.is_empty() {
        writeln!(output, "-1").ok();
    } else {
        for u in ans {
            write!(output, "{} ", u + 1).ok();
        }
    }
}
