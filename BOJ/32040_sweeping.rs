use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    io::Write,
};

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

// chunk_by in std >= 1.77
fn group_by<T, P, F>(xs: &[T], mut pred: P, mut f: F)
where
    P: FnMut(&T, &T) -> bool,
    F: FnMut(usize, usize),
{
    let mut i = 0;
    while i < xs.len() {
        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        f(i, j);
        i = j;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Dir {
    N = 0,
    E = 1,
    S = 2,
    W = 3,
}
use Dir::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CollisionDir {
    NE = 0,
    NW = 1,
    SW = 2,
    SE = 3,
    NS = 4,
    EW = 5,
}
use CollisionDir::*;

impl CollisionDir {
    fn all() -> [CollisionDir; 6] {
        [NE, NW, SW, SE, NS, EW]
    }
}

const UNSET: u32 = !0;

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let mut lines = vec![vec![]; 6];
    for i in 0..n {
        let x: i32 = input.i32();
        let y: i32 = input.i32();
        let dir = match input.token() {
            "N" => N,
            "E" => E,
            "S" => S,
            "W" => W,
            _ => unreachable!(),
        };

        for collision_dir in CollisionDir::all() {
            let gid = match collision_dir {
                NE | SW => x - y,
                NW | SE => x + y,
                NS => x * 2 + y % 2,
                EW => y * 2 + x % 2,
            };
            let proj = match collision_dir {
                SE | NW => x - y,
                NE | SW => x + y,
                NS => y,
                EW => x,
            };
            let to_left = match (collision_dir, dir) {
                (SE, S) | (NW, W) => true,
                (SE, E) | (NW, N) => false,
                (NE, N) | (SW, W) => true,
                (NE, E) | (SW, S) => false,
                (NS, N) | (EW, W) => true,
                (NS, S) | (EW, E) => false,
                _ => continue,
            };
            lines[collision_dir as usize].push((gid, proj, to_left, i));
        }
    }

    let mut node_idx = vec![[UNSET; 6]; n];
    let mut links = vec![vec![[UNSET; 2]; n]; 6];
    let mut queue: BinaryHeap<_> = Default::default();
    for cdir in CollisionDir::all() {
        let line = &mut lines[cdir as usize];
        let link = &mut links[cdir as usize];
        line.sort_unstable();

        for (u, &(.., i)) in line.iter().enumerate() {
            node_idx[i][cdir as usize] = u as u32;
        }

        group_by(
            line,
            |a, b| a.0 == b.0,
            |i, j| {
                for k in i..j {
                    if k > i {
                        link[k][0] = k as u32 - 1;
                    }
                    if k + 1 < j {
                        link[k][1] = k as u32 + 1;
                    }
                }

                for k in i + 1..j {
                    let (_, t0, l0, i0) = line[k - 1];
                    let (_, t1, l1, i1) = line[k];
                    if !l0 && l1 {
                        queue.push((Reverse(t1 - t0), i0, i1));
                    }
                }
            },
        );
    }

    let mut destroyed = vec![false; n];
    loop {
        // Pop all invalid events
        while let Some(&(_, iu, iv)) = queue.peek() {
            if !destroyed[iu as usize] && !destroyed[iv as usize] {
                break;
            }
            queue.pop();
        }

        // Find all collided ships
        let mut to_destroy = HashSet::new();
        let Some(&(t0, ..)) = queue.peek() else {
            break;
        };
        while let Some(&(t, iu, iv)) = queue.peek() {
            if t != t0 {
                break;
            }
            queue.pop();

            if destroyed[iu as usize] || destroyed[iv as usize] {
                continue;
            }

            to_destroy.insert(iu);
            to_destroy.insert(iv);
        }

        if to_destroy.is_empty() {
            break;
        }

        // Delete corresponding nodes in linked lists, and add newly formed events to the queue
        for i in to_destroy {
            destroyed[i as usize] = true;

            for cdir in CollisionDir::all() {
                let u = node_idx[i][cdir as usize];
                if u == UNSET {
                    continue;
                }

                let [left, right] = links[cdir as usize][u as usize];
                if left != UNSET {
                    links[cdir as usize][left as usize][1] = right;
                }
                if right != UNSET {
                    links[cdir as usize][right as usize][0] = left;
                }

                if left != UNSET && right != UNSET {
                    let (_, t0, l0, i0) = lines[cdir as usize][left as usize];
                    let (_, t1, l1, i1) = lines[cdir as usize][right as usize];
                    if !l0 && l1 {
                        queue.push((Reverse(t1 - t0), i0, i1));
                    }
                }
            }
        }
    }

    for u in 0..n {
        if !destroyed[u] {
            writeln!(output, "{}", u + 1).unwrap();
        }
    }
}
