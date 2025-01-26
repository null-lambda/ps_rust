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

const INF: i32 = (1 << 30) - 1;

// Max-plus algebra
fn mul_mat(n: usize, xs: &[i32], ys: &[i32]) -> Vec<i32> {
    // Cache-friendly matrix multiplication, without much effort
    // https://stackoverflow.com/questions/13312625/cache-friendly-method-to-multiply-two-matrices
    let mut res = vec![-INF; n * n];
    let mut ys_j = vec![-INF; n];
    for j in 0..n {
        for k in 0..n {
            ys_j[k] = ys[k * n + j];
        }
        for i in 0..n {
            for k in 0..n {
                res[i * n + j] = res[i * n + j].max(xs[i * n + k] + ys_j[k]);
            }
        }
    }
    res
}

fn add_mat(n: usize, xs: &[i32], ys: &[i32]) -> Vec<i32> {
    let mut res = vec![-INF; n * n];
    for i in 0..n {
        for j in 0..n {
            res[i * n + j] = xs[i * n + j].max(ys[i * n + j]);
        }
    }
    res
}

fn trace(n: usize, xs: &[i32]) -> i32 {
    let mut res = -INF;
    for i in 0..n {
        res = res.max(xs[i * n + i]);
    }
    res
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut base = vec![-INF; n * n];
    for _ in 0..m {
        let a = input.u32() as usize - 1;
        let b = input.u32() as usize - 1;
        let m: i32 = input.i32();
        let s: i32 = input.i32();
        base[a * n + b] = s - m;
    }

    // Find least k such at score(1 + base^1 + .. + base^k) > 0
    let n_bits = (u64::BITS - u64::leading_zeros(n.next_power_of_two() as u64)) as usize + 1;
    let pows: Vec<_> = std::iter::successors(Some(base.clone()), |last| {
        Some(add_mat(n, &last, &mul_mat(n, last, last)))
    })
    .take(n_bits)
    .collect();

    let mut exp = 1;
    let mut curr = base.clone();
    for bit in (0..n_bits).rev() {
        let next = add_mat(n, &curr, &mul_mat(n, &pows[bit], &curr));
        if trace(n, &next) <= 0 {
            curr = next;
            exp += 1 << bit;
        }
    }

    while trace(n, &curr) <= 0 {
        curr = add_mat(n, &curr, &mul_mat(n, &base, &curr));
        exp += 1;
    }

    writeln!(output, "{} {}", exp, trace(n, &curr)).unwrap();
}
