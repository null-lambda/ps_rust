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

pub mod cht {
    use core::{num::NonZeroU32, ops::RangeInclusive};

    // max Li-Chao tree of lines
    // TODO: add segment insertion
    type V = f64;
    type K = f64;
    const NEG_INF: V = f64::MIN;
    const EPS: f64 = 1e-9;

    #[derive(Clone)]
    pub struct Line {
        slope: V,
        intercept: V,
    }

    impl Line {
        pub fn new(slope: V, intercept: V) -> Self {
            Self { slope, intercept }
        }

        fn eval(&self, x: K) -> V {
            self.slope * x as f64 + self.intercept
        }

        fn bottom() -> Self {
            Self {
                slope: 0.0,
                intercept: NEG_INF,
            }
        }
    }

    #[derive(Clone)]
    struct NodeRef(NonZeroU32);

    struct Node {
        children: [Option<NodeRef>; 2],
        line: Line,
    }

    impl Node {
        fn new() -> Self {
            Self {
                children: [None, None],
                line: Line::bottom(),
            }
        }
    }

    pub struct LiChaoTree {
        pool: Vec<Node>,
        interval: RangeInclusive<K>,
    }

    impl LiChaoTree {
        pub fn new(interval: RangeInclusive<K>) -> Self {
            Self {
                pool: vec![Node::new()],
                interval,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let index = self.pool.len();
            self.pool.push(node);
            NodeRef(NonZeroU32::new(index as u32).unwrap())
        }

        // pub fn insert_segment(&mut self, interval: (V, V), mut line: Line) {
        //     unimplemented!()
        // }

        pub fn insert(&mut self, mut line: Line) {
            let mut u = 0;
            let (mut x_left, mut x_right) = self.interval.clone().into_inner();
            loop {
                let x_mid = (x_left + x_right) / 2.0;
                let top = &mut self.pool[u].line;
                if top.eval(x_mid) < line.eval(x_mid) {
                    std::mem::swap(top, &mut line);
                }
                u = if top.eval(x_left) < line.eval(x_left) {
                    x_right = x_mid;
                    match self.pool[u].children[0] {
                        Some(ref c) => c.0.get() as usize,
                        None => {
                            let c = self.alloc(Node::new());
                            self.pool[u].children[0] = Some(c.clone());
                            c.0.get() as usize
                        }
                    }
                } else if top.eval(x_right) < line.eval(x_right) {
                    x_left = x_mid + 1.0;
                    match self.pool[u].children[1] {
                        Some(ref c) => c.0.get() as usize,
                        None => {
                            let c = self.alloc(Node::new());
                            self.pool[u].children[1] = Some(c.clone());
                            c.0.get() as usize
                        }
                    }
                } else {
                    return;
                };
            }
        }

        pub fn eval(&self, x: K) -> V {
            debug_assert!(self.interval.contains(&x));
            let mut u = 0;
            let mut result = self.pool[u].line.eval(x);
            let (mut x_left, mut x_right) = self.interval.clone().into_inner();
            loop {
                let x_mid = (x_left + x_right) / 2.0;
                let branch = if x <= x_mid {
                    x_right = x_mid;
                    0
                } else {
                    x_left = x_mid;
                    1
                };
                if let Some(c) = &self.pool[u].children[branch] {
                    u = c.0.get() as usize;
                } else {
                    return result;
                }
                result = result.max(self.pool[u].line.eval(x));
            }
        }
    }
}

const T_MAX: f64 = 1e18;

fn partition_point_f64<P>(
    mut left: f64,
    mut right: f64,
    eps: f64,
    mut max_iter: u32,
    mut pred: P,
) -> f64
where
    P: FnMut(f64) -> bool,
{
    while right - left > eps && max_iter > 0 {
        let mid = left + (right - left) / 2.0;
        if pred(mid) {
            left = mid;
        } else {
            right = mid;
        }
        max_iter -= 1;
    }
    left
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: i64 = input.value();
    let p: usize = input.value();
    let c: f64 = input.value();
    let mut pills = vec![];
    for _ in 0..p {
        let t: i64 = input.value();
        let x: f64 = input.value();
        let y: f64 = input.value();
        let slope = -y / x;
        pills.push((t, slope));
    }
    pills.sort_unstable_by_key(|&(t, _)| t);

    let mut hull = cht::LiChaoTree::new(0.0..=T_MAX);
    hull.insert(cht::Line::new(-1.0, n as f64));
    let mut ans = 0.0f64;
    for (t, slope) in pills {
        let hp = hull.eval(t as f64);
        let y0 = hp - slope * t as f64 - c;
        hull.insert(cht::Line::new(slope, y0));

        let lifetime = partition_point_f64(t as f64, T_MAX as f64, 1e-9, 100, |t| {
            hull.eval(t as f64) >= 0.0
        });
        if hull.eval(lifetime) >= 0.0 {
            ans = ans.max(lifetime);
        }
    }
    writeln!(output, "{:.9}", ans).unwrap();
}
