use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

#[derive(Clone, Copy, Default)]
struct NodeDp {
    count: u32,
    end: u32,
}

struct BucketDecomp {
    bucket_base: usize,
    bucket_cap: usize,
    n_buckets: usize,

    tail: Vec<u32>,
    xs_sorted: Vec<u32>,
    dp: Vec<NodeDp>,
}

fn gen_head(bucket_cap: usize) -> impl Fn(usize) -> usize {
    move |b| b * bucket_cap
}

impl BucketDecomp {
    fn new(n: usize, xs: impl Iterator<Item = u32>) -> Self {
        let bucket_base: usize = ((n as f64).sqrt() as usize).max(4);
        let bucket_cap = bucket_base * 2;
        let n_buckets = n.div_ceil(bucket_base);

        let head = gen_head(bucket_cap);
        let mut tail = (0..n).map(|b| head(b) as u32).collect::<Vec<_>>();

        let mut xs_sorted = vec![0u32; head(n_buckets)];
        let dp = vec![NodeDp::default(); head(n_buckets)];

        let mut i = 0;
        let mut b = 0;
        for x in xs {
            xs_sorted[tail[b] as usize] = x;
            tail[b] += 1;

            i += 1;
            if i == bucket_base {
                i = 0;
                b += 1;
            }
        }

        let mut this = Self {
            bucket_base,
            bucket_cap,
            n_buckets,

            tail,
            xs_sorted,
            dp,
        };
        for b in 0..n_buckets {
            this.rebuild_bucket(b);
        }
        this
    }
    fn rebuild(&mut self) {
        let head = gen_head(self.bucket_cap);

        let mut k = self.xs_sorted.len();
        for b in (0..self.n_buckets).rev() {
            for i in (head(b)..self.tail[b] as usize).rev() {
                k -= 1;
                self.xs_sorted[k] = self.xs_sorted[i];
            }
        }

        let mut i = 0;
        let mut b = 0;
        while k < self.xs_sorted.len() {
            self.xs_sorted[self.tail[b] as usize] = self.xs_sorted[k];
            self.tail[b] += 1;
            k += 1;

            i += 1;
            if i == self.bucket_base {
                i = 0;
                b += 1;
            }
        }
        for b in 0..self.n_buckets {
            self.rebuild_bucket(b);
        }
    }
    fn rebuild_bucket(&mut self, b: usize) {
        let view = gen_head(self.bucket_cap)(b)..self.tail[b] as usize;

        let mut count = 0;
        let mut start;

        // todo!()
    }

    fn add(&mut self, x: u32) {
        let head = gen_head(self.bucket_cap);
        let b = (0..self.n_buckets)
            .take_while(|&b| self.xs_sorted[head(b)] <= x)
            .last()
            .unwrap_or(0);
        let i = (head(b)..self.tail[b] as usize)
            .find(|&i| self.xs_sorted[i] > x)
            .unwrap_or(self.tail[b] as usize);

        self.xs_sorted[self.tail[b] as usize] = x;
        self.xs_sorted[i..self.tail[b] as usize + 1].rotate_right(1);
        self.tail[b] += 1;

        if self.tail[b] as usize - head(b) == self.bucket_cap {
            self.rebuild();
        } else {
            self.rebuild_bucket(b);
        }
    }
    fn remove(&mut self, x: u32) {
        let head = gen_head(self.bucket_cap);
        let b = (0..self.n_buckets)
            .take_while(|&b| self.xs_sorted[head(b)] <= x)
            .last()
            .unwrap_or(0);
        let i = (head(b)..self.tail[b] as usize)
            .find(|&i| self.xs_sorted[i] == x)
            .unwrap();
        self.xs_sorted[i..self.tail[b] as usize].rotate_left(1);
        self.tail[b] -= 1;

        self.rebuild_bucket(b);
    }
    fn query(&self) -> u32 {
        todo!()
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let l: u32 = input.value();
    let m: usize = input.value();
}
