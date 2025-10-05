use buffered_io::BufReadExt;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Option<Self> {
            let mut seed = 0;
            unsafe { (std::arch::x86_64::_rdrand64_step(&mut seed) == 1).then(|| Self(seed)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            debug_assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            if n == 0 {
                return;
            }

            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

pub struct EmulatedInteractor<'a> {
    xs: Vec<u32>,
    n_query_calls: u32,
    rng: &'a mut rand::SplitMix64,
}

impl<'a> EmulatedInteractor<'a> {
    pub fn new(n: usize, rng: &'a mut rand::SplitMix64) -> Self {
        let mut xs: Vec<_> = (0..n as u32).collect();
        rng.shuffle(&mut xs);
        Self {
            xs,
            n_query_calls: 0,
            rng,
        }
    }

    pub fn apply(&mut self, i: usize, j: usize) -> Result<bool, ()> {
        let should_swap = self.xs[i] > self.xs[j];
        if should_swap {
            self.xs.swap(i, j);
        }

        if self.xs.windows(2).all(|w| w[0] <= w[1]) {
            return Err(());
        }

        self.n_query_calls += 1;
        if self.n_query_calls % (self.xs.len() as u32 * 2) == 0 {
            let i = self.rng.range_u64(0..self.xs.len() as u64) as usize;
            let shift = self.rng.range_u64(1..self.xs.len() as u64) as usize;
            let j = (i + shift) % self.xs.len();
            self.xs.swap(i, j);
        }

        if self.n_query_calls > 10000 {
            panic!()
        }

        Ok(should_swap)
    }
}

pub struct Interactor<'a, R, W> {
    input: &'a mut R,
    output: &'a mut W,
    n_query_calls: u32,
}

impl<'a, R: buffered_io::BufReadExt, W: std::io::Write> Interactor<'a, R, W> {
    pub fn new(input: &'a mut R, output: &'a mut W) -> Self {
        Self {
            input,
            output,
            n_query_calls: 0,
        }
    }

    pub fn apply(&mut self, i: usize, j: usize) -> Result<bool, ()> {
        writeln!(self.output, "{} {}", i + 1, j + 1).unwrap();
        self.output.flush().unwrap();
        self.n_query_calls += 1;

        let res = self.input.token();
        if res == "WIN" {
            return Err(());
        }
        Ok(res == "SWAPPED")
    }
}

fn run(f: impl FnOnce() -> Result<(), ()>) {
    f().unwrap_err()
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();

    let mut itr = Interactor::new(&mut input, &mut output);

    // let mut rng = rand::SplitMix64::from_entropy().unwrap();
    // let mut itr = EmulatedInteractor::new(n, &mut rng);

    run(|| {
        loop {
            let mut rem = 2 * n as u32;

            for i in 1..n {
                itr.apply(i - 1, i)?;
                rem -= 1;
            }

            for i in (1..n - 1).rev() {
                itr.apply(i - 1, i)?;
                rem -= 1;
            }

            while rem > 0 {
                itr.apply(0, 1)?;
                rem -= 1;
            }
        }
    })
}
