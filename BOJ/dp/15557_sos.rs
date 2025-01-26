use std::io::Write;

use buffered_io::InputStream;

mod buffered_io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        fn value<T: FromStr>(&mut self) -> T
        where
            <T as FromStr>::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    // cheap and unsafe whitespace check
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};

    pub struct LineSyncedInput<R: BufRead> {
        line_buf: Vec<u8>,
        line_cursor: usize,
        inner: R,
    }

    impl<R: BufRead> LineSyncedInput<R> {
        pub fn new(r: R) -> Self {
            Self {
                line_buf: Vec::new(),
                line_cursor: 0,
                inner: r,
            }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.line_buf.len() - self.line_cursor);
            let slice = &self.line_buf[self.line_cursor..self.line_cursor + n];
            self.line_cursor += n;
            slice
        }

        fn eol(&self) -> bool {
            self.line_cursor == self.line_buf.len()
        }

        fn refill_line_buf(&mut self) -> bool {
            self.line_buf.clear();
            self.line_cursor = 0;
            let result = self.inner.read_until(b'\n', &mut self.line_buf).is_ok();
            result
        }
    }

    impl<R: BufRead> InputStream for LineSyncedInput<R> {
        fn token(&mut self) -> &[u8] {
            loop {
                if self.eol() {
                    let b = self.refill_line_buf();
                    if !b {
                        panic!(); // EOF
                    }
                }
                self.take(
                    self.line_buf[self.line_cursor..]
                        .iter()
                        .position(|&c| !is_whitespace(c))
                        .unwrap_or_else(|| self.line_buf.len() - self.line_cursor),
                );

                let idx = self.line_buf[self.line_cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.line_buf.len() - self.line_cursor);
                if idx > 0 {
                    return self.take(idx);
                }
            }
        }

        fn line(&mut self) -> &[u8] {
            if self.eol() {
                self.refill_line_buf();
            }

            self.line_cursor = self.line_buf.len();
            trim_newline(self.line_buf.as_slice())
        }
    }

    pub fn stdin() -> LineSyncedInput<BufReader<Stdin>> {
        LineSyncedInput::new(BufReader::new(std::io::stdin()))
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let l: usize = input.value();
    let q: usize = input.value();
    let xs: Vec<_> = input.token().iter().map(|&b| b - b'0').collect();

    let mut sos: Vec<_> = xs.iter().map(|&b| b as u32).collect(); // Sum over subsets
    let mut inv_sos: Vec<_> = sos.clone(); // Sum over supersets
    for i in 0..l {
        let b = 1 << i;
        for mask in 0..1 << l {
            if mask & b != 0 {
                sos[mask] += sos[mask ^ b];
            }
        }
    }

    for i in 0..l {
        let b = 1 << i;
        for mask in 0..1 << l {
            if mask & b == 0 {
                inv_sos[mask] += inv_sos[mask ^ b];
            }
        }
    }

    for _ in 0..q {
        let pattern = input.token();
        let unknown = pattern
            .iter()
            .fold(0, |acc, &b| (acc << 1) | (b == b'?') as u32);
        let ones = pattern
            .iter()
            .fold(0, |acc, &b| (acc << 1) | (b == b'1') as u32);
        let zeros = ((1 << l) - 1) ^ ones ^ unknown;

        let mut ans = 0;
        if unknown.count_ones() <= l.div_ceil(3) as u32 {
            let mut submask = unknown;
            loop {
                ans += xs[(submask | ones) as usize] as i32;
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & unknown;
            }
        } else if ones.count_ones() <= zeros.count_ones() {
            let mut submask = ones;
            loop {
                let parity = if submask.count_ones() % 2 == 0 { 1 } else { -1 };
                ans += (sos[(submask | unknown) as usize] as i32) * parity;
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & ones;
            }
            if ones.count_ones() % 2 == 1 {
                ans = -ans;
            }
        } else {
            let mut submask = zeros;
            loop {
                let parity = if submask.count_ones() % 2 == 0 { 1 } else { -1 };
                ans += (inv_sos[((1 << l) - 1) ^ (submask | unknown) as usize] as i32) * parity;
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & zeros;
            }
            if zeros.count_ones() % 2 == 1 {
                ans = -ans;
            }
        }

        writeln!(output, "{}", ans).unwrap();
    }
}
