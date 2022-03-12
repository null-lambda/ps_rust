mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let y_max: f64 = input.value();
    let y_min: f64 = input.value();
    let n: usize = input.value();
    let mut segs: Vec<(usize, (i32, i32))> = (0..n)
        .map(|i| (i, (input.value(), input.value())))
        .collect();
    use std::cmp::Reverse;
    segs.sort_unstable_by_key(|&(_, (x_upper, x_lower))| (Reverse(x_upper - x_lower), x_upper));
    segs.dedup_by_key(|&mut (_, (x_upper, x_lower))| x_upper - x_lower);

    let mut cvhull = Vec::new();
    'main: for &(i, seg) in segs.iter() {
        let mut t = 0.0;
        while let Some(&(_, t_last, last)) = cvhull.last() {
            let ((u1, l1), (u2, l2)) = (last, seg);
            t = (l1 - l2) as f64 / ((u2 - l2) - (u1 - l1)) as f64;
            if t_last < t {
                break;
            }
            cvhull.pop();
        }
        cvhull.push((i, t, seg));
    }

    let n_queries = input.value();
    for _ in 0..n_queries {
        let y: f64 = input.value();
        let t_query: f64 = (y - y_min) / (y_max - y_min) + 1e-10;
        let (i, ..) = cvhull[cvhull.partition_point(|&(_, t, _)| t <= t_query) - 1];
        writeln!(output_buf, "{}", i + 1).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
