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
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read};
// use std::io::Write;

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

type Point<T> = (T, T);

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let n: u32 = input.value();
    let mut points: Vec<Point<i64>> = (0..n).map(|_| (input.value(), input.value())).collect();

    // monotone chain algorithm
    assert!(n >= 3);
    points.sort_unstable_by_key(|&(x, y)| (x, y));

    fn ccw(p: Point<i64>, q: Point<i64>, r: Point<i64>) -> bool {
        (q.0 - p.0) * (r.1 - p.1) > (q.1 - p.1) * (r.0 - p.0)
    }

    let mut lower = Vec::<_>::new();
    let mut upper = Vec::<_>::new();
    for &p in &points {
        while matches!(lower.as_slice(), [.., l1, l2] if !ccw(*l1, *l2, p)) {
            lower.pop();
        }
        lower.push(p);
    }
    for &p in points.iter().rev() {
        while matches!(upper.as_slice(), [.., l1, l2] if !ccw(*l1, *l2, p)) {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();

    // println!("{:?}", points);
    // println!("{:?}", lower);
    // println!("{:?}", upper);

    let n_convex_hull = lower.len() + upper.len();
    println!("{}", n_convex_hull);

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
