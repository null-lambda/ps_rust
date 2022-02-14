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

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn sum_connected_components(segments: &[(i32, i32)]) -> u64 {
    let n = segments.len();
    if n == 0 {
        return 0;
    }
    let mut total = 0;
    let (mut start_current, mut end_current) = segments[0];
    for &(start, end) in segments[1..].iter() {
        if start <= end_current {
            // merge two segs
            end_current = end_current.max(end);
        } else {
            // update connected component
            total += (end_current - start_current) as u64;
            start_current = start;
            end_current = end;
        }
    }
    total += (end_current - start_current) as u64;
    total
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let m: u64 = input.value();
    let mut inv_segments: Vec<(i32, i32)> = (0..n)
        .map(|_| (input.value(), input.value()))
        .filter_map(|(start, end)| (start > end).then(|| (end, start)))
        .collect();
    inv_segments.sort_by_key(|&(start, _end)| start);

    let result = 2 * sum_connected_components(&inv_segments) + m;
    writeln!(output_buf, "{}", result).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
