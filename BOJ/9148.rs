mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
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

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
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

    pub trait ReadValue<T> {
        fn value(&mut self) -> T;
    }

    impl<T: FromStr, I: InputStream> ReadValue<T> for I
    where
        T::Err: Debug,
    {
        #[inline]
        fn value(&mut self) -> T {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }
}

use std::fmt::Write;
use std::io::{BufReader, Read};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn hash((a, b, c): (i32, i32, i32)) -> usize {
    (20 * ((20 * a) + b) + c) as usize
}

const HASH_MAX: usize = 21 * 21 * 21;

fn w(cache: &mut [i32; HASH_MAX], (a, b, c): (i32, i32, i32)) -> i32 {
    if a <= 0 || b <= 0 || c <= 0 {
        return 1;
    } else if a > 20 || b > 20 || c > 20 {
        return w(cache, (20, 20, 20));
    }
    let h = hash((a, b, c));
    let result = cache[h];
    if result != -1 {
        return result;
    }
    let result = if a < b && b < c {
        w(cache, (a, b, c - 1)) + w(cache, (a, b - 1, c - 1)) - w(cache, (a, b - 1, c))
    } else {
        w(cache, (a - 1, b, c)) + w(cache, (a - 1, b - 1, c)) + w(cache, (a - 1, b, c - 1))
            - w(cache, (a - 1, b - 1, c - 1))
    };
    cache[h] = result;
    result
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = String::new();

    let mut cache = [-1; HASH_MAX];
    loop {
        let (a, b, c) = (input.value(), input.value(), input.value());
        if (a, b, c) == (-1, -1, -1) {
            break;
        }
        writeln!(
            &mut output_buf,
            "w({}, {}, {}) = {}",
            a,
            b,
            c,
            w(&mut cache, (a, b, c))
        )
        .unwrap();
    }
    print!("{}", output_buf);
}
