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

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn id_matrix(n: usize) -> Vec<Vec<i32>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1 } else { 0 }).collect())
        .collect()
}

fn clone_matrix(n: usize, a: &Vec<Vec<i32>>, out: &mut Vec<Vec<i32>>) {
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][j];
        }
    }
}

fn mul_matrix(n: usize, a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>, out: &mut Vec<Vec<i32>>) {
    for i in 0..n {
        for j in 0..n {
            out[i][j] = (0..n).map(|l| a[i][l] * b[l][j]).sum();
            out[i][j] %= 1000;
        }
    }
}

fn pow_matrix(n: usize, a: &Vec<Vec<i32>>, mut exponent: u64, out: &mut Vec<Vec<i32>>) {
    debug_assert!(exponent >= 1);

    *out = id_matrix(n);
    let mut a: Vec<Vec<_>> = a.clone();
    let mut temp: Vec<Vec<_>> = a.clone();

    while exponent > 0 {
        if exponent % 2 == 0 {
            mul_matrix(n, &a, &a, &mut temp);
            clone_matrix(n, &temp, &mut a);
            exponent /= 2;
        } else {
            mul_matrix(n, &a, out, &mut temp);
            clone_matrix(n, &temp, out);
            exponent -= 1;
        }
    }
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let (n, b) = (input.value(), input.value());
    let a = (0..n)
        .map(|_| (0..n).map(|_| input.value()).collect())
        .collect();

    let mut result = id_matrix(n);
    pow_matrix(n, &a, b, &mut result);

    let s = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| result[i][j].to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    writeln!(output_buf, "{}", s).unwrap();

    std::io::stdout().write(&output_buf[..]).unwrap();
}
