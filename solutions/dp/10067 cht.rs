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

    let n = input.value();
    let k_bound = input.value();

    use std::iter::once;
    let sum: Vec<i64> = once(0)
        .chain((0..n).scan(0, |acc, _| {
            let x: i32 = input.value();
            *acc += x as i64;
            Some(*acc)
        }))
        .collect();
    let mut dp: [Vec<i64>; 2] = [vec![0; n + 1], vec![0; n + 1]];
    let mut split_index: Vec<Vec<u32>> = (0..=k_bound).map(|_| vec![0; n + 1]).collect();

    for k in 1..=k_bound {
        let mut cvhull: Vec<(f64, u32, (i64, i64))> = vec![];
        dp[k % 2][0] = 0;
        for i in 1..=n {
            let mut x = -1.0;
            let line = (dp[(k - 1) % 2][i - 1] - sum[i - 1] * sum[i - 1], sum[i - 1]);
            (|| {
                while let Some(&(x_last, _, last)) = cvhull.last() {
                    if last.1 - line.1 == 0 {
                        if line.0 < last.0 {
                            return;
                        }
                    } else {
                        x = (line.0 - last.0) as f64 / (last.1 - line.1) as f64;
                        if x_last < x {
                            break;
                        }
                    }
                    cvhull.pop();
                }
                cvhull.push((x, (i - 1) as u32, line));
            })();
            // println!("{:?}", cvhull);

            let (_, j, line) = cvhull[cvhull.partition_point(|&(x, ..)| x <= sum[i] as f64) - 1];
            split_index[k][i] = j;
            dp[k % 2][i] = line.0 + sum[i] * line.1;
        }
    }

    writeln!(output_buf, "{}", dp[k_bound % 2][n]).unwrap();
    (1..=k_bound)
        .rev()
        .scan(n, |j, k| {
            *j = split_index[k][*j] as usize;
            Some(*j)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .for_each(|i| {
            write!(output_buf, "{} ", i).unwrap();
        });

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
