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

use std::io::{BufReader, Read /*, Write*/};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn solve(grades: &[u8], n: usize, m: usize) -> u32 {
    const N_MAX: usize = 14;
    const M_MAX: usize = 14;
    assert!((1..=N_MAX).contains(&n));
    assert!((1..=M_MAX).contains(&m));
    assert!(grades.len() >= n * m);

    let p: u32 = 9901;

    let price_table: [[u32; 5]; 5] = [
        [10, 8, 7, 5, 1],
        [8, 6, 4, 3, 1],
        [7, 4, 3, 2, 1],
        [5, 3, 2, 2, 1],
        [1, 1, 1, 1, 0],
    ];
    let grade = |i1, i2| price_table[grades[i1] as usize][grades[i2] as usize];
    // println!("{:?}", grades);

    let mut dp: [[u32; 1 << M_MAX]; M_MAX + 1] = [[0; 1 << M_MAX]; M_MAX + 1];
    let window = m + 1;
    for i in 1..m * n {
        for filled in 0..1 << m {
            dp[i % window][filled] = {
                let mut value = dp[(i - 1) % window][(filled >> 1) | (1 << (m - 1))];
                if i >= m && filled & 1 == 1 {
                    value = value.max((dp[(i - 1) % window][filled >> 1] + grade(i, i - m)) % p);
                }
                if i % m >= 1 && filled & 0b11 == 0b11 {
                    debug_assert!(m >= 2);
                    value = value.max(
                        (dp[(i + window - 2) % window][(filled >> 2) | (0b11 << (m - 2))]
                            + grade(i, i - 1))
                            % p,
                    );
                }
                value
            };
            // println!("{} {:b}: {}", i, filled, dp[i % window][filled]);
        }
        // println!("{}: {}", i, dp[i % window][(1 << m) - 1]);
    }

    dp[(m * n - 1) % window][(1 << m) - 1]
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let m = input.value();
    input.skip_line();

    let grades: Vec<_> = (0..n)
        .map(|_| input.line()[0..m].to_vec())
        .flat_map(|line: Vec<u8>| {
            line.iter()
                .map(|&c| match c {
                    b'A' => 0,
                    b'B' => 1,
                    b'C' => 2,
                    b'D' => 3,
                    b'F' => 4,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    println!("{:?}", solve(&grades[..], n, m));

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
