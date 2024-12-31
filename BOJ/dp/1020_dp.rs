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
            .map(|&c| matches! { c, b'\n' | b'\r' | 0  })
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
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn solve(n: usize, x: i64) -> i64 {
    assert!(n <= 15);

    const INF: i64 = 1_000_000_000_000_000;
    const N_MAX: usize = 15;

    let segments = [6, 2, 5, 5, 4, 5, 6, 3, 7, 5];
    const S_MIN: i64 = 2;
    const S_MAX: i64 = 7;

    // memoization - table[digits][segments]
    let mut table = [[INF; (S_MAX as usize) * N_MAX + 1]; N_MAX + 1];

    table[0][0] = 0;

    // for x = 237,
    // check 238 239.
    // if failed, update table[digits=1] for later calculation.
    // check 24x 25x ... 29x
    // if failed, update table[digits=2] for later calculation.
    // check 3xx 4xx ... 9xx
    // check from the beginning - 0xx 1xx 2xx
    // done!

    let mut s_target: i64 = 0;
    let s_range = |d| (S_MIN * (d as i64)..=S_MAX * (d as i64));
    for d in 1..=n {
        let pow_10 = 10i64.pow((d - 1) as u32);
        let dth_digit = (x / pow_10) % 10;
        s_target += segments[dth_digit as usize];

        let check_solution = |leading_coeff| {
            let s_remains = s_target
                .checked_sub(segments[leading_coeff as usize])
                .unwrap_or(INF);

            if s_range(d - 1).contains(&s_remains) && table[d - 1][s_remains as usize] < INF {
                let x_truncated = (x / (pow_10 * 10)) * pow_10 * 10;
                let x1 = x_truncated + leading_coeff * pow_10;
                return Some(table[d - 1][s_remains as usize] + x1 - x);
            }
            None
        };

        if let Some(result) = ((dth_digit + 1)..=9)
            .flat_map(|lc| check_solution(lc))
            .next()
        {
            return result;
        }
        if d == n {
            if let Some(result) = (0..=(dth_digit + 1))
                .flat_map(|lc| check_solution(lc))
                .next()
            {
                return result + pow_10 * 10;
            }
        } else if d < n {
            for s_target in s_range(d) {
                for leading_coeff in 0..=9 {
                    let s_remains = s_target
                        .checked_sub(segments[leading_coeff as usize])
                        .unwrap_or(INF);
                    if s_range(d - 1).contains(&s_remains)
                        && table[d - 1][s_remains as usize] < INF
                        && table[d][s_target as usize] == INF
                    {
                        table[d][s_target as usize] =
                            table[d - 1][s_remains as usize] + leading_coeff * pow_10;
                    }
                }
            }
        }
    }

    panic!()
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let s = std::str::from_utf8(input.line()).unwrap().trim();
    let n: usize = s.len();
    let x = s.parse().unwrap();

    writeln!(output_buf, "{}", solve(n, x)).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
