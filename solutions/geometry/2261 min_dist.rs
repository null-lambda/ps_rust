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
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
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
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let mut points: Box<[_]> = (0..n).map(|_| [input.value(), input.value()]).collect();
    points.sort_unstable();

    assert!(n >= 2);

    fn dist_sq(p: &[i16; 2], q: &[i16; 2]) -> i32 {
        let dr = [(p[0] - q[0]) as i32, (p[1] - q[1]) as i32];
        return dr[0] * dr[0] + dr[1] * dr[1];
    }

    fn dnc(points: &[[i16; 2]], min_dist_sq: &mut i32) {
        match points {
            [] | [_] => return,
            [p, q] => {
                *min_dist_sq = (*min_dist_sq).min(dist_sq(p, q));
                return;
            }
            [p, q, r] => {
                *min_dist_sq = (*min_dist_sq)
                    .min(dist_sq(p, q))
                    .min(dist_sq(q, r))
                    .min(dist_sq(p, r));
            }
            _ => (),
        };

        let mid = points.len() / 2;
        let (left, right) = points.split_at(mid);
        dnc(left, min_dist_sq);
        dnc(right, min_dist_sq);

        let sq = |x: i16| x as i32 * x as i32;
        let mut band: Box<[_]> = points
            .iter()
            .filter(|&[x, _]| sq(right.first().unwrap()[0] - x) <= *min_dist_sq)
            .collect();
        band.sort_unstable_by_key(|&[_, y]| y);
        for i in 0..band.len() {
            for j in i + 1..band.len() {
                if sq(band[j][1] - band[i][1]) >= *min_dist_sq {
                    break;
                }
                *min_dist_sq = (*min_dist_sq).min(dist_sq(band[i], band[j]));
            }
        }
    }

    let mut min_dist_sq = i32::MAX;
    dnc(&points, &mut min_dist_sq);
    println!("{:?}", min_dist_sq);

    std::io::stdout().write(&output_buf[..]).unwrap();
}
