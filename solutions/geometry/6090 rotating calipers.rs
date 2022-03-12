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

type Point<T> = (T, T);

fn signed_area(p: Point<i64>, q: Point<i64>, r: Point<i64>) -> i64 {
    (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    // https://www.acmicpc.net/problem/6090
    let test_cases = input.value();
    for _ in 0..test_cases {
        let n = input.value();
        let poly: Vec<(i64, i64)> = (0..n).map(|_| (input.value(), input.value())).collect();

        assert!(n >= 3);

        // count and remove colinear points
        let mut convex_poly = vec![];
        let mut multiplicity = vec![];

        {
            let inc = |i| (i + 1) % n;
            let i0 = poly.iter().enumerate().min_by_key(|(_i, &p)| p).unwrap().0;
            let mut i = i0;
            let mut group_start = i0;
            let mut yield_group = |start: usize, end: usize| {
                convex_poly.push(poly[start]);
                multiplicity.push((end - start) as u64 + 1);
            };
            loop {
                if signed_area(poly[i], poly[(i + 1) % n], poly[(i + 2) % n]) != 0 {
                    yield_group(group_start, i);
                    group_start = inc(i);
                }

                i = inc(i);
                if i == (i0 + n - 1) % n {
                    yield_group(group_start, i);
                    break;
                }
            }
        }

        let n = convex_poly.len();

        let count = (|| {
            if n <= 2 {
                return (poly.len() * (poly.len() - 1) / 2) as u64;
            }
            
            // rotating calipers
            // find number of antipodal pairs
            use std::cmp::Ordering;
            let inc = |i| (i + 1) % n;
            let signed_area = |i, j, k| signed_area(convex_poly[i], convex_poly[j], convex_poly[k]);
            let compare_segments = |i, j| {
                signed_area(i, inc(i), inc(j))
                    .cmp(&signed_area(i, inc(i), j))
                    .reverse()
            };

            let mut count: u64 = 0;
            let mut update_result = |i, j, angle_relation| {
                count += match angle_relation {
                    Ordering::Less => multiplicity[i],
                    Ordering::Greater => multiplicity[j],
                    Ordering::Equal => multiplicity[i] * multiplicity[j],
                };

                // special case: divide on edge
                if inc(j) == i {
                    count += multiplicity[j] * (multiplicity[j] + 1) / 2 - 1;
                } else if inc(i) == j {
                    count += multiplicity[i] * (multiplicity[i] + 1) / 2 - 1;
                }

                /*
                println!(
                    "update_result {} {}, {:?} {}",
                    i,
                    j,
                    angle_relation,
                    count
                );
                */
            };

            let mut i = 0;
            let mut j = (1..n).find(|&j| compare_segments(i, j).is_le()).unwrap();
            let i_last = j;
            let j_last = i;

            while (i, j) != (i_last, j_last) {
                let angle_relation = compare_segments(i, j);
                update_result(i, j, angle_relation);
                match angle_relation {
                    Ordering::Less => {
                        i = inc(i);
                    }
                    Ordering::Greater => {
                        j = inc(j);
                    }
                    Ordering::Equal => {
                        update_result(i, inc(j), Ordering::Less);
                        update_result(inc(i), j, Ordering::Greater);
                        i = inc(i);
                        j = inc(j);
                    }
                }
            }

            count
        })();
        
        println!("{:?}", count);
    }

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
