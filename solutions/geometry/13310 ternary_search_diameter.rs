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
// use std::io::Write;

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

// monotone chain algorithm
fn convex_hull(points: &mut [Point<i64>]) -> Vec<Point<i64>> {
    let n = points.len();
    assert!(n >= 2);
    points.sort_unstable_by_key(|&(x, y)| (x, y));

    let ccw = |p, q, r| signed_area(p, q, r) > 0;

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &p in points.iter() {
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

    lower.extend(upper);
    lower
}

// the points should be ordered strictly ccw.
fn diameter_sq(convex_poly: &[Point<i64>]) -> i64 {
    use std::cmp::Ordering;

    let n = convex_poly.len();
    assert!(n >= 2);
    // should consider the case n = 2;

    let inc = |i| (i + 1) % n;
    let signed_area = |i, j, k| signed_area(convex_poly[i], convex_poly[j], convex_poly[k]);
    let compare_segments = |i, j| signed_area(i, inc(i), inc(j)).cmp(&signed_area(i, inc(i), j));

    let mut result = 0;
    let mut update_result = |i, j| {
        let p: Point<_> = convex_poly[i];
        let q: Point<_> = convex_poly[j];
        let dr = (p.0 - q.0, p.1 - q.1);
        result = result.max(dr.0 * dr.0 + dr.1 * dr.1);
    };

    let mut i = 0;
    let mut j = (1..n).find(|&j| compare_segments(i, j).is_le()).unwrap();
    let i_last = j;
    let j_last = i;

    while (i, j) != (i_last, j_last) {
        update_result(i, j);
        match compare_segments(i, j) {
            Ordering::Less => {
                i = inc(i);
            }
            Ordering::Greater => {
                j = inc(j);
            }
            Ordering::Equal => {
                update_result(i, inc(j));
                update_result(inc(i), j);
                i = inc(i);
                j = inc(j);
            }
        }
    }
    result
}

// find minimum of unimodal function
fn ternary_search<T: Ord>(mut f: impl FnMut(&i32) -> T, mut left: i32, mut right: i32) -> T {
    while right - left >= 3 {
        let left_third = left + (right - left) / 3;
        let right_third = right - (right - left) / 3;
        if f(&left_third) < f(&right_third) {
            right = right_third;
        } else {
            left = left_third;
        }
    }
    (left..=right).map(|x| f(&x)).min().unwrap()
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let max_time: i32 = input.value();

    let mut points: Vec<(Point<i32>, Point<i32>)> = (0..n)
        .map(|_| {
            (
                (input.value(), input.value()),
                (input.value(), input.value()),
            )
        })
        .collect();

    // as t grows, the velocity of stars becomes the dominant factor in determining the position of stars.
    //
    points.sort_unstable_by_key(|&(_, (vx, vy))| (vx, vy));

    let mut points_moved: Box<[Point<i64>]> = (0..n).map(|_| (0, 0)).collect();
    let (min_diameter, time) = ternary_search(
        |&time: &i32| {
            assert!(points_moved.len() == n);
            for ((p_initial, velocity), p) in points.iter().zip(points_moved.iter_mut()) {
                p.0 = (p_initial.0 + velocity.0 * time) as i64;
                p.1 = (p_initial.1 + velocity.1 * time) as i64;
            }
            let convex_hull = convex_hull(&mut points_moved[..]);
            (diameter_sq(&convex_hull[..]), time)
        },
        0,
        max_time,
    );

    writeln!(output_buf, "{}\n{}", time, min_diameter).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
