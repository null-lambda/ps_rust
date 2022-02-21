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

fn signed_area(p: Point<i32>, q: Point<i32>, r: Point<i32>) -> i32 {
    (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
}

// monotone chain algorithm
fn convex_hull(points: &mut [Point<i32>]) -> Vec<Point<i32>> {
    let n = points.len();
    if n == 0 || n == 1 {
        return points.to_vec();
    }
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

fn convex_hull_contains(poly: &[Point<i32>], p: Point<i32>) -> bool {
    use std::iter::once;
    assert!(poly.len() >= 3);
    (poly.iter().zip(poly.iter().skip(1).chain(once(&poly[0]))))
        .all(|(&l1, &l2)| signed_area(l1, l2, p) > 0)
}

fn interval_overlaps((a1, a2): (i32, i32), (b1, b2): (i32, i32)) -> bool {
    let reorder = |a, b| if a < b { (a, b) } else { (b, a) };
    let (a1, a2) = reorder(a1, a2);
    let (b1, b2) = reorder(b1, b2);
    a1.max(b1) <= a2.min(b2)
}

fn segment_intersects(p1: Point<i32>, p2: Point<i32>, q1: Point<i32>, q2: Point<i32>) -> bool {
    use std::cmp::Ordering;

    // aabb test for fast rejection
    if !interval_overlaps((p1.0, p2.0), (q1.0, q2.0))
        || !interval_overlaps((p1.1, p2.1), (q1.1, q2.1))
    {
        return false;
    }

    let sub = |p: Point<i32>, q: Point<i32>| (p.0 - q.0, p.1 - q.1);
    // intersection = p1 + t * (p2 - p1) = q1 + s * (q2 - q1),
    // => t (p2 - p1) - s (q2 - q1) + (p1 - q1) = 0
    // => t (p2 - p1) - s (q2 - q1) = q1 - p1
    let pd = sub(p2, p1);
    let qd = sub(q2, q1);
    let r = sub(q1, p1);

    // solve linear equation
    let det = -pd.0 * qd.1 + pd.1 * qd.0;
    let mul_det_t = -qd.1 * r.0 + qd.0 * r.1;
    let mul_det_s = -pd.1 * r.0 + pd.0 * r.1;

    match &det.cmp(&0) {
        Ordering::Greater => (0..=det).contains(&mul_det_t) && (0..=det).contains(&mul_det_s),
        Ordering::Less => (det..=0).contains(&mul_det_t) && (det..=0).contains(&mul_det_s),
        Ordering::Equal => signed_area((0, 0), pd, r) == 0,
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let t = input.value();
    for _ in 0..t {
        let n = input.value();
        let m = input.value();
        let mut read_cvhull = |n| {
            let mut points: Vec<Point<i32>> =
                (0..n).map(|_| (input.value(), input.value())).collect();
            convex_hull(&mut points[..])
        };

        use std::iter::once;
        let cv1 = read_cvhull(n);
        let cv2 = read_cvhull(m);
        let mut intersects = n == 0 || m == 0;
        intersects = intersects || n == 1 && m == 2 && signed_area(cv1[0], cv2[0], cv2[1]) == 0;
        intersects = intersects || n == 2 && m == 1 && signed_area(cv2[0], cv1[0], cv1[1]) == 0;
        intersects = intersects || m >= 3 && cv1.iter().any(|&p| convex_hull_contains(&cv2[..], p));
        intersects = intersects || n >= 3 && cv2.iter().any(|&p| convex_hull_contains(&cv1[..], p));
        intersects = intersects
            || (n >= 2 && m >= 2)
                && (cv1.iter().zip(cv1.iter().skip(1).chain(once(&cv1[0]))))
                    .flat_map(|(&l1, &l2)| {
                        (cv2.iter().zip(cv2.iter().skip(1).chain(once(&cv2[0]))))
                            .map(move |(&m1, &m2)| (l1, l2, m1, m2))
                    })
                    .any(|(l1, l2, m1, m2)| segment_intersects(l1, l2, m1, m2));
        // println!("{:?}", cv1);
        // println!("{:?}", cv2);
        println!("{}", if intersects { "NO" } else { "YES" });
    }
    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
