use std::{
    cmp::{Ordering, Reverse},
    io::Write,
};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)>
where
    I: IntoIterator,
    I::Item: Clone,
    J: IntoIterator,
    J::IntoIter: Clone,
{
    let j = j.into_iter();
    i.into_iter()
        .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
}

#[macro_use]
mod geometry {
    use std::{
        cmp::Ordering,
        ops::{Add, Index, IndexMut, Mul, Sub},
    };

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
    {
        fn zero() -> Self {
            Self::default()
        }
    }

    impl Scalar for f64 {}
    impl Scalar for i64 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T>(pub [T; 2]);

    impl<T: Scalar> From<[T; 2]> for Point<T> {
        fn from(p: [T; 2]) -> Self {
            Point(p)
        }
    }

    impl<T: Scalar> Index<usize> for Point<T> {
        type Output = T;
        fn index(&self, i: usize) -> &Self::Output {
            &self.0[i]
        }
    }

    impl<T: Scalar> IndexMut<usize> for Point<T> {
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.0[i]
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for Point<T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Point([self[0].$fn(other[0]), self[1].$fn(other[1])])
                }
            }
        };
    }

    impl_binop!(Add, add);
    impl_binop!(Sub, sub);
    impl_binop!(Mul, mul);

    impl<T: Scalar> Mul<T> for Point<T> {
        type Output = Self;
        fn mul(self, k: T) -> Self::Output {
            Point([self[0].mul(k), self[1].mul(k)])
        }
    }

    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    }

    pub fn convex_hull<T: Scalar>(points: &mut [Point<T>]) -> Vec<Point<T>> {
        // monotone chain algorithm
        let n = points.len();
        if n <= 1 {
            return points.to_vec();
        }
        assert!(n >= 2);

        points.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());

        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for &p in points.iter() {
            while matches!(lower.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero()) {
                lower.pop();
            }
            lower.push(p);
        }
        for &p in points.iter().rev() {
            while matches!(upper.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero()) {
                upper.pop();
            }
            upper.push(p);
        }
        lower.pop();
        upper.pop();

        lower.extend(upper);
        lower
    }

    pub fn on_boundary<T: Scalar + Ord>(polygon: &[Point<T>], p: Point<T>) -> bool {
        if polygon.len() == 1 {
            return p == polygon[0];
        }

        polygon
            .iter()
            .zip(polygon.iter().cycle().skip(1))
            .any(|(&a, &b)| {
                signed_area(a, b, p) == T::zero()
                    && a[0].min(b[0]) <= p[0]
                    && p[0] <= a[0].max(b[0])
                    && a[1].min(b[1]) <= p[1]
                    && p[1] <= a[1].max(b[1])
            })
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    use geometry::Point;

    let n: usize = input.value();
    let m: usize = input.value();

    type Grid<T> = Vec<Vec<T>>;

    let grid: Grid<u8> = (0..n)
        .map(|_| {
            input
                .token()
                .bytes()
                .take(m)
                .map(|c| match c {
                    b'R' => 0,
                    b'G' => 1,
                    b'B' => 2,
                    _ => panic!(),
                })
                .collect()
        })
        .collect();
    let mut grid_sep: Vec<Vec<Point<i64>>> = vec![vec![]; 3];
    for i in 0..n {
        for j in 0..m {
            grid_sep[grid[i][j] as usize].push(Point([i as i64, j as i64].into()));
        }
    }

    let mut grid_cvhull: Vec<Vec<Point<i64>>> = vec![vec![]; 3];
    for c in 0..3 {
        grid_cvhull[c] = geometry::convex_hull(&mut grid_sep[c]);
        grid_cvhull[c] = grid_sep[c]
            .iter()
            .cloned()
            .filter(|&p| geometry::on_boundary(&grid_cvhull[c], p))
            .collect();
    }

    let mut ans = grid_sep[0].len() as u64 * grid_sep[1].len() as u64 * grid_sep[2].len() as u64;

    let max_area = |u0: Point<i64>, u1: Point<i64>| -> Option<i64> {
        let c0 = grid[u0[0] as usize][u0[1] as usize];
        let c1 = grid[u1[0] as usize][u1[1] as usize];
        assert_ne!(c0, c1);

        let c2 = 3 - c0 - c1;

        grid_cvhull[c2 as usize]
            .iter()
            .map(|&p| geometry::signed_area(u0, u1, p).abs())
            .max()
    };
    for &u0 in &grid_cvhull[0] {
        for &u1 in &grid_cvhull[1] {
            let a = max_area(u0, u1);
            let cnt = grid_cvhull[2]
                .iter()
                .filter(|&&p| {
                    a == Some(geometry::signed_area(u0, u1, p).abs())
                        && a == max_area(u0, p)
                        && a == max_area(u1, p)
                })
                .count() as u64;

            ans -= cnt;
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
