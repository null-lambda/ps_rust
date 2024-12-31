use std::{
    cmp::{Ordering, Reverse},
    io::Write,
};

use geometry::{Angle, Point};

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

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

#[macro_use]
mod geometry {
    use std::{
        cmp::Ordering,
        ops::{Add, Index, IndexMut, Mul, Neg, Sub},
    };

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
    {
        fn zero() -> Self {
            Self::default()
        }

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }
    }

    impl Scalar for f64 {}
    impl Scalar for i64 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T>([T; 2]);

    impl<T: Scalar> Point<T> {
        pub fn new(x: T, y: T) -> Self {
            Point([x, y])
        }

        pub fn dot(self, other: Self) -> T {
            self[0] * other[0] + self[1] * other[1]
        }

        pub fn norm_sq(self) -> T {
            self.dot(self)
        }

        pub fn cross(self, other: Self) -> T {
            self[0] * other[1] - self[1] * other[0]
        }

        pub fn rot(self) -> Self {
            Point([-self[1], self[0]])
        }
    }

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

    #[derive(Debug, Clone, Copy)]
    pub struct Angle<T>(pub Point<T>);

    impl<T: Scalar> Angle<T> {
        pub fn on_lower_half(self) -> bool {
            (self.0[1], self.0[0]) < (T::zero(), T::zero())
        }

        pub fn circular_cmp(&self, other: &Self) -> Ordering {
            T::zero().partial_cmp(&self.0.cross(other.0)).unwrap()
        }
    }

    impl<T: Scalar> PartialEq for Angle<T> {
        fn eq(&self, other: &Self) -> bool {
            self.on_lower_half() == other.on_lower_half() && self.0.cross(other.0) == T::zero()
        }
    }

    impl<T: Scalar> Eq for Angle<T> {}

    impl<T: Scalar> PartialOrd for Angle<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(
                (self.on_lower_half().cmp(&other.on_lower_half()))
                    .then_with(|| self.circular_cmp(other)),
            )
        }
    }

    impl<T: Scalar> Ord for Angle<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        (q - p).cross(r - p)
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

    pub fn convex_hull_area<I>(points: I) -> f64
    where
        I: IntoIterator<Item = [f64; 2]>,
        I::IntoIter: Clone,
    {
        let mut area: f64 = 0.0;
        let points = points.into_iter();
        let points_shifted = points.clone().skip(1).chain(points.clone().next());
        for ([x1, y1], [x2, y2]) in points.zip(points_shifted) {
            area += x1 * y2 - x2 * y1;
        }
        area = (area / 2.0).abs();
        area
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum EventType {
    Enter = 1,
    Exit = 0,
    Raycast = 2,
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: i64 = input.value();
    let m: i64 = input.value();

    let center: Point<i64> = Point::new(input.value(), input.value());
    let mut events = vec![];

    let [bx0, bx1] = [-center[0], n - center[0]];
    let [by0, by1] = [-center[1], m - center[1]];

    fn vline_inter(dir: Point<i64>, x: i64) -> Option<Point<f64>> {
        if dir[0] * x <= 0 {
            return None;
        }

        let t = x as f64 / dir[0] as f64;
        let y = dir[1] as f64 * t;
        Some(Point::new(x as f64, y))
    }

    fn hline_inter(dir: Point<i64>, y: i64) -> Option<Point<f64>> {
        if dir[1] * y <= 0 {
            return None;
        }
        let t = y as f64 / dir[1] as f64;
        let x = dir[0] as f64 * t;
        Some(Point::new(x, y as f64))
    }

    let bbox_inter = |dir: Point<i64>| -> Point<f64> {
        let mut candidates = vec![];
        for x in [bx0, bx1].iter().copied() {
            if let Some(p) = vline_inter(dir, x) {
                candidates.push(p);
            }
        }
        for y in [by0, by1].iter().copied() {
            if let Some(p) = hline_inter(dir, y) {
                candidates.push(p);
            }
        }

        candidates
            .into_iter()
            .min_by(|&p, &q| p.norm_sq().partial_cmp(&q.norm_sq()).unwrap())
            .unwrap()
    };

    for x in [bx0, bx1].iter().copied() {
        for y in [by0, by1].iter().copied() {
            events.push((Angle(Point::new(x, y)), EventType::Raycast, usize::MAX));
        }
    }

    let mut n_segs = 0;
    for _ in 0..input.value() {
        let n_points = input.value();
        let mut ps: Vec<Angle<i64>> = (0..n_points)
            .map(|_| Angle(Point::new(input.value(), input.value()) - center))
            .collect();

        // Adjust origin
        for i in 0..n_points {
            let prev = ps[(i + n_points - 1) % n_points];
            let curr = ps[i];
            let next = ps[(i + 1) % n_points];
            if curr.0 == Point::new(0, 0) {
                // assert!(prev.0 != Point::new(0, 0) || next.0 != Point::new(0, 0));
                ps[i].0 = match prev.circular_cmp(&next) {
                    Ordering::Less => prev.0 + next.0,
                    Ordering::Equal => {
                        // assert!(prev.on_lower_half() != next.on_lower_half());
                        next.0.rot()
                    }
                    Ordering::Greater => (prev.0 + next.0) * -1,
                }
            }
        }

        // Remove consecutive, duplicate angles
        ps.dedup();
        if ps.len() >= 2 && ps[0] == ps[ps.len() - 1] {
            ps.pop();
        }
        if ps.len() <= 1 {
            continue;
        }
        // println!("{:?}", ps);

        // Find all increasing segment groups
        // Loop multiple times and discard two ends
        let n_points = ps.len();
        let mut groups = vec![];
        let mut curr_start = None;
        for i in 0..n_points * 4 {
            let i = i % n_points;
            let prev = ps[(i + n_points - 1) % n_points];
            let curr = ps[i];
            let next = ps[(i + 1) % n_points];
            // assert!(prev != curr);
            // assert!(curr != next);

            if prev.circular_cmp(&curr) == Ordering::Greater
                && curr.circular_cmp(&next) == Ordering::Less
            {
                if curr_start == None {
                    curr_start = Some(i);
                }
            }
            if prev.circular_cmp(&curr) == Ordering::Less
                && curr.circular_cmp(&next) == Ordering::Greater
            {
                if let Some(start) = curr_start {
                    groups.push((start, i));
                    curr_start = None;
                }
            }
        }
        if groups.len() <= 2 {
            continue;
        }
        let mut groups = groups[1..groups.len() - 1].to_vec();
        groups.sort_unstable();
        groups.dedup();

        for &(start, end) in &groups {
            if start == end {
                continue;
            }
            // println!("{:?} {:?}", ps[start].0, ps[end].0);
            events.push((ps[start], EventType::Enter, n_segs));
            events.push((ps[end], EventType::Exit, n_segs));
            n_segs += 1;
        }
    }
    events.sort_unstable();

    // Loop 1: Count the number of active segments at initial state
    let mut active = vec![false; n_segs];
    for (_, event_type, seg_id) in &events {
        match event_type {
            EventType::Enter => active[*seg_id] = true,
            EventType::Exit => active[*seg_id] = false,
            _ => {}
        }
    }
    let mut active_count = (0..n_segs).filter(|&i| active[i]).count();
    drop(active);

    // Loop 2: Raycast
    let mut furthest_target = (0.0, None);
    for (angle, event_type, _) in &events {
        match event_type {
            EventType::Enter => active_count += 1,
            EventType::Exit => active_count -= 1,
            _ => {}
        }

        if active_count == 0 {
            let inter = bbox_inter(angle.0);
            let len = inter.norm_sq();
            if (len, Some(Reverse(inter))) >= furthest_target {
                furthest_target = (len, Some(Reverse(inter)));
            }
        }
    }

    if let (_, Some(Reverse(mut p))) = furthest_target {
        p[0] += center[0] as f64;
        p[1] += center[1] as f64;

        writeln!(output, "{} {}", p[0], p[1]).unwrap();
    } else {
        writeln!(output, "GG").unwrap();
    }
}
