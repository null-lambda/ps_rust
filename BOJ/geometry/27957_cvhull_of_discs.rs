use std::{collections::HashMap, io::Write};

use geometry::Point;

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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

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
        + PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Default
    {
        fn zero() -> Self {
            Self::default()
        }

        fn one() -> Self;

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }
    }

    impl Scalar for i64 {
        fn one() -> Self {
            1
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Point<T>([T; 2]);

    impl<T> Point<T> {
        pub fn new(x: T, y: T) -> Self {
            Point([x, y])
        }

        pub fn map<S>(self, f: impl FnMut(T) -> S) -> Point<S> {
            Point(self.0.map(f))
        }
    }

    impl<T: Scalar> Point<T> {
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

    impl<T: Scalar> Neg for Point<T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Point(self.0.map(T::neg))
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for Point<T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Point(std::array::from_fn(|i| self[i].$fn(other[i])))
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
}

pub mod ordered {
    use std::{
        cmp::Ordering,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
        str::FromStr,
    };

    #[derive(Clone, Copy, Default)]
    pub struct F64(pub f64);

    impl F64 {
        pub const fn new(x: f64) -> Self {
            Self(x)
        }

        pub fn map(self, f: impl FnOnce(f64) -> f64) -> Self {
            Self(f(self.0))
        }

        pub fn with<T>(self, f: impl FnOnce(f64) -> T) -> T {
            f(self.0)
        }
    }

    impl PartialEq for F64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.total_cmp(&other.0).is_eq()
        }
    }

    impl Eq for F64 {}

    impl PartialOrd for F64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.0.total_cmp(&other.0))
        }
    }

    impl Ord for F64 {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    impl std::hash::Hash for F64 {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.0.to_bits().hash(state);
        }
    }

    impl std::fmt::Debug for F64 {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident, $trait_assign:ident, $fn_assign:ident) => {
            impl $trait for F64 {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Self(self.0.$fn(other.0))
                }
            }

            impl $trait_assign for F64 {
                fn $fn_assign(&mut self, other: Self) {
                    self.0.$fn_assign(other.0);
                }
            }
        };
    }

    impl Neg for F64 {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl_binop!(Add, add, AddAssign, add_assign);
    impl_binop!(Sub, sub, SubAssign, sub_assign);
    impl_binop!(Mul, mul, MulAssign, mul_assign);
    impl_binop!(Div, div, DivAssign, div_assign);

    impl super::geometry::Scalar for F64 {
        fn one() -> Self {
            Self(1.0)
        }
    }

    impl FromStr for F64 {
        type Err = <f64 as FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(Self(f64::from_str(s)?))
        }
    }
}

pub mod cvhull {
    use core::f64;
    use std::f64::consts::PI;

    use crate::{geometry::Point, ordered};

    pub type T = ordered::F64;
    const EPS: T = T::new(1.0e-15);

    pub type Circle = (Point<T>, T);

    fn arg(p: Point<T>) -> T {
        T::new(p[1].0.atan2(p[0].0))
    }

    fn mod_2pi(arg: T) -> T {
        arg.map(|x| x.rem_euclid(2.0 * PI))
    }

    fn unit(arg: T) -> Point<T> {
        Point::new(arg.0.cos(), arg.0.sin()).map(T::new)
    }

    fn lower_tangent(c0: &Circle, c1: &Circle, tol: T) -> Option<T> {
        let &(p0, r0) = c0;
        let &(p1, r1) = c1;
        let dp = p1 - p0;
        let (d, a01) = (dp.norm_sq().map(|x| x.sqrt()), arg(dp));

        let sin = (r0 - r1) / d;
        if !(sin.0.abs() <= 1.0 + tol.0) {
            return None;
        }

        let a = sin.map(|x| x.min(1.0).max(-1.0).asin());
        Some(mod_2pi(a01 + a))
    }

    // David Rappaport (1992). "A convex hull algorithm for discs, and applications".
    // https://doi.org/10.1016/0925-7721(92)90015-K
    pub fn of_discs(cs: &mut Vec<Circle>) -> Vec<Circle> {
        cs.sort_unstable();
        cs.dedup();
        dnc(cs)
    }

    pub fn dnc(cs: &[Circle]) -> Vec<Circle> {
        if cs.len() <= 1 {
            return cs.to_vec();
        }
        let (lhs, rhs) = cs.split_at(cs.len() / 2);
        merge_hull(&dnc(lhs), &dnc(rhs))
    }

    struct Finished;

    #[must_use]
    fn push_dedup(hull: &mut Vec<Circle>, p: &Circle) -> Result<(), Finished> {
        if hull.last() != Some(p) {
            hull.push(*p);
            if hull.len() >= 3 && hull[..2] == hull[hull.len() - 2..] {
                hull.pop();
                hull.pop();
                return Err(Finished);
            }
        }
        Ok(())
    }

    fn advance(
        ps: &[Circle],
        qs: &[Circle],
        hull: &mut Vec<Circle>,
        base: &mut T,
        i: &mut usize,
        j: &mut usize,
    ) -> Result<(), Finished> {
        let n = ps.len();
        let m = qs.len();
        debug_assert!(n >= 1 && m >= 1);

        let i_inc = (*i + 1) % n;
        let j_inc = (*j + 1) % m;

        let angles = [
            lower_tangent(&ps[*i], &qs[*j], -EPS),
            lower_tangent(&qs[*j], &ps[*i], -EPS),
            lower_tangent(&ps[*i], &ps[i_inc], EPS),
            lower_tangent(&qs[*j], &qs[j_inc], EPS),
        ];
        let angles_rel = angles.map(|a| {
            a.map(|x| mod_2pi(x - *base))
                .unwrap_or(T::new(f64::INFINITY))
        });

        if angles_rel[0] < angles_rel[2] && angles_rel[0] < angles_rel[3] {
            push_dedup(hull, &qs[*j])?;
        }
        if angles_rel[1] < angles_rel[2] && angles_rel[1] < angles_rel[3] {
            push_dedup(hull, &ps[*i])?;
        }

        if angles_rel[2] < angles_rel[3] {
            *base = angles[2].unwrap_or(*base);
            *i = i_inc;
        } else {
            *base = angles[3].unwrap_or(*base);
            *j = j_inc;
        }

        Ok(())
    }

    fn merge_hull(ps: &[Circle], qs: &[Circle]) -> Vec<Circle> {
        let n = ps.len();
        let m = qs.len();
        debug_assert!(n >= 1 && m >= 1);

        let mut merged = vec![];
        let mut i = 0;
        let mut j = 0;
        let mut angle = T::new(0.0f64);
        let mut iter = 0;

        (|| -> Result<(), Finished> {
            loop {
                let normal_inward = unit(angle).rot();
                let p = ps[i].0 - normal_inward * ps[i].1;
                let q = qs[j].0 - normal_inward * qs[j].1;
                let w = (q - p).dot(normal_inward);

                let dom = if lower_tangent(&ps[i], &qs[j], -EPS).is_some() {
                    w.0 >= 0.0
                } else {
                    ps[i].1 > qs[j].1
                };

                if dom {
                    push_dedup(&mut merged, &ps[i])?;
                    advance(ps, qs, &mut merged, &mut angle, &mut i, &mut j)?;
                } else {
                    push_dedup(&mut merged, &qs[j])?;
                    advance(qs, ps, &mut merged, &mut angle, &mut j, &mut i)?;
                }

                iter += 1;
                if iter >= 4 * n + 4 {
                    // Should be unnecessary for robust predicates
                    return Err(Finished);
                }
            }
        })()
        .ok();

        while merged.len() >= 2 && merged[0].0 == merged[merged.len() - 1].0 {
            merged.pop();
        }
        merged
    }

    pub fn perimeter(hull: &[Circle]) -> T {
        let n = hull.len();
        assert!(n >= 1);

        if n == 1 {
            let (_, r) = hull[0];
            return r * T::new(2.0 * PI);
        }

        let mut res = T::new(0.0);
        for i in 0..n {
            let c0 = hull[i];
            let c1 = hull[(i + 1) % n];
            let c2 = hull[(i + 2) % n];

            let (p0, r0) = c0;
            let (p1, r1) = c1;

            let dr = r1 - r0;
            res += ((p1 - p0).norm_sq() - dr * dr).map(|x| x.max(0.0).sqrt());

            let a0 = lower_tangent(&c0, &c1, T::new(0.0));
            let a1 = lower_tangent(&c1, &c2, T::new(0.0));
            if let (Some(a0), Some(a1)) = (a0, a1) {
                res += r1 * mod_2pi(a1 - a0).map(|x| x.abs());
            }
        }

        res
    }
}

type RenderTag = u8;
fn circles_to_svg<I>(cs: I, filename: &str) -> std::io::Result<()>
where
    I: Iterator<Item = (cvhull::Circle, RenderTag)> + Clone,
{
    let mut file = std::fs::File::create(filename)?;

    let mut vx0 = f32::MAX;
    let mut vx1 = f32::MIN;
    let mut vy0 = f32::MAX;
    let mut vy1 = f32::MIN;

    for ((p, r), _) in cs.clone() {
        vx0 = vx0.min(p[0].0 as f32 - r.0 as f32);
        vx1 = vx1.max(p[0].0 as f32 + r.0 as f32);
        vy0 = vy0.min(p[1].0 as f32 - r.0 as f32);
        vy1 = vy1.max(p[1].0 as f32 + r.0 as f32);
    }
    let vw = vx1 - vx0;
    let vh = vy1 - vy0;
    let s = 1.3;

    writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        file,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}">"#,
        vx0 - vw * (s - 1.0) * 0.5,
        vy0 - vh * (s - 1.0) * 0.5,
        vw * s,
        vh * s,
    )?;

    let mut counter = HashMap::<RenderTag, u32>::new();
    for ((p, r), c) in cs {
        let color = match c {
            1 => "red",
            _ => "black",
        };
        let sw = match c {
            1 => 1,
            _ => 2,
        };

        let t = *counter.entry(c).or_default();
        counter.insert(c, t + 1);

        writeln!(
            file,
            r#"<circle cx="{}" cy="{}" r="{}" fill="none" stroke="{}" stroke-width="{}" />"#,
            p[0].0, p[1].0, r.0, color, sw
        )?;

        writeln!(
            file,
            r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="middle">{}</text>"#,
            p[0].0 + (t as f64 - 5.0) / 10.0 * r.0,
            p[1].0,
            r.0 * 0.6,
            color,
            t
        )?;
    }

    writeln!(file, "</svg>")?;
    Ok(())
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    // for i_tc in 0..input.value() {
    for i_tc in 0..1 {
        let n: usize = input.value();
        let mut cs: Vec<cvhull::Circle> = (0..n)
            .map(|_| {
                let x: ordered::F64 = input.value();
                let y: ordered::F64 = input.value();
                let r: ordered::F64 = input.value();
                (Point::new(x, y), r)
            })
            .collect();
        cs.sort_unstable();
        cs.dedup();

        let hull = cvhull::of_discs(&mut cs);
        let ans = cvhull::perimeter(&hull);

        debug::with(|| {
            std::fs::create_dir_all("./dbg").unwrap();
            circles_to_svg(
                (cs.iter().map(|&c| (c, 0))).chain(hull.iter().map(|&c| (c, 1))),
                &format!("./dbg/hull_{:02}.svg", i_tc),
            )
            .unwrap();
        });

        writeln!(output, "{:.10}", ans.0).unwrap();
    }
}
