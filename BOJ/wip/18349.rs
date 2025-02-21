use std::{collections::HashMap, io::Write};

use geometry::Point;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

mod cmp {
    // The equalizer of all things
    use std::cmp::Ordering;

    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            // All values are equal, but Some(_)™ are more equal than others...
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}

#[allow(dead_code)]
#[macro_use]
pub mod geometry {
    use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct Ordered<T>(T);

    impl From<f64> for Ordered<f64> {
        fn from(x: f64) -> Self {
            debug_assert!(!x.is_nan());
            Self(x)
        }
    }

    impl Into<f64> for Ordered<f64> {
        fn into(self) -> f64 {
            self.0
        }
    }

    impl From<f32> for Ordered<f64> {
        fn from(x: f32) -> Self {
            debug_assert!(x.is_finite());
            Self(x as f64)
        }
    }

    impl<T: PartialEq> Eq for Ordered<T> {}
    impl<T: PartialOrd> Ord for Ordered<T> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
        + std::fmt::Debug
    {
        fn zero() -> Self {
            Self::default()
        }

        fn one() -> Self;

        fn two() -> Self {
            Self::one() + Self::one()
        }

        fn min(self, other: Self) -> Self {
            if self < other {
                self
            } else {
                other
            }
        }

        fn max(self, other: Self) -> Self {
            if self < other {
                other
            } else {
                self
            }
        }

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }

        fn sq(self) -> Self {
            self * self
        }
    }

    impl Scalar for f64 {
        fn one() -> Self {
            1.0
        }
    }

    impl Scalar for i64 {
        fn one() -> Self {
            1
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PointNd<const N: usize, T>(pub [T; N]);

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        pub fn map<F, S>(self, mut f: F) -> PointNd<N, S>
        where
            F: FnMut(T) -> S,
        {
            PointNd(self.0.map(|x| f(x)))
        }
    }

    impl<const N: usize, T: Scalar> From<[T; N]> for PointNd<N, T> {
        fn from(p: [T; N]) -> Self {
            Self(p)
        }
    }

    impl<const N: usize, T: Scalar> Index<usize> for PointNd<N, T> {
        type Output = T;
        fn index(&self, i: usize) -> &Self::Output {
            &self.0[i]
        }
    }

    impl<const N: usize, T: Scalar> IndexMut<usize> for PointNd<N, T> {
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.0[i]
        }
    }

    macro_rules! impl_binop_dims {
        ($N:expr, $($idx:expr )+, $trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for PointNd<$N, T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    PointNd([$(self[$idx].$fn(other[$idx])),+])
                }
            }
        };
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl_binop_dims!(2, 0 1, $trait, $fn);
            impl_binop_dims!(3, 0 1 2, $trait, $fn);
        };
    }

    impl_binop!(Add, add);
    impl_binop!(Sub, sub);
    impl_binop!(Mul, mul);
    impl_binop!(Div, div);

    impl<const N: usize, T: Scalar> Default for PointNd<N, T> {
        fn default() -> Self {
            PointNd([T::zero(); N])
        }
    }

    impl<const N: usize, T: Scalar> Neg for PointNd<N, T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            PointNd(self.0.map(|x| -x))
        }
    }

    impl<const N: usize, T: Scalar> Mul<T> for PointNd<N, T> {
        type Output = Self;
        fn mul(self, k: T) -> Self::Output {
            PointNd(self.0.map(|x| x * k))
        }
    }

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        pub fn zero() -> Self {
            Self::default()
        }

        pub fn dot(self, other: Self) -> T {
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a * b)
                .reduce(|acc, x| acc + x)
                .unwrap()
        }

        pub fn cross(self, other: Point<T>) -> T {
            self[0] * other[1] - self[1] * other[0]
        }

        pub fn norm_sq(self) -> T {
            self.dot(self)
        }

        pub fn max_norm(self) -> T {
            self.0
                .into_iter()
                .map(|a| a.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    pub type Point<T> = PointNd<2, T>;

    impl<T: Scalar> Point<T> {
        pub fn new(x: T, y: T) -> Self {
            Self([x, y])
        }

        pub fn rot(&self) -> Self {
            Point::new(-self[1], self[0])
        }
    }

    // predicate 1 for voronoi diagram
    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        (q - p).cross(r - p)
    }

    pub fn line_intersection(
        p1: Point<f64>,
        dir1: Point<f64>,
        p2: Point<f64>,
        dir2: Point<f64>,
    ) -> Option<Point<f64>> {
        let denom = dir1.cross(dir2);
        let result = p1 + dir1 * (p2 - p1).cross(dir2) * denom.recip();
        (result[0].is_finite() && result[1].is_finite()).then(|| result)
    }

    pub fn circumcenter(p: Point<f64>, q: Point<f64>, r: Point<f64>) -> Option<Point<f64>> {
        line_intersection((p + q) * 0.5, (q - p).rot(), (p + r) * 0.5, (r - p).rot())
    }

    // predicate 2 for voronoi diagram
    pub fn breakpoint_x(left: Point<f64>, right: Point<f64>, sweepline: f64) -> f64 {
        let y1 = left[1] - sweepline;
        let y2 = right[1] - sweepline;
        let xm = (left[0] + right[0]) * 0.5;
        let dx_half = (right[0] - left[0]) * 0.5;

        let a = y2 - y1;
        let b_2 = (y1 + y2) * dx_half;
        let c = y1 * y2 * (y1 - y2) - (y1 - y2) * dx_half * dx_half;

        let det_4 = b_2 * b_2 - a * c;
        let sign = a.signum() * (y2 - y1).signum();
        let mut x = xm + (-b_2 - sign * det_4.max(0.0).sqrt()) / a;
        if !x.is_finite() {
            if dx_half > 0.0 {
                x = xm - c * 0.5 / b_2;
                if !x.is_finite() {
                    x = xm;
                }
            } else {
                x = f64::INFINITY;
            }
        }
        x
    }
}

// Assign an unique tag for each node, for debug print
#[cfg(debug_assertions)]
pub mod tag {
    pub type Tag = i32;
    static mut TAG: i32 = -1;
    pub fn next() -> Tag {
        unsafe {
            TAG += 1;
            TAG
        }
    }
}

// Planar graph
pub mod planar_graph {
    use super::geometry::Point;

    pub const UNSET: u32 = 1 << 31;
    pub const INF: u32 = 1 << 30;

    #[derive(Debug, Copy, Clone)]
    pub struct Vertex {
        pub half_edge: u32,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct HalfEdge {
        pub src: u32,
        pub face_on_left: u32,
        // pub cycle_next: u32, // TODO!
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Face {
        pub half_edge: u32,
    }

    #[derive(Debug)]
    pub struct PlanarGraph {
        // Graph topology
        pub verts: Vec<Vertex>,
        pub half_edges: Vec<HalfEdge>,
        pub faces: Vec<Face>,

        // Planar embedding
        pub vert_coord: Vec<Point<f64>>,
        pub face_center: Vec<Point<f64>>,
    }

    impl PlanarGraph {
        pub fn twin(&self, e: u32) -> u32 {
            e ^ 1
        }
    }
}

pub mod voronoi {
    pub mod splay {
        // bottom-up splay tree for beachline

        use std::{cmp::Ordering, mem::MaybeUninit, num::NonZeroU32};

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Branch {
            Left = 0,
            Right = 1,
        }

        use Branch::{Left, Right};

        impl Branch {
            pub fn usize(&self) -> usize {
                *self as usize
            }

            pub fn inv(&self) -> Self {
                match self {
                    Branch::Left => Branch::Right,
                    Branch::Right => Branch::Left,
                }
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct NodeRef {
            idx: NonZeroU32,
        }

        impl NodeRef {
            fn get(&self) -> usize {
                self.idx.get() as usize
            }
        }

        impl std::fmt::Debug for NodeRef {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.get())
            }
        }

        #[repr(align(32))]
        #[derive(Debug, Default)]
        pub struct BreakpointNode {
            // Node data
            pub site: u32,
            pub left_half_edge: u32,

            // Traditional links of a bottom-up splay tree
            pub children: [Option<NodeRef>; 2],
            pub parent: Option<NodeRef>,

            // Doubly linked list structure
            pub side: [Option<NodeRef>; 2],
        }

        #[derive(Debug)]
        pub struct SplayForest {
            pub pool: Vec<MaybeUninit<BreakpointNode>>,
        }

        impl SplayForest {
            pub fn new() -> Self {
                let dummy = MaybeUninit::uninit();
                Self { pool: vec![dummy] }
            }

            pub fn add_root(&mut self, node: BreakpointNode) -> NodeRef {
                let idx = self.pool.len();
                self.pool.push(MaybeUninit::new(node));
                NodeRef {
                    idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
                }
            }

            pub fn is_root(&self, u: NodeRef) -> bool {
                self.get(u).parent.is_none()
            }

            pub fn get<'a>(&'a self, u: NodeRef) -> &'a BreakpointNode {
                unsafe { &self.pool[u.get()].assume_init_ref() }
            }

            unsafe fn get_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut BreakpointNode {
                self.pool[u.get()].assume_init_mut()
            }

            #[inline(always)]
            pub fn with<T>(&mut self, u: NodeRef, f: impl FnOnce(&mut BreakpointNode) -> T) -> T {
                f(unsafe { self.get_mut(u) })
            }

            pub fn get_parent(&self, u: NodeRef) -> Option<(NodeRef, Branch)> {
                let p = self.get(u).parent?;
                let branch = if self.get(p).children[Branch::Left.usize()] == Some(u) {
                    Branch::Left
                } else {
                    Branch::Right
                };
                Some((p, branch))
            }

            fn link_sides(&mut self, lhs: NodeRef, rhs: NodeRef) {
                self.with(lhs, |lhs| lhs.side[Right.usize()] = Some(rhs));
                self.with(rhs, |rhs| rhs.side[Left.usize()] = Some(lhs));
            }

            fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
                debug_assert_ne!(u, child);
                self.with(u, |u| u.children[branch.usize()] = Some(child));
                self.with(child, |child| child.parent = Some(u));
            }

            fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
                unsafe {
                    let child = self.get_mut(u).children[branch.usize()].take()?;
                    self.with(child, |child| child.parent = None);
                    Some(child)
                }
            }

            fn rotate(&mut self, u: NodeRef) {
                let (parent, branch) = self.get_parent(u).expect("Root shouldn't be rotated");
                let child = self.detach(u, branch.inv());

                if let Some(child) = child {
                    self.attach(parent, child, branch);
                } else {
                    self.detach(parent, branch);
                }

                match self.get_parent(parent) {
                    Some((grandparent, grandbranch)) => self.attach(grandparent, u, grandbranch),
                    None => self.with(u, |u| u.parent = None),
                }
                self.attach(u, parent, branch.inv());
            }

            pub fn splay(&mut self, u: NodeRef) {
                while let Some((parent, branch)) = self.get_parent(u) {
                    if let Some((_, grandbranch)) = self.get_parent(parent) {
                        if branch != grandbranch {
                            self.rotate(u);
                        } else {
                            self.rotate(parent);
                        }
                    }
                    self.rotate(u);
                }
            }

            // splay the last truthy element in [true, ..., true, false, ..., false]
            // if there is no such element, return false
            pub fn splay_last_by<F>(&mut self, root: &mut NodeRef, pred: F) -> bool
            where
                F: Fn(&Self, NodeRef) -> bool,
            {
                debug_assert!(self.is_root(*root));
                let mut current = *root;
                let mut last_pred;
                loop {
                    last_pred = pred(self, current);
                    let branch = if last_pred { Right } else { Left };
                    let Some(child) = self.get(current).children[branch.usize()] else {
                        break;
                    };
                    current = child;
                }
                if !last_pred {
                    let Some(left) = self.get(current).side[Left.usize()] else {
                        return false;
                    };
                    current = left;
                }

                self.splay(current);
                *root = current;
                true
            }

            pub fn splay_by(
                &mut self,
                root: &mut NodeRef,
                mut cmp: impl FnMut(&BreakpointNode) -> Ordering,
            ) {
                debug_assert!(self.is_root(*root));
                let mut current = *root;
                loop {
                    let current_ref = self.get(current);
                    let branch = match cmp(current_ref) {
                        Ordering::Less => Left,
                        Ordering::Greater => Right,
                        Ordering::Equal => break,
                    };
                    let Some(child) = current_ref.children[branch.usize()] else {
                        break;
                    };
                    current = child;
                }
                self.splay(current);
                *root = current;
            }

            pub fn splay_first(&mut self, root: &mut NodeRef) {
                self.splay_by(root, |_| Ordering::Less);
            }

            pub fn splay_last(&mut self, root: &mut NodeRef) {
                self.splay_by(root, |_| Ordering::Greater);
            }

            pub fn insert_right(&mut self, root: NodeRef, new_node: NodeRef) {
                debug_assert!(self.is_root(root));
                debug_assert_ne!(root, new_node);

                let old_right = self.get(root).children[Right.usize()];
                self.attach(root, new_node, Right);
                old_right.map(|old_right| {
                    let next = unsafe { self.get(root).side[Right.usize()].unwrap_unchecked() };
                    self.attach(new_node, old_right, Right);
                    self.link_sides(new_node, next);
                });
                self.link_sides(root, new_node);
            }

            pub fn pop_root(&mut self, root: &mut Option<NodeRef>) -> Option<NodeRef> {
                let Some(old_root) = *root else {
                    return None;
                };
                debug_assert!(self.is_root(old_root));

                let left = self.detach(old_root, Left);
                let right = self.detach(old_root, Right);
                let [prev, next] = self.get(old_root).side;
                self.with(old_root, |old_root| old_root.side = [None, None]);

                match (left, right) {
                    (Some(_), Some(right)) => {
                        let prev = unsafe { prev.unwrap_unchecked() };
                        let next = unsafe { next.unwrap_unchecked() };
                        self.link_sides(prev, next);

                        self.splay(prev);
                        self.attach(prev, right, Right);
                        *root = Some(prev);
                    }
                    (Some(left), None) => {
                        let prev = unsafe { prev.unwrap_unchecked() };
                        self.with(prev, |prev| prev.side[Right.usize()] = None);
                        *root = Some(left);
                    }
                    (None, Some(right)) => {
                        let next = unsafe { next.unwrap_unchecked() };
                        self.with(next, |next| next.side[Left.usize()] = None);
                        *root = Some(right);
                    }
                    (None, None) => {
                        *root = None;
                    }
                }

                Some(old_root)
            }

            pub fn inorder(&mut self, root: NodeRef, visitor: &mut impl FnMut(&Self, NodeRef)) {
                self.get(root).children[Branch::Left.usize()].map(|lhs| self.inorder(lhs, visitor));
                visitor(self, root);
                self.get(root).children[Branch::Right.usize()]
                    .map(|rhs| self.inorder(rhs, visitor));
            }
        }

        impl Drop for SplayForest {
            fn drop(&mut self) {
                for (_, node) in self.pool.iter_mut().enumerate().skip(1) {
                    unsafe {
                        node.assume_init_drop();
                    }
                }
            }
        }
    }

    use std::{cmp::Reverse, collections::BinaryHeap};

    use super::planar_graph::{self, HalfEdge, PlanarGraph};
    use super::{
        cmp::Trivial,
        geometry::{self, Ordered, Point, PointNd},
    };

    const NEG_INF: f64 = f64::NEG_INFINITY;
    const EPS: f64 = 1e-9;

    #[derive(Debug, Clone)]
    pub enum Event {
        Site(Site),
        Circle(Circle),
    }

    #[derive(Debug, Clone)]
    pub struct Site {
        pub idx: u32,
    }

    #[derive(Debug, Clone)]
    pub struct Circle {
        pub node: splay::NodeRef,
        pub side_sites: [u32; 2],
    }

    #[derive(Debug)]
    pub struct Beachline {
        pub pool: splay::SplayForest,
        pub root: Option<splay::NodeRef>,
    }

    #[derive(Debug)]
    pub struct Builder {
        pub events: BinaryHeap<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)>,
        pub beachline: Beachline,
        pub directrix: Ordered<f64>,
        pub graph: PlanarGraph,
        _init: bool,
    }

    impl Beachline {
        fn splay_at_x(&mut self, graph: &PlanarGraph, directrix: Ordered<f64>, px: f64) {
            // Do a raycast to the beachline from (x, y) = (px, directrix)
            if self.root.is_none() {
                return;
            }
            self.pool
                .splay_last_by(self.root.as_mut().unwrap(), |pool, node| {
                    // Binary search
                    let node = pool.get(node);
                    let left_breakpoint_x =
                        node.side[splay::Branch::Left.usize()].map_or(NEG_INF, |left| {
                            geometry::breakpoint_x(
                                graph.face_center[pool.get(left).site as usize],
                                graph.face_center[node.site as usize],
                                directrix.into(),
                            )
                        });
                    left_breakpoint_x <= px
                });
        }

        fn new_circle_event(
            &self,
            u: splay::NodeRef,
            graph: &PlanarGraph,
        ) -> Option<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)> {
            let node = self.pool.get(u);
            let prev = node.side[splay::Branch::Left.usize()]?;
            let next = node.side[splay::Branch::Right.usize()]?;

            let ps = [u, prev, next].map(|u| graph.face_center[self.pool.get(u).site as usize]);
            if geometry::signed_area(ps[0], ps[1], ps[2]) >= EPS {
                return None;
            }

            let center = geometry::circumcenter(ps[0], ps[1], ps[2])?;
            let radius = (center - ps[0]).norm_sq().sqrt();
            let y = center[1] + radius;
            let x = center[0];

            let event = Circle {
                node: u,
                side_sites: [prev, next].map(|u| self.pool.get(u).site),
            };
            Some((Reverse((y.into(), x.into())), Trivial(Event::Circle(event))))
        }

        fn is_circle_event_valid(&self, circle: &Circle) -> bool {
            let node = self.pool.get(circle.node);
            (0..2)
                .all(|b| node.side[b].map(|u| self.pool.get(u).site) == Some(circle.side_sites[b]))
        }
    }

    impl Builder {
        pub fn new() -> Self {
            Self {
                events: BinaryHeap::new(),
                beachline: Beachline {
                    pool: splay::SplayForest::new(),
                    root: None,
                },
                directrix: f64::NEG_INFINITY.into(),

                graph: PlanarGraph {
                    verts: vec![],
                    half_edges: vec![],
                    faces: vec![],

                    vert_coord: vec![],
                    face_center: vec![],
                },
                _init: false,
            }
        }

        pub fn add_points<I>(&mut self, points: I)
        where
            I: IntoIterator<Item = Point<f64>>,
        {
            assert!(!self._init, "No modification after init");
            for p in points {
                self.graph.face_center.push(p);

                let idx = self.graph.face_center.len() - 1;
                self.events.push((
                    Reverse((p[1].into(), p[0].into())),
                    Trivial(Event::Site(Site { idx: idx as u32 })),
                ));
            }
        }

        fn init(&mut self) {
            if self._init {
                return;
            }
            self._init = true;

            if self.events.is_empty() {
                return;
            }

            // create root arc (no breakpoint),
            // which has x = -infty (see eval_breakpoint_x)
            let Event::Site(Site { idx }) = self.events.pop().unwrap().1 .0 else {
                unreachable!()
            };

            self.beachline.root = Some(self.beachline.pool.add_root(splay::BreakpointNode {
                site: idx,
                left_half_edge: planar_graph::UNSET,
                ..Default::default()
            }));

            self.graph.faces = (0..self.graph.face_center.len())
                .map(|_| planar_graph::Face {
                    half_edge: planar_graph::UNSET,
                })
                .collect();
        }

        fn step(&mut self) -> bool {
            if !self._init {
                return false;
            }

            loop {
                let Some((Reverse((py, _px)), Trivial(event))) = self.events.pop() else {
                    return false;
                };
                self.directrix = py;
                match event {
                    Event::Site(Site { idx: idx_site }) => {
                        let PointNd([px, _py]) = self.graph.face_center[idx_site as usize];

                        self.beachline.splay_at_x(&self.graph, self.directrix, px);
                        let old_site = self.beachline.pool.get(self.beachline.root.unwrap()).site;

                        let left = self.beachline.root.unwrap();
                        let graph: &mut PlanarGraph = &mut self.graph;
                        let idx_half_edge = graph.half_edges.len() as u32;
                        graph.half_edges.push(HalfEdge {
                            src: planar_graph::INF,
                            face_on_left: idx_site,
                        });
                        graph.half_edges.push(HalfEdge {
                            src: planar_graph::INF,
                            face_on_left: old_site,
                        });
                        if graph.faces[old_site as usize].half_edge == planar_graph::UNSET {
                            graph.faces[old_site as usize].half_edge = idx_half_edge;
                        }
                        if graph.faces[idx_site as usize].half_edge == planar_graph::UNSET {
                            graph.faces[idx_site as usize].half_edge = idx_half_edge + 1;
                        }

                        let mid = self.beachline.pool.add_root(splay::BreakpointNode {
                            site: idx_site,
                            left_half_edge: idx_half_edge + 1,
                            ..Default::default()
                        });
                        let right = self.beachline.pool.add_root(splay::BreakpointNode {
                            site: old_site,
                            left_half_edge: idx_half_edge,
                            ..Default::default()
                        });
                        self.beachline.pool.insert_right(left, right);
                        self.beachline.pool.insert_right(left, mid);

                        self.events
                            .extend(self.beachline.new_circle_event(left, &self.graph));
                        self.events
                            .extend(self.beachline.new_circle_event(right, &self.graph));
                    }
                    Event::Circle(circle) => {
                        if !self.beachline.is_circle_event_valid(&circle) {
                            continue;
                        };

                        let Circle { node, .. } = circle;
                        self.beachline.pool.splay(node);
                        self.beachline.root = Some(node);

                        let [prev, next] = self
                            .beachline
                            .pool
                            .get(node)
                            .side
                            .map(|u| unsafe { u.unwrap_unchecked() });
                        let node = self.beachline.pool.pop_root(&mut self.beachline.root);

                        let Some(node) = node else {
                            return false;
                        };

                        // add new circle events
                        self.events
                            .extend(self.beachline.new_circle_event(prev, &self.graph));
                        self.events
                            .extend(self.beachline.new_circle_event(next, &self.graph));

                        let i_he1 = self.beachline.pool.get(node).left_half_edge;
                        let i_he2 = self.beachline.pool.get(next).left_half_edge;
                        let he1 = self.graph.half_edges[i_he1 as usize];
                        let he2 = self.graph.half_edges[i_he2 as usize];

                        let f1 = he1.face_on_left;
                        let f2 =
                            self.graph.half_edges[self.graph.twin(i_he1) as usize].face_on_left;
                        let f2_alt = he2.face_on_left;
                        let f3 =
                            self.graph.half_edges[self.graph.twin(i_he2) as usize].face_on_left;
                        debug_assert_eq!(f2, f2_alt);

                        let pf1 = self.graph.face_center[f1 as usize];
                        let pf2 = self.graph.face_center[f2 as usize];
                        let pf3 = self.graph.face_center[f3 as usize];
                        if let Some(center) = geometry::circumcenter(pf1, pf2, pf3) {
                            let vert_idx = self.graph.verts.len();
                            debug_assert_eq!(self.graph.vert_coord.len(), vert_idx);
                            debug_assert_eq!(self.graph.verts.len(), vert_idx);

                            self.graph.half_edges[i_he1 as usize].src = vert_idx as u32;
                            self.graph.half_edges[i_he2 as usize].src = vert_idx as u32;

                            let i_he3 = self.graph.half_edges.len();
                            self.graph.half_edges.push(HalfEdge {
                                src: vert_idx as u32,
                                face_on_left: self.beachline.pool.get(next).site,
                            });
                            self.graph.half_edges.push(HalfEdge {
                                src: planar_graph::INF,
                                face_on_left: self.beachline.pool.get(prev).site,
                            });
                            self.beachline
                                .pool
                                .with(next, |next| next.left_half_edge = i_he3 as u32 + 1);

                            self.graph.vert_coord.push(center);
                            self.graph.verts.push(planar_graph::Vertex {
                                half_edge: i_he3 as u32,
                            });
                        } else {
                            debug_assert_eq!(
                                self.graph.half_edges[i_he1 as usize].src,
                                planar_graph::INF
                            );
                            debug_assert_eq!(
                                self.graph.half_edges[i_he2 as usize].src,
                                planar_graph::INF
                            );
                        }
                    }
                }

                return true;
            }
        }

        pub fn run(mut self) -> PlanarGraph {
            self.init();
            while self.step() {}
            self.graph
        }
    }
}

mod kd_tree {
    // 2D static KD-tree, for nearest neighbor query
    // - Robust (no sensitive geometric predicates, except the metric)
    // - Best only for nearly-uniform distribution. Worst case O(N)

    const UNSET: u32 = !0;

    type T = i32;
    type D = i64;
    const INF_DIST_SQ: D = 1e18 as D;
    const POINT_AT_INFINITY: [T; 2] = [-1e9 as T, -1e9 as T];

    // L2 norm
    fn dist_sq(p: [T; 2], q: [T; 2]) -> D {
        let dr: [_; 2] = std::array::from_fn(|i| (p[i] - q[i]) as D);
        dr[0] * dr[0] + dr[1] * dr[1]
    }

    // No child pointers (same indexing scheme as static segment trees)
    #[derive(Clone, Debug)]
    pub struct Node {
        pub point: [T; 2],
        pub idx: u32,
    }

    pub struct KDTree {
        pub nodes: Vec<Node>,
    }

    impl KDTree {
        pub fn from_iter(points: impl IntoIterator<Item = [T; 2]>) -> Self {
            let mut points = points
                .into_iter()
                .enumerate()
                .map(|(idx, point)| Node {
                    point,
                    idx: idx as u32,
                })
                .collect::<Vec<_>>();
            let n = points.len() + 1;

            let dummy = Node {
                point: POINT_AT_INFINITY,
                idx: UNSET,
            };
            points.resize(n - 1, dummy.clone());

            let mut this = Self {
                nodes: vec![dummy; n],
            };
            this.build_rec(1, &mut points, 0);
            this
        }

        fn build_rec(&mut self, u: usize, ps: &mut [Node], depth: usize) {
            if ps.is_empty() {
                return;
            }

            let left_size = {
                let n = ps.len();
                let p = (n + 1).next_power_of_two() / 2;
                (n - p / 2).min(p - 1)
            };
            let (left, pivot, right) = ps.select_nth_unstable_by_key(left_size, |p| p.point[depth]);
            self.nodes[u] = pivot.clone();

            self.build_rec(u << 1, left, depth ^ 1);
            self.build_rec(u << 1 | 1, right, depth ^ 1);
        }

        pub fn find_nn(&self, p: [T; 2]) -> (D, &Node) {
            let mut res = (INF_DIST_SQ, 0);
            self.find_nn_rec(p, &mut res, 1, 0);
            (res.0, &self.nodes[res.1 as usize])
        }

        fn find_nn_rec(&self, p: [T; 2], opt: &mut (D, u32), u: usize, depth: usize) {
            if u >= self.nodes.len() {
                return;
            }

            let d_sq = dist_sq(self.nodes[u].point, p);
            if d_sq < opt.0 {
                *opt = (d_sq, u as u32);
            }

            let d_ax = p[depth] - self.nodes[u].point[depth];
            let branch = (d_ax > 0) as usize;

            self.find_nn_rec(p, opt, u << 1 | branch, depth ^ 1);
            if (d_ax as D * d_ax as D) < opt.0 {
                self.find_nn_rec(p, opt, u << 1 | branch ^ 1, depth ^ 1);
            }
        }
    }
}

pub mod suffix_trie {
    use std::ops::Range;

    /// O(N) representation of a suffix trie using a suffix automaton.
    /// References: https://cp-algorithms.com/string/suffix-automaton.html

    pub type T = u8;

    pub type NodeRef = u32;
    pub const UNSET: NodeRef = 1 << 30;

    // HashMap-based transition table, for generic types
    // #[derive(Clone, Debug, Default)]
    // pub struct TransitionTable(std::collections::HashMap<T, NodeRef>);

    // impl TransitionTable {
    //     pub fn get(&self, c: T) -> NodeRef {
    //         self.0.get(&c).copied().unwrap_or(UNSET)
    //     }

    //     pub fn set(&mut self, c: T, u: NodeRef) {
    //         self.0.insert(c, u);
    //     }
    // }

    // array-based transition table, for small set of alphabets.
    pub const N_ALPHABET: usize = 26;

    // Since the number of transition is O(N), Most transitions tables are slim.
    pub const STACK_CAP: usize = 1;

    #[derive(Clone, Debug)]
    pub enum TransitionTable {
        Stack([(T, NodeRef); STACK_CAP]),
        HeapFixed(Box<[NodeRef; N_ALPHABET]>),
    }

    impl TransitionTable {
        pub fn get(&self, key: T) -> NodeRef {
            match self {
                Self::Stack(items) => items
                    .iter()
                    .find_map(|&(c, u)| (c == key).then(|| u))
                    .unwrap_or(UNSET),
                Self::HeapFixed(vec) => vec[key as usize],
            }
        }
        pub fn set(&mut self, key: T, u: NodeRef) {
            match self {
                Self::Stack(arr) => {
                    for (c, v) in arr.iter_mut() {
                        if c == &(N_ALPHABET as u8) {
                            *c = key;
                        }
                        if c == &key {
                            *v = u;
                            return;
                        }
                    }

                    let mut vec = Box::new([UNSET; N_ALPHABET]);
                    for (c, v) in arr.iter() {
                        vec[*c as usize] = *v;
                    }
                    vec[key as usize] = u;
                    *self = Self::HeapFixed(vec);
                }
                Self::HeapFixed(vec) => vec[key as usize] = u,
            }
        }
    }

    impl Default for TransitionTable {
        fn default() -> Self {
            Self::Stack([(N_ALPHABET as u8, UNSET); STACK_CAP])
        }
    }

    #[derive(Debug, Default, Clone)]
    pub struct Node {
        /// DAWG of the string
        pub children: TransitionTable,

        /// Suffix tree of the reversed string (equivalent to the compressed trie of the reversed prefixes)
        pub rev_depth: u32,
        pub rev_parent: NodeRef,
        // /// An auxilary tag for the substring reconstruction
        // pub first_endpos: u32,
    }

    impl Node {
        // pub fn is_rev_terminal(&self) -> bool {
        //     self.first_endpos + 1 == self.rev_depth
        // }
    }

    pub struct SuffixAutomaton {
        pub nodes: Vec<Node>,
        pub tail: NodeRef,
    }

    impl SuffixAutomaton {
        pub fn new() -> Self {
            let root = Node {
                children: Default::default(),
                rev_parent: UNSET,
                rev_depth: 0,
                // first_endpos: UNSET,
            };

            Self {
                nodes: vec![root],
                tail: 0,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let u = self.nodes.len() as u32;
            self.nodes.push(node);
            u
        }

        pub fn push(&mut self, x: T) {
            let u = self.alloc(Node {
                children: Default::default(),
                rev_parent: 0,
                rev_depth: self.nodes[self.tail as usize].rev_depth + 1,
                // first_endpos: self.nodes[self.tail as usize].rev_depth,
            });
            let mut p = self.tail;
            self.tail = u;

            while p != UNSET {
                let c = self.nodes[p as usize].children.get(x);
                if c != UNSET {
                    if self.nodes[p as usize].rev_depth + 1 == self.nodes[c as usize].rev_depth {
                        self.nodes[u as usize].rev_parent = c;
                    } else {
                        let c_cloned = self.alloc(Node {
                            rev_depth: self.nodes[p as usize].rev_depth + 1,
                            ..self.nodes[c as usize].clone()
                        });

                        self.nodes[u as usize].rev_parent = c_cloned;
                        self.nodes[c as usize].rev_parent = c_cloned;

                        while p != UNSET && self.nodes[p as usize].children.get(x) == c {
                            self.nodes[p as usize].children.set(x, c_cloned);
                            p = self.nodes[p as usize].rev_parent;
                        }
                    }

                    break;
                }

                self.nodes[p as usize].children.set(x, u);
                p = self.nodes[p as usize].rev_parent;
            }
        }

        pub fn substr_range(&self, u: NodeRef) -> Range<usize> {
            unimplemented!()
        }

        pub fn lcs_with<'a>(&self, t: &'a [T], mut f: impl FnMut(&T) -> T) -> &'a [T] {
            let mut u = 0;
            let mut len = 0;
            let mut lcs_len = 0;
            let mut lcs_end = 0;

            for (i, b) in t.iter().map(|b| f(b)).enumerate() {
                while u != 0 && self.nodes[u as usize].children.get(b) == UNSET {
                    u = self.nodes[u as usize].rev_parent;
                    len = self.nodes[u as usize].rev_depth as usize;
                }
                let c = self.nodes[u as usize].children.get(b);
                if c != UNSET {
                    u = c;
                    len += 1;
                }
                if len > lcs_len {
                    lcs_len = len;
                    lcs_end = i;
                }
            }

            let lcs = &t[lcs_end - lcs_len + 1..=lcs_end];
            lcs
        }
    }
}

mod dset {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

fn parse_u8(b: u8) -> u8 {
    b - b'a'
}

fn ordered_pair(u: u32, v: u32) -> (u32, u32) {
    if u < v {
        (u, v)
    } else {
        (v, u)
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let q = input.value::<usize>() + input.value::<usize>();
    let mut ps = vec![];
    let mut names = vec![];
    let mut automatons = vec![];
    for _ in 0..n {
        ps.push([input.i32(), input.i32()]);

        let s = input.token();
        let mut automaton = suffix_trie::SuffixAutomaton::new();
        for b in s.bytes().map(parse_u8) {
            automaton.push(b);
        }
        names.push(s);
        automatons.push(automaton);
    }

    let mut builder = voronoi::Builder::new();
    builder.add_points(ps.iter().map(|&[x, y]| Point::new(x as f64, y as f64)));
    let graph = builder.run();
    let planar_graph::PlanarGraph {
        verts,
        half_edges,
        faces,
        vert_coord,
        face_center,
    } = graph;

    let mut neighbors = vec![vec![]; n];
    for i_h1 in (0..half_edges.len()).step_by(2) {
        let h1 = &half_edges[i_h1];
        let h2 = &half_edges[i_h1 ^ 1];
        let mut f1 = h1.face_on_left;
        let mut f2 = h2.face_on_left;
        if names[f1 as usize].len() < names[f2 as usize].len() {
            std::mem::swap(&mut f1, &mut f2);
        }
        let weight = automatons[f1 as usize]
            .lcs_with(names[f2 as usize].as_bytes(), |&b| parse_u8(b))
            .len() as i32;
        neighbors[f1 as usize].push((f2, weight));
        neighbors[f2 as usize].push((f1, weight));
    }

    // for u in 1..n {
    //     let w_max = neighbors[u].iter().map(|&(_, w)| w).max().unwrap();
    //     let (visited, unvisited): (Vec<_>, Vec<_>) = neighbors[u]
    //         .iter()
    //         .filter(|&&(_, w)| w == w_max)
    //         .map(|&(v, _)| v)
    //         .partition(|&v| (v as usize) < u);
    //     let v = *visited
    //         .iter()
    //         .max_by_key(|&v| v)
    //         .or_else(|| unvisited.iter().min_by_key(|&v| v))
    //         .unwrap();

    //     parent[u] = v;
    // }

    // for u in 0..n {
    //     neighbors[u].sort_unstable_by_key(|&(v, _)| v);
    // }

    println!("{:?}", neighbors);

    let kd_tree = kd_tree::KDTree::from_iter(ps.iter().copied());
}
