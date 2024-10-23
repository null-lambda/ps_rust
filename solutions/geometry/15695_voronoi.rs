// Interactive visualization:
// https://null-lambda.github.io/algo-vis-rust/voronoi-fortune/
//
use geometry::Point;
use std::io::Write;

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

pub mod cmp {
    use std::cmp::Ordering;

    // x <= y iff x = y
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        #[inline]
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        #[inline]
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        #[inline]
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}

#[allow(dead_code)]
#[macro_use]
pub mod geometry {
    use core::f64;
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
        fn map<F, S>(self, mut f: F) -> PointNd<N, S>
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
            x = xm - c * 0.5 / b_2;
            if !x.is_finite() {
                x = xm;
            }
            if dx_half < 0.0 {
                x = f64::INFINITY;
            }
        }
        // println!(
        //     "left: {:?}, right: {:?}, sweepline: {}",
        //     left, right, sweepline
        // );
        // println!("a: {}, b_2: {}, c: {}", a, b_2, c);
        // println!("x = {}, det = {}", x, det_4);
        x
    }

    pub fn polygon_contains<T: Scalar>(ps: &[Point<T>], q: Point<T>) -> bool {
        // O(n) ray casting (+x direction)
        let edges = ps.iter().zip(ps[1..].iter().chain(ps.first()));

        let mut n_inter = 0;
        for (&e1, &e2) in edges {
            let s = signed_area(e1, e2, q);
            if e1[1] <= q[1] && q[1] < e2[1] && s > T::zero()
                || e2[1] <= q[1] && q[1] < e1[1] && s < T::zero()
            {
                n_inter += 1;
            }
        }
        n_inter % 2 == 1
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

// Doubly connected edge list (WIP)
pub mod graph {
    use super::geometry::Point;

    pub const UNSET: usize = 1 << 31;
    pub const INF: usize = 1 << 30;

    #[derive(Debug, Copy, Clone)]
    pub struct Vertex {
        pub half_edge: usize,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct HalfEdge {
        pub vert: usize,
        pub face_left: usize,
        pub twin: usize,
        // pub next: usize, // TODO!
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Face {
        pub half_edge: usize,
    }

    #[derive(Debug, Default)]
    pub struct Topology {
        pub verts: Vec<Vertex>,
        pub half_edges: Vec<HalfEdge>,
        pub faces: Vec<Face>,
    }

    #[derive(Debug, Default)]
    pub struct Graph {
        pub topo: Topology,
        pub vert_coord: Vec<Point<f64>>,
        pub face_center: Vec<Point<f64>>,
    }
}

#[cfg(not(debug_assertions))]
pub mod tag {
    pub type Tag = ();
    pub fn next() -> Tag {}
}

pub mod builder {
    pub mod splay {
        // bottom-up splay tree for beachline
        use super::super::tag;

        use std::{
            cmp::Ordering,
            fmt, iter,
            ptr::{self, NonNull},
        };

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Branch {
            Left = 0,
            Right = 1,
        }

        use Branch::{Left, Right};

        impl Branch {
            pub fn inv(&self) -> Self {
                match self {
                    Branch::Left => Branch::Right,
                    Branch::Right => Branch::Left,
                }
            }

            pub fn iter() -> iter::Chain<iter::Once<Self>, iter::Once<Self>> {
                iter::once(Branch::Left).chain(iter::once(Branch::Right))
            }
        }

        impl From<usize> for Branch {
            fn from(x: usize) -> Self {
                match x {
                    0 => Left,
                    1 => Right,
                    _ => panic!(),
                }
            }
        }

        type Link = Option<NonNull<Node>>;

        #[derive(Debug)]
        pub struct Breakpoint {
            pub site: usize,
            pub left_half_edge: usize,
        }

        pub struct Node {
            tag: tag::Tag,
            pub children: [Link; 2], // binary search tree structure
            pub side: [Link; 2],     // linked list structure
            pub parent: Link,
            pub value: Breakpoint,
        }

        impl Node {
            pub fn new(value: Breakpoint) -> Self {
                Self {
                    tag: tag::next(),
                    children: [None, None],
                    side: [None, None],
                    parent: None,
                    value,
                }
            }

            pub fn new_nonnull(value: Breakpoint) -> NonNull<Self> {
                unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Self::new(value)))) }
            }

            fn link_sides(mut lhs: NonNull<Self>, mut rhs: NonNull<Self>) {
                unsafe {
                    lhs.as_mut().side[Right as usize] = Some(rhs);
                    rhs.as_mut().side[Left as usize] = Some(lhs);
                }
            }

            fn attach(&mut self, branch: Branch, child: Option<NonNull<Self>>) {
                unsafe {
                    debug_assert_ne!(Some(self as *mut _), child.map(|x| x.as_ptr()));
                    self.children[branch as usize] = child;
                    if let Some(mut child) = child {
                        child.as_mut().parent = Some(self.into());
                    }
                }
            }

            fn detach(&mut self, branch: Branch) -> Option<NonNull<Self>> {
                unsafe {
                    self.children[branch as usize].take().map(|mut child| {
                        child.as_mut().parent = None;
                        child
                    })
                }
            }

            fn branch(node: NonNull<Self>) -> Option<(Branch, NonNull<Node>)> {
                unsafe {
                    node.as_ref().parent.map(|parent| {
                        let branch = match parent.as_ref().children[Branch::Left as usize] {
                            Some(child) if ptr::eq(node.as_ptr(), child.as_ptr()) => Branch::Left,
                            _ => Branch::Right,
                        };
                        (branch, parent)
                    })
                }
            }

            fn rotate(mut node: NonNull<Self>) -> Option<()> {
                unsafe {
                    let (branch, mut parent) = Node::branch(node)?;

                    let child = node.as_mut().detach(branch.inv());
                    parent.as_mut().attach(branch, child);

                    if let Some((grandbranch, mut grandparent)) = Node::branch(parent) {
                        grandparent.as_mut().attach(grandbranch, Some(node));
                    } else {
                        node.as_mut().parent = None;
                    }
                    node.as_mut().attach(branch.inv(), Some(parent));

                    Some(())
                }
            }

            pub fn validate_parents(node: NonNull<Self>) {
                unsafe {
                    if let Some((branch, parent)) = Node::branch(node) {
                        debug_assert_eq!(
                            node.as_ptr(),
                            parent.as_ref().children[branch as usize].unwrap().as_ptr(),
                            "Parent's child pointer does not point to self"
                        );
                    }
                    for branch in Branch::iter() {
                        if let Some(child) = node.as_ref().children[branch as usize] {
                            debug_assert_eq!(
                                node.as_ptr(),
                                child.as_ref().parent.unwrap().as_ptr(),
                                "Child's parent pointer does not point to self: {:?} {:?}",
                                node,
                                child
                            );
                            debug_assert_ne!(child.as_ptr(), node.as_ptr(), "Self loop detected");
                        }
                    }
                }
            }

            pub fn splay(root: &mut NonNull<Node>, node: NonNull<Node>) {
                while let Some((branch, parent)) = Node::branch(node) {
                    if let Some((grandbranch, _)) = Node::branch(parent) {
                        if branch == grandbranch {
                            Node::rotate(parent);
                        } else {
                            Node::rotate(node);
                        }
                    }
                    Node::rotate(node);
                }
                *root = node;
            }

            // splay the last truthy element in [true, ..., true, false, ..., false]
            // if there is no such element, return false
            pub fn splay_last_by<F>(root: &mut NonNull<Node>, pred: F) -> bool
            where
                F: Fn(&Node) -> bool,
            {
                unsafe {
                    let mut current = *root;
                    let mut last_pred;
                    loop {
                        last_pred = pred(current.as_ref());
                        let branch = if last_pred { Right } else { Left };
                        let Some(child) = current.as_ref().children[branch as usize] else {
                            break;
                        };
                        current = child;
                    }
                    if !last_pred {
                        let Some(left) = current.as_ref().side[Left as usize] else {
                            return false;
                        };
                        current = left;
                    }
                    debug_assert!(
                        pred(current.as_ref())
                            && current.as_ref().children[Right as usize]
                                .map_or(true, |x| !pred(x.as_ref()))
                    );
                    Node::splay(root, current);
                    true
                }
            }

            pub fn splay_by<F>(root: &mut NonNull<Node>, cmp: F)
            where
                F: Fn(&Node) -> Ordering,
            {
                unsafe {
                    let mut current = *root;
                    loop {
                        let branch = match cmp(current.as_ref()) {
                            Ordering::Less => Left,
                            Ordering::Greater => Right,
                            Ordering::Equal => break,
                        };
                        let Some(child) = current.as_ref().children[branch as usize] else {
                            break;
                        };
                        current = child;
                    }
                    Node::splay(root, current);
                }
            }

            pub fn splay_first(root: &mut NonNull<Node>) {
                Node::splay_by(root, |_| Ordering::Less);
            }

            pub fn splay_last(root: &mut NonNull<Node>) {
                Node::splay_by(root, |_| Ordering::Greater);
            }

            pub fn insert_right(mut root: NonNull<Node>, mut new_node: NonNull<Node>) {
                unsafe {
                    let right = root.as_mut().children[Right as usize];
                    if right.is_some() {
                        let next = root.as_ref().side[Right as usize].unwrap();
                        Node::link_sides(new_node, next);
                    }
                    Node::link_sides(root, new_node);

                    root.as_mut().attach(Right, Some(new_node));
                    new_node.as_mut().attach(Right, right);
                }
            }

            pub fn pop_root(root: &mut Option<NonNull<Node>>) -> Option<NonNull<Node>> {
                unsafe {
                    let mut old_root = (*root)?;

                    let left = old_root.as_mut().detach(Left);
                    let right = old_root.as_mut().detach(Right);

                    match (left, right) {
                        (Some(_), Some(_)) => {
                            let prev = old_root.as_ref().side[Left as usize].unwrap();
                            let next = old_root.as_ref().side[Right as usize].unwrap();
                            Node::link_sides(prev, next);

                            let mut left = left.unwrap();
                            // Node::splay_last(&mut left);
                            Node::splay(&mut left, prev);
                            debug_assert!(left.as_ref().children[Right as usize].is_none());

                            left.as_mut().attach(Right, right);
                            *root = Some(left);
                        }
                        (Some(_), None) => {
                            let mut left = left.unwrap();
                            Node::splay_last(&mut left);
                            left.as_mut().side[Right as usize] = None;
                            *root = Some(left);
                        }
                        (None, Some(_)) => {
                            let mut right = right.unwrap();
                            Node::splay_first(&mut right);
                            right.as_mut().side[Left as usize] = None;
                            *root = Some(right);
                        }
                        (None, None) => {
                            *root = None;
                        }
                    }
                    old_root.as_mut().side = [None, None];
                    Some(old_root)
                }
            }

            // Test whether the linked list structure is valid
            pub fn validate_side_links(root: NonNull<Node>) {
                if !cfg!(debug_assertions) {
                    return;
                }

                unsafe {
                    // grab all nodes in inorder
                    let mut inorder = vec![];
                    Node::inorder(root, |node| inorder.push(node));

                    // check the linked list structure
                    for i in 0..inorder.len() {
                        let node: NonNull<Node> = inorder[i];
                        let prev = (i >= 1).then(|| inorder[i - 1]);
                        let next = (i + 1 < inorder.len()).then(|| inorder[i + 1]);

                        fn option_nonnull_to_ptr<T>(x: Option<NonNull<T>>) -> *const T {
                            x.map_or(ptr::null(), |x| x.as_ptr())
                        }

                        debug_assert!(ptr::eq(
                            option_nonnull_to_ptr(node.as_ref().side[Left as usize]),
                            option_nonnull_to_ptr(prev)
                        ));
                        debug_assert!(
                            ptr::eq(
                                option_nonnull_to_ptr(node.as_ref().side[Right as usize]),
                                option_nonnull_to_ptr(next),
                            ),
                            "side_right: {:?}, inorder_right: {:?}",
                            node.as_ref().side[Right as usize],
                            next
                        );
                        if let Some(prev) = prev {
                            debug_assert!(ptr::eq(
                                option_nonnull_to_ptr(prev.as_ref().side[Right as usize]),
                                node.as_ref()
                            ));
                        }
                        if let Some(next) = next {
                            debug_assert!(ptr::eq(
                                option_nonnull_to_ptr(next.as_ref().side[Left as usize]),
                                node.as_ref()
                            ));
                        }
                    }
                }
            }

            pub fn inorder<F>(root: NonNull<Node>, mut visitor: F)
            where
                F: FnMut(NonNull<Node>),
            {
                pub fn inner<F>(node: NonNull<Node>, visitor: &mut F)
                where
                    F: FnMut(NonNull<Node>),
                {
                    unsafe {
                        if let Some(left) = node.as_ref().children[Left as usize] {
                            inner(left, visitor);
                        }
                        visitor(node);
                        if let Some(right) = node.as_ref().children[Right as usize] {
                            inner(right, visitor);
                        }
                    }
                }

                inner(root, &mut visitor);
            }
        }

        impl fmt::Debug for Node {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                unsafe {
                    write!(
                        f,
                        "{} {:?} {}",
                        self.children[Left as usize]
                            .as_ref()
                            .map_or("_".to_owned(), |x| format!("({:?})", x.as_ref())),
                        self.tag,
                        self.children[Right as usize]
                            .as_ref()
                            .map_or("_".to_owned(), |x| format!("({:?})", x.as_ref())),
                    )
                }
            }
        }
    }

    use core::f64;
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, HashSet},
        fmt,
        ptr::NonNull,
    };

    use splay::{Branch, Node};

    use super::{
        cmp::Trivial,
        geometry::{self, Ordered, Point, PointNd},
    };

    use super::graph::{self, HalfEdge};

    #[derive(Debug, Clone)]
    pub enum Event {
        Site(Site),
        Circle(Circle),
    }

    #[derive(Debug, Clone)]
    pub struct Site {
        pub idx: usize,
    }

    #[derive(Clone)]
    pub struct Circle {
        pub node: NonNull<Node>,
        pub prev_idx: usize,
        pub next_idx: usize,
    }

    impl fmt::Debug for Circle {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unsafe {
                write!(
                    f,
                    "Circle {{ node: {:?}, left: {:?}, right: {:?} }}",
                    self.node.as_ref(),
                    self.prev_idx,
                    self.next_idx,
                )
            }
        }
    }

    #[derive(Debug)]
    pub struct Builder {
        pub events: BinaryHeap<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)>,
        pub added_circles: HashSet<[usize; 3]>, // double-checker for robustness
        pub beachline: NonNull<splay::Node>,
        pub directrix: Ordered<f64>,
        pub graph: graph::Graph,
        pub beachline_node_pool: Vec<Box<Node>>,
        _init: bool,
    }

    fn allocate_node(pool: &mut Vec<Box<Node>>, value: splay::Breakpoint) -> NonNull<Node> {
        pool.push(Box::new(Node::new(value)));
        unsafe { NonNull::new_unchecked(pool.last_mut().unwrap().as_mut()) }
    }

    fn eval_left_breakpoint_x(
        graph: &graph::Graph,
        sweep: Ordered<f64>,
        node: &Node,
    ) -> Ordered<f64> {
        let left = node.side[Branch::Left as usize];
        let left_x: Ordered<f64> = left
            .map_or(f64::NEG_INFINITY, |left| unsafe {
                geometry::breakpoint_x(
                    graph.face_center[left.as_ref().value.site],
                    graph.face_center[node.value.site],
                    sweep.into(),
                )
            })
            .into();
        left_x
    }

    fn new_circle_event(
        node: NonNull<Node>,
        graph: &graph::Graph,
        added_circles: &mut HashSet<[usize; 3]>,
    ) -> Option<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)> {
        unsafe {
            let prev = node.as_ref().side[Branch::Left as usize]?;
            let next = node.as_ref().side[Branch::Right as usize]?;

            let mut indices = [
                node.as_ref().value.site,
                prev.as_ref().value.site,
                next.as_ref().value.site,
            ];

            let p0 = graph.face_center[indices[0]];
            let p1 = graph.face_center[indices[1]];
            let p2 = graph.face_center[indices[2]];

            if geometry::signed_area(p0, p1, p2) > 1e-15 {
                return None;
            }
            indices.sort();
            if !added_circles.insert(indices) {
                return None;
            }

            let center = geometry::circumcenter(p0, p1, p2)?;
            // .unwrap_or(Point::new(0.0, f64::INFINITY));
            let radius = (center - p0).norm_sq().sqrt();
            let y = center[1] + radius;
            let x = center[0];

            let event = Circle {
                node,
                prev_idx: prev.as_ref().value.site,
                next_idx: next.as_ref().value.site,
            };
            Some((Reverse((y.into(), x.into())), Trivial(Event::Circle(event))))
        }
    }

    pub fn check_circle_event(circle: &Circle) -> bool {
        unsafe {
            let Some(prev) = circle.node.as_ref().side[Branch::Left as usize] else {
                return false;
            };
            let Some(next) = circle.node.as_ref().side[Branch::Right as usize] else {
                return false;
            };
            prev.as_ref().value.site == circle.prev_idx
                && next.as_ref().value.site == circle.next_idx
        }
    }

    impl Builder {
        pub fn new() -> Self {
            Self {
                events: BinaryHeap::new(),
                added_circles: HashSet::new(),
                beachline: NonNull::dangling(),
                beachline_node_pool: vec![],
                directrix: f64::NEG_INFINITY.into(),
                graph: Default::default(),
                _init: false,
            }
        }

        pub fn add_points<I>(&mut self, points: I)
        where
            I: IntoIterator<Item = Point<f64>>,
        {
            debug_assert!(!self._init, "No modification after init");
            for p in points {
                self.graph.face_center.push(p);

                let idx = self.graph.face_center.len() - 1;
                self.events.push((
                    Reverse((p[1].into(), p[0].into())),
                    Trivial(Event::Site(Site { idx })),
                ));
            }
        }

        pub fn init(&mut self) {
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

            self.beachline = splay::Node::new_nonnull(splay::Breakpoint {
                site: idx,
                left_half_edge: graph::UNSET,
            });

            self.graph.topo.faces = (0..self.graph.face_center.len())
                .map(|_| graph::Face {
                    half_edge: graph::UNSET,
                })
                .collect();
        }

        pub fn step(&mut self) -> bool {
            // debug_assert!(
            //     self._init,
            //     "Builder::init must be called before Builder::step"
            // );
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
                        let PointNd([px, _py]) = self.graph.face_center[idx_site];

                        Node::splay_last_by(&mut self.beachline, |node| {
                            eval_left_breakpoint_x(&self.graph, self.directrix, node)
                                <= Ordered::from(px)
                        });
                        let old_site = unsafe { self.beachline.as_ref().value.site };
                        // unsafe {
                        //     println!("{:?}", &self.beachline.as_ref());
                        // }

                        // println!("coord: {:?}", self.graph.face_center[idx_site]);
                        // println!("site: {:?}, directrix: {:?}", idx_site, self.directrix);
                        // println!("old_site: {:?}", old_site);
                        // println!(
                        //     "{:?}', eval_left: {:?}",
                        //     unsafe { self.beachline.as_ref() },
                        //     eval_left_breakpoint_x(&self.graph, self.directrix, unsafe {
                        //         self.beachline.as_ref()
                        //     })
                        // );

                        let left = self.beachline;
                        let pool: &mut Vec<Box<Node>> = &mut self.beachline_node_pool;
                        let graph: &mut graph::Graph = &mut self.graph;
                        let idx_half_edge = graph.topo.half_edges.len();
                        graph.topo.half_edges.push(HalfEdge {
                            vert: graph::INF,
                            face_left: idx_site,
                            twin: idx_half_edge + 1,
                        });
                        graph.topo.half_edges.push(HalfEdge {
                            vert: graph::INF,
                            face_left: old_site,
                            twin: idx_half_edge,
                        });
                        // if graph.topo.faces[old_site].half_edge == graph::UNSET {
                        //     graph.topo.faces[old_site].half_edge = idx_half_edge;
                        // }
                        // if graph.topo.faces[idx_site].half_edge == graph::UNSET {
                        //     graph.topo.faces[idx_site].half_edge = idx_half_edge + 1;
                        // }
                        let mid = allocate_node(
                            pool,
                            splay::Breakpoint {
                                site: idx_site,
                                left_half_edge: idx_half_edge + 1,
                            },
                        );
                        let right = allocate_node(
                            pool,
                            splay::Breakpoint {
                                site: old_site,
                                left_half_edge: idx_half_edge,
                            },
                        );
                        Node::insert_right(self.beachline, right);
                        Node::insert_right(self.beachline, mid);

                        self.events.extend(new_circle_event(
                            left,
                            &self.graph,
                            &mut self.added_circles,
                        ));
                        self.events.extend(new_circle_event(
                            right,
                            &self.graph,
                            &mut self.added_circles,
                        ));
                    }
                    Event::Circle(circle) => unsafe {
                        if !check_circle_event(&circle) {
                            continue;
                        };

                        let Circle { node, .. } = circle;
                        Node::splay(&mut self.beachline, node);

                        let mut next = node.as_ref().side[Branch::Right as usize].unwrap();
                        let prev = node.as_ref().side[Branch::Left as usize].unwrap();

                        let mut beachline: Option<NonNull<Node>> = Some(self.beachline);
                        Node::pop_root(&mut beachline);
                        let Some(beachline) = beachline else {
                            return false;
                        };
                        self.beachline = beachline;

                        // add new circle events
                        self.events.extend(
                            new_circle_event(prev, &self.graph, &mut self.added_circles)
                                .into_iter(),
                        );
                        self.events.extend(
                            new_circle_event(next, &self.graph, &mut self.added_circles)
                                .into_iter(),
                        );

                        Node::validate_side_links(self.beachline);
                        Node::validate_parents(self.beachline);

                        let i_he1 = node.as_ref().value.left_half_edge;
                        let i_he2 = next.as_ref().value.left_half_edge;
                        let he1 = self.graph.topo.half_edges[i_he1];
                        let he2 = self.graph.topo.half_edges[i_he2];

                        let f1 = he1.face_left;
                        let f2 = self.graph.topo.half_edges[he1.twin].face_left;
                        let f2_alt = he2.face_left;
                        let f3 = self.graph.topo.half_edges[he2.twin].face_left;
                        debug_assert_eq!(f2, f2_alt);

                        let pf1 = self.graph.face_center[f1];
                        let pf2 = self.graph.face_center[f2];
                        let pf3 = self.graph.face_center[f3];
                        if let Some(center) = geometry::circumcenter(pf1, pf2, pf3) {
                            let vert_idx = self.graph.topo.verts.len();
                            debug_assert_eq!(self.graph.vert_coord.len(), vert_idx);

                            self.graph.topo.half_edges[i_he1].vert = vert_idx;
                            self.graph.topo.half_edges[i_he2].vert = vert_idx;

                            let he3_idx = self.graph.topo.half_edges.len();
                            self.graph.topo.half_edges.push(HalfEdge {
                                vert: vert_idx,
                                face_left: next.as_ref().value.site,
                                twin: he3_idx + 1,
                            });
                            self.graph.topo.half_edges.push(HalfEdge {
                                vert: graph::INF,
                                face_left: prev.as_ref().value.site,
                                twin: he3_idx,
                            });
                            next.as_mut().value.left_half_edge = he3_idx + 1;

                            self.graph.vert_coord.push(center);
                            self.graph
                                .topo
                                .verts
                                .push(graph::Vertex { half_edge: he3_idx });
                        } else {
                            self.graph.topo.half_edges[i_he1].vert = graph::INF;
                            self.graph.topo.half_edges[i_he2].vert = graph::INF;
                        }
                    },
                }
                Node::validate_parents(self.beachline);
                Node::validate_side_links(self.beachline);

                return true;
            }
        }

        pub fn run(&mut self) {
            self.init();
            while self.step() {}
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let points_i64 = (0..input.value()).map(|_| {
        let x = input.value();
        let y = input.value();
        Point::<i64>::new(x, y)
    });
    let points_f64 = points_i64
        .map(|p| Point::new(p[0] as f64, p[1] as f64))
        .collect::<Vec<_>>();
    let mut builder = builder::Builder::new();
    builder.add_points(points_f64.iter().cloned());
    builder.run();

    let graph::Graph {
        topo:
            graph::Topology {
                verts,
                half_edges,
                faces,
            },
        vert_coord,
        face_center,
    } = &builder.graph;

    let mut ans = 0.0f64;
    for v in vert_coord {
        // println!("{:?}", (v, geometry::polygon_contains(&points_f64, *v)));
        if !geometry::polygon_contains(&points_f64, *v) {
            continue;
        }
        let r_min = points_f64
            .iter()
            .map(|&p| (p - *v).norm_sq())
            .min_by_key(|&x| geometry::Ordered::from(x))
            .unwrap()
            .sqrt();
        // dbg!(r_min);
        ans = ans.max(r_min);
    }

    let edges =
        || (0..points_f64.len()).map(|i| (points_f64[i], points_f64[(i + 1) % points_f64.len()]));
    for (i_h1, h1) in half_edges.iter().enumerate() {
        let i_h2 = h1.twin;
        if i_h1 > i_h2 {
            continue;
        }
        let h2 = &half_edges[i_h2];

        let pf1 = face_center[h1.face_left];
        let pf2 = face_center[h2.face_left];
        let pm = (pf1 + pf2) * 0.5;
        let mut ph = (pf2 - pf1).rot();
        ph = ph * ph.norm_sq().recip();

        let pv1 = if h1.vert != graph::INF {
            vert_coord[h1.vert]
        } else {
            pm + ph * 5e4
        };
        let pv2 = if h2.vert != graph::INF {
            vert_coord[h2.vert]
        } else {
            pm - ph * 5e4
        };

        for (q1, q2) in edges() {
            let Some(inter) = geometry::line_intersection(pv1, pv2 - pv1, q1, q2 - q1) else {
                continue;
            };

            let lv = (pv2 - pv1).norm_sq();
            let lq = (q2 - q1).norm_sq();
            let dv1 = (inter - pv1).norm_sq();
            let dv2 = (inter - pv2).norm_sq();
            let dq1 = (inter - q1).norm_sq();
            let dq2 = (inter - q2).norm_sq();
            let tol = 1e-19;
            if !(lv - dv1 >= tol && lv - dv2 >= -tol && lq - dq1 >= -tol && lq - dq2 >= -tol) {
                continue;
            }
            let r_min = points_f64
                .iter()
                .map(|&p| (p - inter).norm_sq())
                .min_by_key(|&x| geometry::Ordered::from(x))
                .unwrap()
                .sqrt();
            // dbg!(r_min);
            ans = ans.max(r_min);
        }
    }
    writeln!(output, "{}", ans).ok();
}
