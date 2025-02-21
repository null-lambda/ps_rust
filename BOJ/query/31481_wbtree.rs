use std::{cmp::Ordering, io::Write};

use wbtree::persistent as wb;

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

pub mod rc_acyclic {
    // Shared rc pointers without weak references.
    // Empirically verified with miri, on BOJ 17486
    // Further optimizations: implement a static bump-allocator with zero-sized types

    // use std::ops::Deref;
    //     pub trait Rc<T: Clone>: Deref<Target = T> + Clone {
    //         fn new(value: T) -> Self;
    //         fn make_mut(&mut self) -> &mut T;
    //         fn try_unwrap(self) -> Result<T, Self>;
    //         fn unwrap_or_clone(self) -> T {
    //             Self::try_unwrap(self).unwrap_or_else(|this| (*this).clone())
    //         }
    //     }

    //     impl<T: Clone> Rc<T> for std::rc::Rc<T> {
    //         fn new(value: T) -> Self {
    //             std::rc::Rc::new(value)
    //         }

    //         fn make_mut(&mut self) -> &mut T {
    //             std::rc::Rc::make_mut(self)
    //         }

    //         fn try_unwrap(self) -> Result<T, Self> {
    //             std::rc::Rc::try_unwrap(self)
    //         }

    //         fn unwrap_or_clone(self) -> T {
    //             std::rc::Rc::unwrap_or_clone(self)
    //         }
    //     }

    use std::{cell::Cell, mem::MaybeUninit, ops::Deref, ptr::NonNull};

    #[allow(non_camel_case_types)]
    pub type ucount = u32;

    pub struct RcInner<T: ?Sized> {
        strong_count: Cell<ucount>,
        value: T,
    }

    pub struct Rc<T: ?Sized> {
        ptr: NonNull<RcInner<T>>,
        _marker: std::marker::PhantomData<Box<T>>,
    }

    impl<T> Rc<T> {
        #[inline]
        pub fn new(value: T) -> Self {
            let inner = RcInner {
                value,
                strong_count: Cell::new(1),
            };
            let inner = Box::new(inner);
            unsafe {
                Self {
                    ptr: NonNull::new_unchecked(Box::leak(inner)),
                    _marker: Default::default(),
                }
            }
        }

        #[inline]
        pub fn strong_count(&self) -> usize {
            unsafe { self.ptr.as_ref().strong_count.get() as usize }
        }
    }

    impl<T: ?Sized> Drop for Rc<T> {
        fn drop(&mut self) {
            let inner = unsafe { self.ptr.as_ref() };
            inner
                .strong_count
                .set(inner.strong_count.get().wrapping_sub(1));

            if inner.strong_count.get() == 0 {
                unsafe {
                    let _ = Box::from_raw(self.ptr.as_ptr());
                }
            }
        }
    }

    impl<T: ?Sized> Clone for Rc<T> {
        #[inline]
        fn clone(&self) -> Self {
            let inner = unsafe { self.ptr.as_ref() };
            inner
                .strong_count
                .set(inner.strong_count.get().wrapping_add(1));
            Self {
                ptr: self.ptr,
                _marker: Default::default(),
            }
        }
    }

    impl<T: ?Sized> Deref for Rc<T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &T {
            unsafe { &self.ptr.as_ref().value }
        }
    }

    impl<T: ?Sized + Clone> Rc<T> {
        pub fn make_mut(&mut self) -> &mut T {
            if Rc::strong_count(self) != 1 {
                let mut buffer: Box<MaybeUninit<RcInner<T>>> = Box::new(MaybeUninit::uninit());
                unsafe {
                    std::ptr::write(
                        buffer.as_mut_ptr(),
                        RcInner {
                            value: T::clone(self),
                            strong_count: Cell::new(1),
                        },
                    );
                    let ptr = Box::into_raw(buffer) as *mut RcInner<T>;
                    *self = Rc {
                        _marker: Default::default(),
                        ptr: NonNull::new_unchecked(ptr),
                    };
                }
            }
            &mut unsafe { self.ptr.as_mut() }.value
        }

        pub fn try_unwrap(self) -> Result<T, Self> {
            if Rc::strong_count(&self) == 1 {
                unsafe {
                    let inner = Box::from_raw(self.ptr.as_ptr());
                    std::mem::forget(self);
                    Ok(inner.value)
                }
            } else {
                Err(self)
            }
        }

        pub fn unwrap_or_clone(self) -> T {
            Self::try_unwrap(self).unwrap_or_else(|this| (*this).clone())
        }
    }

    impl<T: ?Sized + std::fmt::Display> std::fmt::Display for Rc<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            std::fmt::Display::fmt(&**self, f)
        }
    }

    impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for Rc<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            std::fmt::Debug::fmt(&**self, f)
        }
    }
}

pub mod wbtree {
    pub mod persistent {
        // Weight-balanced tree
        // https://koosaga.com/342
        // https://yoichihirai.com/bst.pdf

        use std::{cmp::Ordering, ops::Range};

        // pub type Rc<T> = std::rc::Rc<T>;
        pub type Rc<T> = crate::rc_acyclic::Rc<T>;

        #[derive(Debug)]
        pub struct Link<Node: ?Sized> {
            pub size: u32,
            pub children: Option<[Rc<Node>; 2]>,
        }

        impl<Node: ?Sized> Clone for Link<Node> {
            fn clone(&self) -> Self {
                Self {
                    size: self.size,
                    children: self
                        .children
                        .as_ref()
                        .map(|cs| [Rc::clone(&cs[0]), Rc::clone(&cs[1])]),
                }
            }
        }

        impl<Node: ?Sized> Default for Link<Node> {
            fn default() -> Self {
                Self {
                    size: 1,
                    children: None,
                }
            }
        }

        pub trait IntrusiveNode {
            fn link(&self) -> &Link<Self>;
            fn link_mut(&mut self) -> &mut Link<Self>;

            fn pull_up_link(&mut self) {
                let Some([left, right]) = self.link().children.as_ref() else {
                    return;
                };
                self.link_mut().size = left.link().size + right.link().size;
            }

            // type Cx;
            // fn push_down(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
            // fn pull_up(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
        }

        pub trait NodeSpec: Default + Clone + IntrusiveNode {
            fn push_down(&mut self);
            fn pull_up(&mut self);
        }

        // pub trait Action<V: NodeSpec> {
        //     type F;

        //     // failable action, for segtree beats
        //     fn try_apply_to_sum(u: &mut V, f: &Self::F) -> bool;
        // }

        pub trait MonoidalReducer<V: NodeSpec> {
            type X: Clone;
            fn proj(u: &V) -> Self::X;
            fn id() -> Self::X;
            fn combine(x: &Self::X, y: &Self::X) -> Self::X;
        }

        pub struct WBForest<V: NodeSpec> {
            _marker: std::marker::PhantomData<V>,
        }

        fn should_rotate(size_left: u32, size_right: u32) -> bool {
            // (size_left + 1) * 3 < (size_right + 1)
            (size_left + 1) * 4 < (size_right + 1)
        }

        fn should_rotate_twice(size_left: u32, size_right: u32) -> bool {
            // (size_left + 1) * 2 < (size_right + 1)
            (size_left + 1) * 5 < (size_right + 1) * 3
        }

        fn is_balanced(size_left: u32, size_right: u32) -> bool {
            !should_rotate(size_left, size_right) && !should_rotate(size_right, size_left)
        }

        impl<V: NodeSpec> WBForest<V> {
            pub fn check_balance(u: &Rc<V>) -> bool {
                if let Some([left, right]) = u.link().children.as_ref() {
                    is_balanced(left.link().size, right.link().size)
                        && Self::check_balance(&left)
                        && Self::check_balance(&right)
                } else {
                    true
                }
            }

            unsafe fn take_children_unchecked<'a>(u: &'a mut Rc<V>) -> (&'a mut V, [Rc<V>; 2]) {
                let node = Rc::make_mut(u);
                node.push_down();
                let children = unsafe { node.link_mut().children.take().unwrap_unchecked() };
                (node, children)
            }

            fn update_children(u: &mut V, children: Option<[Rc<V>; 2]>) {
                u.link_mut().children = children;
                u.pull_up_link();
                u.pull_up();
            }

            fn new_branch(children: [Rc<V>; 2]) -> Rc<V> {
                let mut node = V::default();
                Self::update_children(&mut node, Some(children));
                Rc::new(node)
            }

            pub fn merge_nonnull(mut lhs: Rc<V>, mut rhs: Rc<V>) -> Rc<V> {
                // Unbalanced merge
                // self.new_branch([lhs, rhs])

                // Balanced merge
                unsafe {
                    if should_rotate(lhs.link().size, rhs.link().size) {
                        let (rhs_mut, [mid, rhs_right]) = Self::take_children_unchecked(&mut rhs);
                        lhs = Self::merge_nonnull(lhs, mid);
                        if is_balanced(lhs.link().size, rhs_right.link().size) {
                            Self::update_children(rhs_mut, Some([lhs, rhs_right]));
                            return rhs;
                        }
                        let (lhs_mut, [lhs_left, mut mid]) =
                            Self::take_children_unchecked(&mut lhs);
                        if !should_rotate_twice(lhs_left.link().size, mid.link().size) {
                            Self::update_children(rhs_mut, Some([mid, rhs_right]));
                            Self::update_children(lhs_mut, Some([lhs_left, rhs]));
                            return lhs;
                        }
                        let (mid_mut, [mid_left, mid_right]) =
                            Self::take_children_unchecked(&mut mid);
                        Self::update_children(lhs_mut, Some([lhs_left, mid_left]));
                        Self::update_children(rhs_mut, Some([mid_right, rhs_right]));
                        Self::update_children(mid_mut, Some([lhs, rhs]));
                        return mid;
                    } else if should_rotate(rhs.link().size, lhs.link().size) {
                        let (lhs_mut, [lhs_left, mid]) = Self::take_children_unchecked(&mut lhs);
                        rhs = Self::merge_nonnull(mid, rhs);
                        if is_balanced(lhs_left.link().size, rhs.link().size) {
                            Self::update_children(lhs_mut, Some([lhs_left, rhs]));
                            return lhs;
                        }
                        let (rhs_mut, [mut mid, rhs_right]) =
                            Self::take_children_unchecked(&mut rhs);
                        if !should_rotate_twice(rhs_right.link().size, mid.link().size) {
                            Self::update_children(lhs_mut, Some([lhs_left, mid]));
                            Self::update_children(rhs_mut, Some([lhs, rhs_right]));
                            return rhs;
                        }
                        let (mid_mut, [mid_left, mid_right]) =
                            Self::take_children_unchecked(&mut mid);
                        Self::update_children(lhs_mut, Some([lhs_left, mid_left]));
                        Self::update_children(rhs_mut, Some([mid_right, rhs_right]));
                        Self::update_children(mid_mut, Some([lhs, rhs]));
                        return mid;
                    } else {
                        Self::new_branch([lhs, rhs])
                    }
                }
            }

            pub fn merge(lhs: Option<Rc<V>>, rhs: Option<Rc<V>>) -> Option<Rc<V>> {
                match (lhs, rhs) {
                    (Some(lhs), Some(rhs)) => Some(Self::merge_nonnull(lhs, rhs)),
                    (Some(lhs), None) => Some(lhs),
                    (None, Some(rhs)) => Some(rhs),
                    (None, None) => None,
                }
            }

            fn split_inner(mut u: Rc<V>, pos: usize) -> (Rc<V>, Rc<V>) {
                debug_assert!(0 < pos && pos < u.link().size as usize);
                let (_, [left, right]) = unsafe { Self::take_children_unchecked(&mut u) };

                let left_size = left.link().size as usize;
                match pos.cmp(&left_size) {
                    Ordering::Equal => (left, right),
                    Ordering::Less => {
                        let (left, mid) = Self::split_inner(left, pos);
                        let right = Self::merge_nonnull(mid, right);
                        (left, right)
                    }
                    Ordering::Greater => {
                        let (mid, right) = Self::split_inner(right, pos - left_size);
                        let left = Self::merge_nonnull(left, mid);
                        (left, right)
                    }
                }
            }

            pub fn split_nonnull(u: Rc<V>, pos: usize) -> (Option<Rc<V>>, Option<Rc<V>>) {
                let n = u.link().size;
                debug_assert!(pos <= n as usize);
                if pos == 0 {
                    (None, Some(u))
                } else if pos == n as usize {
                    (Some(u), None)
                } else {
                    let (left, right) = Self::split_inner(u, pos);
                    (Some(left), Some(right))
                }
            }

            pub fn split(u: Option<Rc<V>>, pos: usize) -> (Option<Rc<V>>, Option<Rc<V>>) {
                u.map(|u| Self::split_nonnull(u, pos))
                    .unwrap_or((None, None))
            }

            pub fn collect_from(xs: impl ExactSizeIterator<Item = V>) -> Option<Rc<V>> {
                let n = xs.len();
                (n > 0).then(|| Self::collect_from_rec(&mut xs.into_iter(), 0..n as u32))
            }

            fn collect_from_rec(xs: &mut impl Iterator<Item = V>, range: Range<u32>) -> Rc<V> {
                let Range { start, end } = range;
                debug_assert!(start != end);
                if start + 1 == end {
                    {
                        let node = xs.next().unwrap();
                        Rc::new(node)
                    }
                } else {
                    let mid = start + end >> 1;
                    Self::new_branch([
                        Self::collect_from_rec(xs, start..mid),
                        Self::collect_from_rec(xs, mid..end),
                    ])
                }
            }

            pub fn inorder(u: &Rc<V>, visitor: &mut impl FnMut(&V)) {
                if let Some([lhs, rhs]) = u.link().children.as_ref() {
                    Self::inorder(lhs, visitor);
                    visitor(u);
                    Self::inorder(rhs, visitor);
                } else {
                    visitor(u);
                }
            }

            pub fn query_range<R: MonoidalReducer<V>>(u: &mut Rc<V>, range: Range<usize>) -> R::X {
                Self::query_range_rec::<R>(&range, 0..u.link().size as usize, u)
            }

            fn query_range_rec<R: MonoidalReducer<V>>(
                query: &Range<usize>,
                view: Range<usize>,
                u: &mut Rc<V>,
            ) -> R::X {
                unsafe {
                    if query.end <= view.start || view.end <= query.start {
                        R::id()
                    } else if query.start <= view.start && view.end <= query.end {
                        let u_mut = Rc::make_mut(u);
                        u_mut.push_down();
                        R::proj(u_mut)
                    } else {
                        let u_mut = Rc::make_mut(u);
                        u_mut.push_down();
                        let [left, right] = u_mut.link_mut().children.as_mut().unwrap_unchecked();
                        let mid = view.start + left.link().size as usize;
                        R::combine(
                            &Self::query_range_rec::<R>(query, view.start..mid, left),
                            &Self::query_range_rec::<R>(query, mid..view.end, right),
                        )
                    }
                }
            }

            pub fn with_range<T>(
                u: &mut Option<Rc<V>>,
                range: Range<usize>,
                mut f: impl FnMut(&mut Rc<V>) -> T,
            ) -> Option<T> {
                let (rest, rhs) = Self::split(u.take(), range.end);
                let (lhs, mut mid) = Self::split(rest, range.start);
                let res = mid.as_mut().map(|mid| f(mid));
                let joined = Self::merge(lhs, mid);
                *u = Self::merge(joined, rhs);
                res
            }
        }
    }
}

const INF: i64 = 1 << 58;
const NEG_INF: i64 = -INF;

#[derive(Clone, Debug)]
struct Node {
    min: i64,
    max3: [i64; 3],
    lazy_add: i64,
    link: wb::Link<Node>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            min: INF,
            max3: [NEG_INF, NEG_INF - 1, NEG_INF - 2],
            lazy_add: 0,
            link: wb::Link::default(),
        }
    }
}

impl Node {
    fn singleton(value: i64) -> Self {
        Self {
            min: value,
            max3: [value, NEG_INF, NEG_INF - 1],
            ..Default::default()
        }
    }

    fn apply_add(&mut self, delta: i64) {
        self.min += delta;
        self.max3.iter_mut().for_each(|x| *x += delta);
        self.lazy_add += delta;
    }
}

impl wb::IntrusiveNode for Node {
    fn link(&self) -> &wb::Link<Self> {
        &self.link
    }
    fn link_mut(&mut self) -> &mut wb::Link<Self> {
        &mut self.link
    }
}

impl wb::NodeSpec for Node {
    fn push_down(&mut self) {
        if self.lazy_add != 0 {
            if let Some(cs) = self.link.children.as_mut() {
                for c in cs {
                    c.make_mut().apply_add(self.lazy_add);
                }
            }
            self.lazy_add = 0;
        }
    }

    fn pull_up(&mut self) {
        let Some([left, right]) = self.link.children.as_mut() else {
            return;
        };

        self.min = left.min.min(right.min);
        let mut lhs = left.max3.iter().copied().peekable();
        let mut rhs = right.max3.iter().copied().peekable();
        for i in 0..3 {
            match lhs.peek().unwrap().cmp(rhs.peek().unwrap()) {
                Ordering::Less => self.max3[i] = rhs.next().unwrap(),
                Ordering::Greater => self.max3[i] = lhs.next().unwrap(),
                Ordering::Equal => {
                    self.max3[i] = lhs.next().unwrap();
                    rhs.next();
                }
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs = (0..n).map(|_| input.value::<i64>());

    type Forest = wb::WBForest<Node>;
    let mut root = Forest::collect_from(xs.map(Node::singleton));
    for _ in 0..input.value() {
        match input.token() {
            "1" => {
                let i = input.value::<usize>() - 1;
                let (rest, rhs) = Forest::split(root, i + 1);
                let (lhs, _) = Forest::split(rest, i);
                root = Forest::merge(lhs, rhs);
            }
            "2" => {
                let i = input.value::<usize>() - 1;
                let r = input.value::<usize>();
                let (s, e) = (i - r, i + r);
                Forest::with_range(&mut root, s..e + 1, |node| {
                    let node = node.make_mut();
                    node.apply_add(-node.min);
                });
            }
            "3" => {
                let i = input.value::<usize>() - 1;
                let r = input.value::<usize>();
                let (s, e) = (i - r, i + r);
                Forest::with_range(&mut root, s..e + 1, |node| {
                    let node = node.make_mut();
                    node.apply_add(node.max3[0]);
                });
            }
            "4" => {
                let l = input.value::<usize>() - 1;
                let r = input.value::<usize>() - 1;
                let mut ans = Forest::with_range(&mut root, l..r + 1, |node| {
                    let node = node.make_mut();
                    node.max3[2]
                })
                .unwrap();
                if ans <= NEG_INF / 2 {
                    ans = -1;
                }
                writeln!(output, "{}", ans).unwrap();
            }

            _ => panic!(),
        }
    }
}
