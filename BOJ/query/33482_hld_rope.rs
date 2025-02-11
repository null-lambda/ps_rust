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

pub mod hld {
    // Heavy-Light Decomposition
    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        #[cold]
        #[inline(always)]
        pub fn cold() {}

        if !b {
            cold();
        }
        b
    }

    const UNSET: u32 = u32::MAX;

    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub chain_bot: Vec<u32>,
        pub segmented_idx: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        pub fn from_edges<'a>(
            n: usize,
            edges: impl IntoIterator<Item = (u32, u32)>,
            root: usize,
            use_dfs_ordering: bool,
        ) -> Self {
            // Fast tree reconstruction with XOR-linked tree traversal
            // https://codeforces.com/blog/entry/135239
            let mut degree = vec![0u32; n];
            let mut xor_neighbors: Vec<u32> = vec![0u32; n];
            for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
                debug_assert!(u != v);
                degree[u as usize] += 1;
                xor_neighbors[u as usize] ^= v;
            }

            let mut size = vec![1; n];
            let mut heavy_child = vec![UNSET; n];
            let mut chain_bot = vec![UNSET; n];
            degree[root] += 2;
            let mut topological_order = Vec::with_capacity(n);
            for mut u in 0..n {
                while degree[u] == 1 {
                    // Topological sort
                    let p = xor_neighbors[u];
                    topological_order.push(u as u32);
                    degree[u] = 0;
                    degree[p as usize] -= 1;
                    xor_neighbors[p as usize] ^= u as u32;

                    // Upward propagation
                    size[p as usize] += size[u as usize];
                    let h = &mut heavy_child[p as usize];
                    if *h == UNSET || size[*h as usize] < size[u as usize] {
                        *h = u as u32;
                    }

                    let h = heavy_child[u as usize];
                    chain_bot[u] = if h == UNSET {
                        u as u32
                    } else {
                        chain_bot[h as usize]
                    };

                    assert!(u != p as usize);
                    u = p as usize;
                }
            }
            topological_order.push(root as u32);
            assert!(topological_order.len() == n, "Invalid tree structure");

            let h = heavy_child[root];
            chain_bot[root] = if h == UNSET {
                root as u32
            } else {
                chain_bot[h as usize]
            };

            let mut parent = xor_neighbors;
            parent[root] = UNSET;

            // Downward propagation
            let mut chain_top = vec![root as u32; n];
            let mut segmented_idx = vec![UNSET; n];
            if !use_dfs_ordering {
                // A rearranged topological index continuous in a chain, for path queries
                let mut timer = 0;
                for mut u in topological_order.into_iter().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }
                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        segmented_idx[u as usize] = timer;
                        timer += 1;
                        u = heavy_child[u as usize];
                        if u == UNSET {
                            break;
                        }
                    }
                }
            } else {
                // DFS ordering for path & subtree queries
                let mut offset = vec![0; n];
                for mut u in topological_order.into_iter().rev() {
                    if segmented_idx[u as usize] != UNSET {
                        continue;
                    }

                    let mut p = parent[u as usize];
                    let mut timer = 0;
                    if likely(p != UNSET) {
                        timer = offset[p as usize] + 1;
                        offset[p as usize] += size[u as usize] as u32;
                    }

                    let u0 = u;
                    loop {
                        chain_top[u as usize] = u0;
                        offset[u as usize] = timer;
                        segmented_idx[u as usize] = timer;
                        timer += 1;

                        p = u as u32;
                        u = heavy_child[p as usize];
                        unsafe { assert_unchecked(u != p) };
                        if u == UNSET {
                            break;
                        }
                        offset[p as usize] += size[u as usize] as u32;
                    }
                }
            }

            Self {
                size,
                parent,
                heavy_child,
                chain_top,
                chain_bot,
                segmented_idx,
            }
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    > self.segmented_idx[self.chain_top[v] as usize]
                {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.segmented_idx[self.chain_top[u] as usize]
                    < self.segmented_idx[self.chain_top[v] as usize]
                {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.segmented_idx[u] > self.segmented_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
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
                // Self::new_branch([lhs, rhs])

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
        }
    }
}

use wbtree::persistent::Rc;

#[derive(Clone, Default, Debug)]
struct XorNode {
    sum: u32,
    inv_lazy: bool,
    link: wbtree::persistent::Link<Self>,
}

impl XorNode {
    fn new(value: u32) -> Self {
        Self {
            sum: value,
            inv_lazy: false,
            link: Default::default(),
        }
    }

    fn size(&self) -> u32 {
        self.link.size
    }

    fn reverse(&mut self) {
        self.inv_lazy ^= true;
    }
}

impl wbtree::persistent::IntrusiveNode for XorNode {
    fn link(&self) -> &wbtree::persistent::Link<Self> {
        &self.link
    }
    fn link_mut(&mut self) -> &mut wbtree::persistent::Link<Self> {
        &mut self.link
    }
}

impl wbtree::persistent::NodeSpec for XorNode {
    fn pull_up(&mut self) {
        self.push_down();
        let Some([left, right]) = self.link.children.as_mut() else {
            return;
        };
        let [left, right] = [Rc::make_mut(left), Rc::make_mut(right)];
        left.push_down();
        right.push_down();

        self.link.size = left.link.size + right.link.size;
        self.sum = left.sum ^ right.sum;
    }

    fn push_down(&mut self) {
        let Some(children) = self.link.children.as_mut() else {
            return;
        };
        if self.inv_lazy {
            self.inv_lazy = false;
            children.swap(0, 1);

            for c in children {
                if c.link.children.is_some() {
                    Rc::make_mut(c).inv_lazy ^= true;
                }
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let edges = (0..n - 1).map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1));
    let hld = hld::HLD::from_edges(n, edges, 0, true);
    let sid = |u: usize| hld.segmented_idx[u] as usize;
    let mut sid_inv = vec![!0; n];
    for u in 0..n {
        sid_inv[sid(u)] = u as u32;
    }
    let chain_range = |u: usize| sid(hld.chain_top[u] as usize)..sid(hld.chain_bot[u] as usize) + 1;

    type Forest = wbtree::persistent::WBForest<XorNode>;

    let mut chains = vec![None; n];
    for u in 0..n {
        if u != hld.chain_top[u] as usize {
            continue;
        }
        let weights = chain_range(u).map(|s| sid_inv[s] + 1);
        chains[u] = Forest::collect_from(weights.map(|u| XorNode::new(u)));
    }

    for _ in 0..q {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let shift: usize = input.value();

        let mut left_acc = None;
        let mut right_acc = None;
        let mut left_path = vec![];
        let mut right_path = vec![];
        hld.for_each_path_splitted(u, v, |u, v, is_left, _is_u_lca| {
            let top = hld.chain_top[u as usize] as usize;
            let range = sid(u) - sid(top)..sid(v) + 1 - sid(top);

            let (rest, rhs) = Forest::split(chains[top].take(), range.end);
            let (lhs, mid) = Forest::split(rest, range.start);

            let mid_size = mid.as_ref().map(|mid| mid.size()).unwrap_or(0);
            if is_left {
                left_path.push((top, lhs, mid_size, rhs));
                left_acc = Forest::merge(mid, left_acc.take());
            } else {
                right_path.push((top, lhs, mid_size, rhs));
                right_acc = Forest::merge(mid, right_acc.take());
            }
        });

        left_acc.as_mut().map(|root| Rc::make_mut(root).reverse());
        let left_size = left_acc.as_ref().map(|root| root.size()).unwrap_or(0);
        let mut acc = Forest::merge(left_acc, right_acc);

        let size = acc.as_ref().map(|acc| acc.size()).unwrap_or(0);
        if size > 0 {
            let shift = shift % size as usize;
            let (lhs, rhs) = Forest::split(acc, size as usize - shift);
            acc = Forest::merge(rhs, lhs);
        }

        let ans = acc.as_ref().map(|acc| acc.sum).unwrap_or(0);
        writeln!(output, "{}", ans).unwrap();

        let (mut left_acc, mut right_acc) = Forest::split(acc, left_size as usize);
        left_acc.as_mut().map(|root| Rc::make_mut(root).reverse());

        for (top, lhs, mid_size, rhs) in left_path.into_iter().rev() {
            let (mut recovered, rest) = Forest::split(left_acc, mid_size as usize);
            recovered = Forest::merge(lhs, recovered);
            recovered = Forest::merge(recovered, rhs);
            chains[top] = recovered;
            left_acc = rest;
        }

        for (top, lhs, mid_size, rhs) in right_path.into_iter().rev() {
            let (mut recovered, rest) = Forest::split(right_acc, mid_size as usize);
            recovered = Forest::merge(lhs, recovered);
            recovered = Forest::merge(recovered, rhs);
            chains[top] = recovered;
            right_acc = rest;
        }
    }
}
