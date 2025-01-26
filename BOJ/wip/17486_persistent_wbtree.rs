use std::{io::Write, ops::Range};

use wbtree::{MonoidalReducer, NodeRef, NodeSpec, WBForest};

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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

pub mod wbtree {
    // Weight-balanced tree
    // https://koosaga.com/342
    // https://yoichihirai.com/bst.pdf

    use std::{cmp::Ordering, mem::MaybeUninit, num::NonZeroU32, ops::Range};

    const PERSISTENT: bool = true;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct NodeRef(NonZeroU32);

    impl NodeRef {
        pub fn get(self) -> usize {
            self.0.get() as usize
        }
    }

    #[derive(Default, Debug, Clone)]
    pub struct SizedNode<V> {
        pub size: u32,
        pub children: Option<[NodeRef; 2]>,
        pub inner: V,
    }

    pub trait NodeSpec: Sized + Default + Clone {
        fn push_down(forest: &mut WBForest<Self>, u: NodeRef);
        fn pull_up(forest: &mut WBForest<Self>, u: NodeRef);
    }

    pub trait ReversibleNodeSpec: NodeSpec {
        fn is_inv(&self) -> bool;
        fn reverse(&mut self);
    }

    pub trait MonoidalReducer<V: NodeSpec> {
        type X: Clone;
        fn proj(forest: &WBForest<V>, u: NodeRef) -> Self::X;
        fn id() -> Self::X;
        fn combine(x: &Self::X, y: &Self::X) -> Self::X;
    }

    pub struct WBForest<V: NodeSpec> {
        pool: Vec<MaybeUninit<SizedNode<V>>>,
        free: Vec<NodeRef>,
    }

    impl<V: NodeSpec> WBForest<V> {
        pub fn new() -> Self {
            let dummy = MaybeUninit::uninit();
            Self {
                pool: vec![dummy],
                free: vec![],
            }
        }

        fn alloc(&mut self, node: SizedNode<V>) -> NodeRef {
            if let Some(u) = self.free.pop() {
                self.pool[u.get()] = MaybeUninit::new(node);
                return u;
            }
            let idx = self.pool.len() as u32;
            self.pool.push(MaybeUninit::new(node));
            unsafe { NodeRef(NonZeroU32::new_unchecked(idx)) }
        }

        pub fn clone(&mut self, u: NodeRef) -> NodeRef {
            if PERSISTENT {
                let node = self.get(u).clone();
                self.alloc(node)
            } else {
                u
            }
        }

        pub unsafe fn free_unchecked(&mut self, u: NodeRef) {
            // WIP
            if !PERSISTENT {
                self.free.push(u);
            }
        }

        pub fn new_leaf(&mut self, value: V) -> NodeRef {
            self.alloc(SizedNode {
                size: 1,
                children: None,
                inner: value,
            })
        }

        pub fn new_branch(&mut self, children: [NodeRef; 2]) -> NodeRef {
            let node = self.alloc(SizedNode {
                size: 0,             // Uninit
                inner: V::default(), // Uninit
                children: Some(children),
            });
            V::pull_up(self, node);
            node
        }

        pub fn get(&self, u: NodeRef) -> &SizedNode<V> {
            unsafe { self.pool[u.get()].assume_init_ref() }
        }

        // Mutable references are inherently unsafe, since NodeRef may alias easily.
        pub unsafe fn get_mut(&mut self, u: NodeRef) -> &mut SizedNode<V> {
            unsafe { self.pool[u.get()].assume_init_mut() }
        }

        pub unsafe fn get_many<const N: usize>(
            &mut self,
            us: [NodeRef; N],
        ) -> Option<[&mut SizedNode<V>; N]> {
            if cfg!(debug_assertions) {
                // Check for multiple aliases
                for i in 0..N {
                    for j in i + 1..N {
                        if us[i] == us[j] {
                            return None;
                        }
                    }
                }
            }

            let ptr = self.pool.as_mut_ptr();
            Some(us.map(|u| unsafe { (&mut *ptr.add(u.get())).assume_init_mut() }))
        }

        pub fn size(&self, u: NodeRef) -> u32 {
            self.get(u).size
        }
    }

    impl<S: NodeSpec> Drop for WBForest<S> {
        fn drop(&mut self) {
            for u in self.free.drain(..) {
                self.pool[u.get()] = MaybeUninit::new(Default::default());
            }
            for i in (1..self.pool.len()).rev() {
                unsafe {
                    self.pool[i].assume_init_drop();
                }
            }
        }
    }

    fn should_rotate(size_left: u32, size_right: u32) -> bool {
        (size_left + 1) * 3 < (size_right + 1)
    }

    fn should_rotate_once(size_left: u32, size_right: u32) -> bool {
        (size_left + 1) < (size_right + 1) * 2
    }

    fn is_balanced(size_left: u32, size_right: u32) -> bool {
        !should_rotate(size_left, size_right) && !should_rotate(size_right, size_left)
    }

    impl<V: NodeSpec> WBForest<V> {
        pub fn check_balance(&mut self, u: NodeRef) -> bool {
            if let Some([left, right]) = self.get(u).children {
                is_balanced(self.size(left), self.size(right))
                    && self.check_balance(left)
                    && self.check_balance(right)
            } else {
                true
            }
        }

        pub fn merge(&mut self, mut lhs: NodeRef, mut rhs: NodeRef) -> NodeRef {
            // Unbalanced merge, should be slower for strong dataset
            // self.new_branch([lhs, rhs])

            // Balanced merge
            unsafe {
                let take_children = |forest: &mut Self, u: &mut NodeRef| {
                    let children = forest
                        .get(*u)
                        .children
                        .map(|cs| cs.map(|c| forest.clone(c)));
                    *u = forest.clone(*u);
                    forest.get_mut(*u).children = children;
                    V::push_down(forest, *u);
                    children
                };
                let update_children = |forest: &mut Self, u: NodeRef, children| {
                    forest.get_mut(u).children = children;
                    V::pull_up(forest, u);
                };

                if should_rotate(self.size(lhs), self.size(rhs)) {
                    let [mid, rhs_right] = take_children(self, &mut rhs).unwrap_unchecked();
                    lhs = self.merge(lhs, mid);
                    if is_balanced(self.size(lhs), self.size(rhs_right)) {
                        update_children(self, rhs, Some([lhs, rhs_right]));
                        return rhs;
                    }
                    let [lhs_left, mut mid] = take_children(self, &mut lhs).unwrap_unchecked();
                    if should_rotate_once(self.size(mid), self.size(lhs_left)) {
                        update_children(self, rhs, Some([mid, rhs_right]));
                        update_children(self, lhs, Some([lhs_left, rhs]));
                        return lhs;
                    }
                    let [mid_left, mid_right] = take_children(self, &mut mid).unwrap_unchecked();
                    update_children(self, lhs, Some([lhs_left, mid_left]));
                    update_children(self, rhs, Some([mid_right, rhs_right]));
                    update_children(self, mid, Some([lhs, rhs]));
                    return mid;
                } else if should_rotate(self.size(rhs), self.size(lhs)) {
                    let [lhs_left, mid] = take_children(self, &mut lhs).unwrap_unchecked();
                    rhs = self.merge(mid, rhs);
                    if is_balanced(self.size(lhs_left), self.size(rhs)) {
                        update_children(self, lhs, Some([lhs_left, rhs]));
                        return lhs;
                    }
                    let [mut mid, rhs_right] = take_children(self, &mut rhs).unwrap_unchecked();
                    if should_rotate_once(self.size(mid), self.size(rhs_right)) {
                        update_children(self, lhs, Some([lhs_left, mid]));
                        update_children(self, rhs, Some([lhs, rhs_right]));
                        return rhs;
                    }
                    let [mid_left, mid_right] = take_children(self, &mut mid).unwrap_unchecked();
                    update_children(self, lhs, Some([lhs_left, mid_left]));
                    update_children(self, rhs, Some([mid_right, rhs_right]));
                    update_children(self, mid, Some([lhs, rhs]));
                    return mid;
                } else {
                    self.new_branch([lhs, rhs])
                }
            }
        }

        fn split_inner(&mut self, mut u: NodeRef, pos: usize) -> (NodeRef, NodeRef) {
            debug_assert!(0 < pos && pos < self.size(u) as usize);
            unsafe {
                let mut cs = self.get(u).children.unwrap_unchecked(); // size >= 2, so it must be a branch
                cs = cs.map(|c| self.clone(c));
                u = self.clone(u);
                self.get_mut(u).children = Some(cs);
                V::push_down(self, u);

                let [left, right] = cs;

                self.free_unchecked(u);

                let left_size = self.size(left) as usize;
                match pos.cmp(&left_size) {
                    Ordering::Equal => (left, right),
                    Ordering::Less => {
                        let (left, mid) = self.split_inner(left, pos);
                        let right = self.merge(mid, right);
                        (left, right)
                    }
                    Ordering::Greater => {
                        let (mid, right) = self.split_inner(right, pos - left_size);
                        let left = self.merge(left, mid);
                        (left, right)
                    }
                }
            }
        }

        pub fn split(&mut self, u: NodeRef, pos: usize) -> (Option<NodeRef>, Option<NodeRef>) {
            let n = self.size(u);
            debug_assert!(pos <= n as usize);
            if pos == 0 {
                (None, Some(u))
            } else if pos == n as usize {
                (Some(u), None)
            } else {
                let (left, right) = self.split_inner(u, pos);
                (Some(left), Some(right))
            }
        }

        pub fn collect_from(&mut self, xs: impl ExactSizeIterator<Item = V>) -> Option<NodeRef> {
            let n = xs.len();
            (n > 0).then(|| self.collect_from_rec(&mut xs.into_iter(), 0..n as u32))
        }

        fn collect_from_rec(
            &mut self,
            xs: &mut impl Iterator<Item = V>,
            range: Range<u32>,
        ) -> NodeRef {
            let Range { start, end } = range;
            debug_assert!(start != end);
            if start + 1 == end {
                self.new_leaf(xs.next().unwrap())
            } else {
                let mid = start + end >> 1;
                let left = self.collect_from_rec(xs, start..mid);
                let right = self.collect_from_rec(xs, mid..end);
                self.new_branch([left, right])
            }
        }

        pub fn query_range<R: MonoidalReducer<V>>(
            &mut self,
            u: &mut NodeRef,
            range: Range<usize>,
        ) -> R::X {
            // *u = self.clone(*u);
            self.query_range_rec::<R>(&range, 0..self.size(*u) as usize, *u)
        }

        fn query_range_rec<R: MonoidalReducer<V>>(
            &mut self,
            query: &Range<usize>,
            view: Range<usize>,
            u: NodeRef,
        ) -> R::X {
            unsafe {
                if query.end <= view.start || view.end <= query.start {
                    return R::id();
                }

                // if let Some(mut cs) = self.get(u).children {
                //     cs = cs.map(|c| self.clone(c));
                //     self.get_mut(u).children = Some(cs);
                // }
                V::push_down(self, u);

                if query.start <= view.start && view.end <= query.end {
                    R::proj(self, u)
                } else {
                    let [left, right] = self.get(u).children.unwrap_unchecked();
                    let mid = view.start + self.size(left) as usize;
                    R::combine(
                        &self.query_range_rec::<R>(query, view.start..mid, left),
                        &self.query_range_rec::<R>(query, mid..view.end, right),
                    )
                }
            }
        }
    }

    impl<V: ReversibleNodeSpec> WBForest<V> {
        pub fn reverse(&mut self, u: &mut NodeRef) {
            unsafe {
                V::push_down(self, *u);
                self.get_mut(*u).inner.reverse();
            }
        }

        pub fn reverse_range(&mut self, u: &mut NodeRef, range: Range<usize>) {
            unsafe {
                let (left, Some(mut rest)) = self.split(*u, range.start) else {
                    return;
                };
                let (Some(mid), right) = self.split(rest, range.end - range.start) else {
                    return;
                };
                self.get_mut(mid).inner.reverse();
                rest = right.map(|right| self.merge(mid, right)).unwrap_or(mid);
                *u = left.map(|left| self.merge(left, rest)).unwrap_or(rest);
            }
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct AdditiveNode {
    sum: i64,
    lazy: i64,
}

impl AdditiveNode {
    fn new(value: i64) -> Self {
        Self {
            sum: value,
            lazy: 0,
        }
    }

    fn apply(&mut self, delta: i64, size: u32) {
        self.sum += delta * size as i64;
        self.lazy += delta;
    }
}

impl NodeSpec for AdditiveNode {
    fn pull_up(forest: &mut WBForest<Self>, u: NodeRef) {
        unsafe {
            let [left, right] = forest.get(u).children.unwrap_unchecked();
            let [u, left, right] = forest.get_many([u, left, right]).unwrap_unchecked();
            u.size = left.size + right.size;
            u.inner.sum = left.inner.sum + right.inner.sum;
        }
    }

    fn push_down(forest: &mut WBForest<Self>, u: NodeRef) {
        unsafe {
            let Some([left, right]) = forest.get(u).children else {
                return;
            };
            let [u, left, right] = forest.get_many([u, left, right]).unwrap_unchecked();
            left.inner.apply(u.inner.lazy, left.size);
            right.inner.apply(u.inner.lazy, right.size);
            u.inner.lazy = 0;
        }
    }
}

fn apply_range(
    forest: &mut WBForest<AdditiveNode>,
    u: &mut NodeRef,
    range: Range<usize>,
    delta: i64,
) {
    fn apply_range_rec(
        forest: &mut WBForest<AdditiveNode>,
        query: &Range<usize>,
        view: Range<usize>,
        u: NodeRef,
        delta: i64,
    ) {
        if query.end <= view.start || view.end <= query.start {
            return;
        }

        if let Some(cs) = forest.get(u).children {
            unsafe { forest.get_mut(u).children = Some(cs.map(|c| forest.clone(c))) };
        }

        if query.start <= view.start && view.end <= query.end {
            let size = forest.size(u);
            unsafe { AdditiveNode::apply(&mut forest.get_mut(u).inner, delta, size) };
            AdditiveNode::push_down(forest, u);
            return;
        }
        let [left, right] = forest.get(u).children.unwrap();
        let mid = view.start + forest.size(left) as usize;
        apply_range_rec(forest, query, view.start..mid, left, delta);
        apply_range_rec(forest, query, mid..view.end, right, delta);
        AdditiveNode::pull_up(forest, u);
    }

    *u = forest.clone(*u);
    apply_range_rec(forest, &range, 0..forest.size(*u) as usize, *u, delta);
}

impl MonoidalReducer<AdditiveNode> for AdditiveNode {
    type X = i64;

    fn proj(forest: &WBForest<AdditiveNode>, u: NodeRef) -> Self::X {
        forest.get(u).inner.sum
    }
    fn id() -> Self::X {
        0
    }
    fn combine(x: &Self::X, y: &Self::X) -> Self::X {
        x + y
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let xs = (0..n).map(|_| input.value());
    let mut forest = WBForest::new();
    let mut root = forest.collect_from(xs.map(AdditiveNode::new)).unwrap();

    let q: usize = input.value();
    for _ in 0..q {
        let cmd = input.token();
        let l = input.value::<usize>() - 1;
        let r = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                let delta: i64 = input.value();
                apply_range(&mut forest, &mut root, l..r + 1, delta);
            }
            "2" => {
                let s = input.value::<usize>() - 1;
                let e = input.value::<usize>() - 1;

                let (rest, _) = forest.split(root, e + 1);
                let (_, mid) = forest.split(rest.unwrap(), s);
                let src = mid.unwrap();

                let (rest, right) = forest.split(root, r + 1);
                let (left, _) = forest.split(rest.unwrap(), l);

                root = src;
                if let Some(left) = left {
                    root = forest.merge(left, root);
                }
                if let Some(right) = right {
                    root = forest.merge(root, right);
                }
            }
            "3" => {
                {
                    // for i in 0..n {
                    //     eprint!(
                    //         "{} ",
                    //         forest.query_range::<AdditiveNode>(&mut root, i..i + 1)
                    //     );
                    // }
                    // eprintln!();
                }

                let ans = forest.query_range::<AdditiveNode>(&mut root, l..r + 1);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }

    // let mut ans = vec![];
    // inorder(&mut forest, root, &mut |forest, u| {
    //     ans.push(forest.get(u).inner.value);
    // });
    // let len_trunc = ans.iter().rposition(|&x| x != UNSET).map_or(0, |x| x + 1);
    // writeln!(output, "{}", len_trunc).unwrap();
    // for a in &ans[..len_trunc] {
    //     write!(output, "{} ", a).unwrap();
    // }
    // writeln!(output).unwrap();
}
