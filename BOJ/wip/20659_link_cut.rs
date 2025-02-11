use std::{cmp::Ordering, io::Write};

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
    use std::{fmt::Debug, rc::Rc};

    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }

    #[cfg(debug_assertions)]
    #[derive(Clone)]
    pub struct Label(Rc<dyn Debug>);

    #[cfg(not(debug_assertions))]
    #[derive(Clone)]
    pub struct Label;

    impl Label {
        #[inline]
        pub fn new_with<T: Debug + 'static>(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(Rc::new(value()))
            }
            #[cfg(not(debug_assertions))]
            {
                Self
            }
        }
    }

    impl Debug for Label {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            #[cfg(debug_assertions)]
            {
                write!(f, "{:?}", self.0)
            }
            #[cfg(not(debug_assertions))]
            {
                write!(f, "()")
            }
        }
    }

    impl Default for Label {
        fn default() -> Self {
            Self::new_with(|| ())
        }
    }
}

pub mod segtree_lazy {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        const IS_X_COMMUTATIVE: bool = false; // TODO
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &mut Self::X);
    }

    pub struct SegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        sum: Vec<M::X>,
        lazy: Vec<M::F>,
        ma: M,
    }

    impl<M: MonoidAction> SegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        pub fn from_iter<I>(iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
                        .take(n),
                )
                .collect();
            for i in (1..n).rev() {
                sum[i] = ma.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        fn apply(&mut self, idx: usize, width: u32, value: &M::F) {
            self.ma.apply_to_sum(&value, width, &mut self.sum[idx]);
            if idx < self.n {
                self.lazy[idx] = self.ma.combine_action(&value, &self.lazy[idx]);
            }
        }

        fn push_down(&mut self, width: u32, node: usize) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(width, start >> height);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(width, end - 1 >> height);
            }
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
        }

        pub fn apply_range(&mut self, range: Range<usize>, value: M::F) {
            let Range { mut start, mut end } = range;
            debug_assert!(start <= end && end <= self.n);
            if start == end {
                return;
            }

            self.push_range(range);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut pull_start, mut pull_end) = (false, false);
            while start < end {
                if pull_start {
                    self.pull_up(start - 1);
                }
                if pull_end {
                    self.pull_up(end);
                }
                if start & 1 != 0 {
                    self.apply(start, width, &value);
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, &value);
                    pull_end = true;
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if pull_start {
                    self.pull_up(start);
                }
                if pull_end && !(pull_start && start == end) {
                    self.pull_up(end);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;

            self.push_range(range);
            start += self.n;
            end += self.n;
            if M::IS_X_COMMUTATIVE {
                let mut result = self.ma.id();
                while start < end {
                    if start & 1 != 0 {
                        result = self.ma.combine(&result, &self.sum[start]);
                        start += 1;
                    }
                    if end & 1 != 0 {
                        end -= 1;
                        result = self.ma.combine(&result, &self.sum[end]);
                    }
                    start >>= 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (self.ma.id(), self.ma.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = self.ma.combine(&result_left, &self.sum[start]);
                    }
                    if end & 1 != 0 {
                        result_right = self.ma.combine(&self.sum[end - 1], &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                self.ma.combine(&result_left, &result_right)
            }
        }

        pub fn query_all(&mut self) -> &M::X {
            assert!(self.n.is_power_of_two());
            self.push_down(self.n as u32, 1);
            &self.sum[1]
        }

        // The following two lines are equivalent.
        // partition_point(0, n, |i| pred(segtree.query_range(0..i+1)));
        // segtree.partition_point_prefix(|prefix| pred(prefix));
        pub fn partition_point_prefix(&mut self, mut pred: impl FnMut(&M::X) -> bool) -> usize {
            assert!(self.n >= 1 && self.n.is_power_of_two());

            let mut u = 1;
            let mut width = self.n as u32;
            let mut prefix = self.ma.id();

            while u < self.n {
                width >>= 1;
                self.push_down(width, u);

                let new_prefix = self.ma.combine(&prefix, &self.sum[u << 1]);
                u = if pred(&new_prefix) {
                    prefix = new_prefix;
                    u << 1 | 1
                } else {
                    u << 1
                };
            }

            let idx = u - self.n;
            if pred(&self.ma.combine(&prefix, &self.sum[u])) {
                idx + 1
            } else {
                idx
            }
        }
    }
}

pub mod splay {
    // Reversible rope, based on a splay tree.
    use std::{
        fmt::{self, Debug},
        mem::MaybeUninit,
        num::NonZeroU32,
    };

    // Adjoin an identity element to a binary operation.
    fn lift_binary<A>(
        combine: impl FnOnce(A, A) -> A,
    ) -> impl FnOnce(Option<A>, Option<A>) -> Option<A> {
        |lhs, rhs| match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => Some(combine(lhs, rhs)),
            (None, rhs) => rhs,
            (lhs, None) => lhs,
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        Left = 0,
        Right = 1,
    }

    impl Branch {
        pub fn usize(self) -> usize {
            self as usize
        }

        pub fn inv(&self) -> Self {
            match self {
                Branch::Left => Branch::Right,
                Branch::Right => Branch::Left,
            }
        }
    }

    // Intrusive node link, invertible.
    #[derive(Default, Debug)]
    pub struct Link {
        children: [Option<NodeRef>; 2],
        parent: Option<NodeRef>,
    }

    pub trait IntrusiveNode {
        fn link(&self) -> &Link;
        fn link_mut(&mut self) -> &mut Link;
    }

    pub trait NodeSpec: IntrusiveNode {
        fn push_down(&mut self, _children: [Option<&mut Self>; 2]) {}
        fn pull_up(&mut self, _children: [Option<&mut Self>; 2]) {}
        fn on_reverse(&mut self) {}

        // type Cx;
        // fn push_down(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
        // fn pull_up(&mut self, cx: &mut Self::Cx, children: [Option<&mut Self>; 2]);
    }

    pub trait SizedNode: NodeSpec {
        fn size(&self) -> usize;
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeRef {
        pub idx: NonZeroU32,
    }

    impl NodeRef {
        fn get(&self) -> usize {
            self.idx.get() as usize
        }
    }

    impl Debug for NodeRef {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct SplayForest<V> {
        pub pool: Vec<MaybeUninit<V>>,
    }

    impl<V: NodeSpec> SplayForest<V> {
        pub fn new() -> Self {
            let dummy = MaybeUninit::uninit();
            Self { pool: vec![dummy] }
        }

        pub fn add_root(&mut self, node: V) -> NodeRef {
            let idx = self.pool.len();
            self.pool.push(MaybeUninit::new(node));
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
            }
        }

        pub fn get<'a>(&'a self, u: NodeRef) -> &'a V {
            unsafe { &self.pool[u.get()].assume_init_ref() }
        }

        pub unsafe fn get_mut<'a>(&'a mut self, u: NodeRef) -> &'a mut V {
            self.pool[u.get()].assume_init_mut()
        }

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: NodeRef,
        ) -> (&'a mut V, [Option<&'a mut V>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = (&mut *pool_ptr.add(u.get())).assume_init_mut();
                let children = node.link().children.map(|child| {
                    child.map(|child| (&mut *pool_ptr.add(child.get())).assume_init_mut())
                });
                (node, children)
            }
        }

        pub fn with<T>(&mut self, u: NodeRef, f: impl FnOnce(&mut V) -> T) -> T {
            f(unsafe { self.get_mut(u) })
        }

        fn push_down(&mut self, u: NodeRef) {
            unsafe {
                let (node, children) = self.get_with_children(u);
                node.push_down(children);
            }
        }

        pub fn pull_up(&mut self, node: NodeRef) {
            unsafe {
                let (node, children) = self.get_with_children(node);
                node.pull_up(children);
            }
        }

        pub fn get_parent(&self, u: NodeRef) -> Option<(NodeRef, Branch)> {
            let p = self.get(u).link().parent?;
            if self.get(p).link().children[Branch::Left.usize()] == Some(u) {
                Some((p, Branch::Left))
            } else if self.get(p).link().children[Branch::Right.usize()] == Some(u) {
                Some((p, Branch::Right))
            } else {
                None
            }
        }

        pub fn is_root(&self, u: NodeRef) -> bool {
            self.get_parent(u).is_none()
        }

        fn attach(&mut self, u: NodeRef, child: NodeRef, branch: Branch) {
            debug_assert_ne!(u, child);
            unsafe {
                self.get_mut(u).link_mut().children[branch as usize] = Some(child);
                self.get_mut(child).link_mut().parent = Some(u);
            }
        }

        fn detach(&mut self, u: NodeRef, branch: Branch) -> Option<NodeRef> {
            unsafe {
                let child = self.get_mut(u).link_mut().children[branch as usize].take()?;
                self.get_mut(child).link_mut().parent = None;
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
                Some((grandparent, grandbranch)) => {
                    self.attach(grandparent, u, grandbranch);
                }
                None => unsafe {
                    self.get_mut(u).link_mut().parent = None;
                },
            }
            self.attach(u, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(u);
        }

        pub fn splay(&mut self, u: NodeRef) {
            while let Some((parent, branch)) = self.get_parent(u) {
                if let Some((grandparent, grandbranch)) = self.get_parent(parent) {
                    self.push_down(grandparent);
                    self.push_down(parent);
                    self.push_down(u);
                    if branch != grandbranch {
                        self.rotate(u);
                    } else {
                        self.rotate(parent);
                    }
                } else {
                    self.push_down(parent);
                    self.push_down(u);
                }
                self.rotate(u);
            }
            self.push_down(u);
        }

        // Caution: breaks amortized time complexity if not splayed afterwards.
        pub unsafe fn find_by(
            &mut self,
            mut u: NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) -> NodeRef {
            loop {
                self.push_down(u);
                if let Some(next) =
                    next(self, u).and_then(|branch| self.get(u).link().children[branch.usize()])
                {
                    u = next;
                } else {
                    break;
                }
            }
            u
        }

        // Caution: if u is not a root, then only the subtree nodes can be accessed.
        // Call splay(u) beforehand to walk on the full tree.
        pub fn splay_by(
            &mut self,
            u: &mut NodeRef,
            mut next: impl FnMut(&Self, NodeRef) -> Option<Branch>,
        ) {
            *u = unsafe { self.find_by(*u, &mut next) };
            self.splay(*u);
        }

        pub fn splay_first(&mut self, u: &mut NodeRef) {
            self.splay_by(u, |_, _| Some(Branch::Left))
        }

        pub fn splay_last(&mut self, u: &mut NodeRef) {
            self.splay_by(u, |_, _| Some(Branch::Right))
        }

        pub fn inorder(&mut self, u: NodeRef, visitor: &mut impl FnMut(&mut Self, NodeRef)) {
            self.push_down(u);
            if let Some(left) = self.get(u).link().children[Branch::Left.usize()] {
                self.inorder(left, visitor);
            }
            visitor(self, u);
            if let Some(right) = self.get(u).link().children[Branch::Right.usize()] {
                self.inorder(right, visitor);
            }
        }

        pub fn split_left(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let left = self.detach(u, Branch::Left)?;
            self.pull_up(u);
            Some(left)
        }

        pub fn split_right(&mut self, u: NodeRef) -> Option<NodeRef> {
            self.splay(u);
            let right = self.detach(u, Branch::Right)?;
            self.pull_up(u);
            Some(right)
        }

        pub fn merge_nonnull(&mut self, mut lhs: NodeRef, mut rhs: NodeRef) -> NodeRef {
            self.splay(lhs);
            self.splay_last(&mut lhs);
            self.splay(rhs);
            self.splay_first(&mut rhs);
            debug_assert!(self.is_root(lhs) && self.is_root(rhs) && lhs != rhs);
            self.attach(rhs, lhs, Branch::Left);
            self.pull_up(rhs);
            rhs
        }

        pub fn merge(&mut self, lhs: Option<NodeRef>, rhs: Option<NodeRef>) -> Option<NodeRef> {
            lift_binary(|lhs, rhs| self.merge_nonnull(lhs, rhs))(lhs, rhs)
        }
    }

    impl<V> Drop for SplayForest<V> {
        fn drop(&mut self) {
            for node in self.pool.iter_mut().skip(1) {
                unsafe {
                    node.assume_init_drop();
                }
            }
        }
    }
}

pub mod euler_tour_tree {
    use std::collections::HashMap;

    use super::splay;
    // use super::wbtree;

    fn rotate_to_front<S: splay::NodeSpec>(forest: &mut splay::SplayForest<S>, u: splay::NodeRef) {
        forest.splay(u);
        let left = forest.split_left(u);
        forest.merge(Some(u), left);
    }

    pub struct DynamicEulerTour<S: splay::NodeSpec> {
        pub forest: splay::SplayForest<S>,
        pub freed: Vec<splay::NodeRef>,

        pub verts: Vec<splay::NodeRef>,
        pub edges: HashMap<(u32, u32), splay::NodeRef>,
        pub n_verts: usize,
    }

    impl<S: splay::NodeSpec> DynamicEulerTour<S> {
        pub fn new(vert_nodes: impl IntoIterator<Item = S>) -> Self {
            let mut this = Self {
                forest: splay::SplayForest::new(),
                freed: vec![], // Reused deleted edge nodes

                verts: vec![],
                edges: HashMap::new(),
                n_verts: 0,
            };
            for node in vert_nodes {
                let u = this.forest.add_root(node);
                this.verts.push(u);
                this.n_verts += 1;
            }
            this
        }

        pub fn add_root(&mut self, node: S) -> splay::NodeRef {
            if let Some(u) = self.freed.pop() {
                self.forest.with(u, |u| {
                    *u = node;
                });
                u
            } else {
                self.forest.add_root(node)
            }
        }

        pub fn reroot(&mut self, u: usize) {
            rotate_to_front(&mut self.forest, self.edges[&(u as u32, u as u32)]);
        }

        pub fn find_root(&mut self, u: usize) -> splay::NodeRef {
            let mut u = self.verts[u];
            self.forest.splay(u);
            self.forest.splay_first(&mut u);
            u
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            self.find_root(u) == self.find_root(v)
        }

        pub fn link(&mut self, u: usize, v: usize, edge_uv: S, edge_vu: S) -> bool {
            if self.is_connected(u, v) {
                return false;
            }
            let vert_u = self.verts[u];
            let vert_v = self.verts[v];
            let edge_uv = self.add_root(edge_uv);
            let edge_vu = self.add_root(edge_vu);
            self.edges.insert((u as u32, v as u32), edge_uv);
            self.edges.insert((v as u32, u as u32), edge_vu);

            rotate_to_front(&mut self.forest, vert_u);
            rotate_to_front(&mut self.forest, vert_v);
            let lhs = self.forest.merge_nonnull(vert_u, edge_uv);
            let rhs = self.forest.merge_nonnull(vert_v, edge_vu);
            self.forest.merge_nonnull(lhs, rhs);
            true
        }

        pub fn cut(&mut self, u: usize, v: usize) -> bool {
            let (Some(edge_uv), Some(edge_vu)) = (
                self.edges.remove(&(u as u32, v as u32)),
                self.edges.remove(&(v as u32, u as u32)),
            ) else {
                return false;
            };

            rotate_to_front(&mut self.forest, edge_uv);
            self.forest.split_right(edge_uv);
            self.forest.split_left(edge_vu);
            self.forest.split_right(edge_vu);
            self.freed.push(edge_uv);
            self.freed.push(edge_vu);
            true
        }
    }
}

pub struct PureNode {
    label: debug::Label,
    link: splay::Link,
}

impl PureNode {
    pub fn new(label: debug::Label) -> Self {
        Self {
            label,
            link: splay::Link::default(),
        }
    }
}

impl splay::IntrusiveNode for PureNode {
    fn link(&self) -> &splay::Link {
        &self.link
    }
    fn link_mut(&mut self) -> &mut splay::Link {
        &mut self.link
    }
}

impl splay::NodeSpec for PureNode {}

#[derive(Clone)]
struct MinCount {
    min: i32,
    count: u32,
}

struct MinCountOp;

const INF: i32 = 1 << 30;

impl segtree_lazy::MonoidAction for MinCountOp {
    type X = MinCount;
    type F = i32;

    fn id(&self) -> Self::X {
        MinCount { min: INF, count: 0 }
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        match lhs.min.cmp(&rhs.min) {
            Ordering::Less => lhs.clone(),
            Ordering::Greater => rhs.clone(),
            Ordering::Equal => MinCount {
                min: lhs.min,
                count: lhs.count + rhs.count,
            },
        }
    }

    fn id_action(&self) -> Self::F {
        0
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        lhs + rhs
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &mut Self::X) {
        if *f != 0 {
            x_sum.min += f;
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let t: usize = input.value();

    let bs: Vec<u32> = (0..t + 1).map(|_| input.value::<u32>() - 1).collect();
    let groups: Vec<Vec<(u32, u32)>> = (0..m)
        .map(|_| {
            let k: usize = input.value();
            (0..k)
                .map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1))
                .collect()
        })
        .collect();
    let priority = |u: usize| groups[u].len() as u32;
    let mut events = vec![];
    for (w, i) in bs.windows(2).zip(0..t as u32) {
        let [mut u, mut v] = [w[0], w[1]];
        if priority(u as usize) < priority(v as usize) {
            std::mem::swap(&mut u, &mut v);
        }
        events.push((priority(u as usize), u, v, i));
    }
    events.sort_unstable();
    println!("{:?}", events);

    let mut conn = euler_tour_tree::DynamicEulerTour::new(
        (0..n).map(|u| PureNode::new(debug::Label::new_with(|| format!("vert {u}")))),
    );

    let mut ans = vec![0; t];
    let (mut u_prev, mut v_prev) = (!0, !0);
    for (_, u, v, i) in events {
        // Change u_prev -> u, v_prev -> v (the priority-based order garuntees O(M sqrt M) mods)
        // Identity forward / back edges
        // Build a min-plus segtree with min count
        if u != u_prev {
            // update u
        }
        if u != u_prev || v != v_prev {
            // update v
        }
    }

    // Constraints: K(1) + ... + K(m) <= C
    // Worst case time complexity:
    //     Assuming K(1) >= ... >= K(m),
    //     T = O(sum(K(i)) + sum min(K(i), K(i+1)))
    //     = O(C + 0 K(1) + 1 K(2) + 2 K(3) + ... + (sqrt(M) - 1) K(sqrt(M)))
    //     By the rearrangement inequality, the minimum is achieved when K(1) = ... = K(sqrt(M)) = C / sqrt(M),
    //     and T = O(C sqrt(M)).
    //
}
