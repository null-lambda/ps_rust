use std::{
    io::Write,
    ptr::{self, NonNull},
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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

pub mod wbtree {
    // Weight-Balanced Binary Tree
    // https://koosaga.com/342

    use std::{
        cmp::Ordering,
        ops::Range,
        ptr::{self, NonNull},
    };

    #[derive(Clone, Copy, Debug)]
    pub struct Data {
        pub min: u32,
        pub max: u32,
        pub sum: u64,
    }

    impl Data {
        pub fn new(value: u32) -> Self {
            Self {
                min: value,
                max: value,
                sum: value as u64,
            }
        }

        pub fn id() -> Self {
            Self {
                min: u32::MAX,
                max: u32::MIN,
                sum: 0,
            }
        }
        pub fn combine(&self, other: &Self) -> Self {
            Self {
                min: self.min.min(other.min),
                max: self.max.max(other.max),
                sum: self.sum as u64 + other.sum as u64,
            }
        }
        pub fn reverse(&mut self) {}
    }

    type V = Data;

    pub type Link = Option<Box<Node>>;

    pub struct Node {
        pub size: u32,
        pub value: V,
        pub tag: Tag,
        parent: Option<NonNull<Node>>,
    }

    pub enum Tag {
        Leaf(Leaf),
        Branch(Branch),
    }

    pub struct Leaf;

    pub struct Branch {
        inv: bool,
        children: [Box<Node>; 2],
    }

    impl Tag {
        pub fn get_branch(&self) -> Option<&Branch> {
            match self {
                Tag::Leaf(..) => None,
                Tag::Branch(b) => Some(b),
            }
        }

        pub fn get_branch_mut(&mut self) -> Option<&mut Branch> {
            match self {
                Tag::Leaf(..) => None,
                Tag::Branch(b) => Some(b),
            }
        }

        pub fn into_branch(self) -> Option<Branch> {
            match self {
                Tag::Leaf(..) => None,
                Tag::Branch(b) => Some(b),
            }
        }

        pub fn get_leaf(&self) -> Option<&Leaf> {
            match self {
                Tag::Leaf(leaf) => Some(leaf),
                Tag::Branch(..) => None,
            }
        }
    }

    impl Node {
        fn new_leaf(value: V) -> Box<Self> {
            Box::new(Self {
                size: 1,
                value,
                tag: Tag::Leaf(Leaf),
                parent: None,
            })
        }

        fn new_branch(children: [Box<Node>; 2]) -> Box<Self> {
            let node = Self {
                size: 0,           // Uninit
                value: Data::id(), // Uninit
                tag: Tag::Branch(Branch {
                    inv: false,
                    children,
                }),
                parent: None,
            };
            let mut node = Box::new(node);
            unsafe {
                let ptr = NonNull::from(&*node);
                let children = &mut node.tag.get_branch_mut().unwrap_unchecked().children;
                children[0].parent = Some(ptr);
                children[1].parent = Some(ptr);
            }
            node.pull_up();
            node
        }

        fn pull_up(&mut self) {
            self.push_down();
            let Some(branch) = self.tag.get_branch_mut() else {
                return;
            };
            debug_assert!(!branch.inv);
            let [left, right] = &mut branch.children;
            left.push_down(); // Propagate inv field
            right.push_down(); // Propagate inv field
            self.size = left.size + right.size;
            self.value = left.value.combine(&right.value);
        }

        fn push_down(&mut self) {
            let Some(branch) = self.tag.get_branch_mut() else {
                return;
            };
            if branch.inv {
                branch.inv = false;
                branch.children.swap(0, 1);
                self.value.reverse();
                for b in 0..2 {
                    if let Tag::Branch(ref mut c) = branch.children[b].tag {
                        c.inv ^= true;
                    }
                }
            }
        }
    }

    pub struct Tree {
        pub root: Link,
    }

    fn should_rotate(size_left: u32, size_right: u32) -> bool {
        (size_left + 1) * 5 / 2 < (size_right + 1)
    }

    fn is_balanced(size_left: u32, size_right: u32) -> bool {
        !should_rotate(size_left, size_right) && !should_rotate(size_right, size_left)
    }

    impl Tree {
        pub fn empty() -> Self {
            Self { root: None }
        }

        pub fn check_balance(&mut self) -> bool {
            fn rec(node: &mut Node) -> bool {
                match &mut node.tag {
                    Tag::Leaf(_) => true,
                    Tag::Branch(branch) => {
                        let [left, right] = &mut branch.children;
                        is_balanced(left.size, right.size) && rec(left) && rec(right)
                    }
                }
            }
            self.root.as_mut().map_or(true, |root| rec(root))
        }

        pub fn check_parents(&self) -> bool {
            fn rec(node: &Node) -> bool {
                match &node.tag {
                    Tag::Leaf(_) => true,
                    Tag::Branch(branch) => {
                        let [left, right] = &branch.children;
                        let left_parent = left.parent.unwrap();
                        let right_parent = right.parent.unwrap();
                        if !ptr::eq(left_parent.as_ptr(), node)
                            || !ptr::eq(right_parent.as_ptr(), node)
                        {
                            return false;
                        }
                        rec(left) && rec(right)
                    }
                }
            }
            self.root.as_ref().map_or(true, |root| rec(root))
        }

        pub fn size(&self) -> usize {
            self.root.as_ref().map_or(0, |h| h.size as usize)
        }

        fn merge_nonempty(left: Box<Node>, right: Box<Node>) -> Box<Node> {
            unsafe {
                let sp = |mut node: Box<Node>| {
                    node.push_down();
                    let mut children = node.tag.into_branch().unwrap_unchecked().children;
                    children[0].parent = None;
                    children[1].parent = None;
                    children
                };
                let b = |left, right| Node::new_branch([left, right]);
                if should_rotate(left.size, right.size) {
                    let [mid, right] = sp(right);
                    let left = Self::merge_nonempty(left, mid);
                    if is_balanced(left.size, right.size) {
                        return b(left, right);
                    }
                    let [left, mid] = sp(left);
                    if is_balanced(left.size, mid.size + right.size)
                        && is_balanced(mid.size, right.size)
                    {
                        return b(left, b(mid, right));
                    }
                    let [mid_left, mid_right] = sp(mid);
                    return b(b(left, mid_left), b(mid_right, right));
                } else if should_rotate(right.size, left.size) {
                    let [left, mid] = sp(left);
                    let right = Self::merge_nonempty(mid, right);
                    if is_balanced(left.size, right.size) {
                        return b(left, right);
                    }
                    let [mid, right] = sp(right);
                    if is_balanced(left.size + mid.size, right.size)
                        && is_balanced(left.size, mid.size)
                    {
                        return b(b(left, mid), right);
                    }
                    let [mid_left, mid_right] = sp(mid);
                    return b(b(left, mid_left), b(mid_right, right));
                }
                b(left, right)
            }
        }

        pub fn merge(self, other: Tree) -> Tree {
            Tree {
                root: match (&self.root, &other.root) {
                    (None, None) => None,
                    (None, Some(_)) => other.root,
                    (Some(_), None) => self.root,
                    (Some(_), Some(_)) => Some(Self::merge_nonempty(
                        self.root.unwrap(),
                        other.root.unwrap(),
                    )),
                },
            }
        }

        fn split_branch(mut node: Node, pos: u32) -> (Box<Node>, Box<Node>) {
            debug_assert!(0 < pos && pos < node.size);
            node.push_down();
            let branch = unsafe {
                node.tag.into_branch().unwrap_unchecked() // size >= 2, so it must be a branch
            };
            let [mut left, mut right] = branch.children;
            left.parent = None;
            right.parent = None;
            match pos.cmp(&left.size) {
                Ordering::Equal => (left, right),
                Ordering::Less => {
                    let (left, mid) = Self::split_branch(*left, pos);
                    let right = Self::merge_nonempty(mid, right);
                    (left, right)
                }
                Ordering::Greater => {
                    let (mid, right) = Self::split_branch(*right, pos - left.size);
                    let left = Self::merge_nonempty(left, mid);
                    (left, right)
                }
            }
        }

        pub fn split(self, pos: usize) -> (Self, Self) {
            unsafe {
                let n = self.size();
                debug_assert!(pos <= n);
                if pos == 0 {
                    (Tree::empty(), Tree { root: self.root })
                } else if pos == n {
                    (Tree { root: self.root }, Tree::empty())
                } else {
                    let root = self.root.unwrap_unchecked();
                    let (left, right) = Self::split_branch(*root, pos as u32);
                    (Tree { root: Some(left) }, Tree { root: Some(right) })
                }
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> V {
            fn rec(node: &mut Node, bound: &Range<usize>, current: Range<usize>) -> V {
                unsafe {
                    if bound.end <= current.start || current.end <= bound.start {
                        V::id()
                    } else if bound.start <= current.start && current.end <= bound.end {
                        node.push_down();
                        node.value.clone()
                    } else {
                        node.push_down();
                        let branch = node.tag.get_branch_mut().unwrap_unchecked();
                        let [left, right] = &mut branch.children;
                        let mid = current.start + left.size as usize;
                        let left_value = rec(left, bound, current.start..mid);
                        let right_value = rec(right, bound, mid..current.end);
                        left_value.combine(&right_value)
                    }
                }
            }

            let n = self.size();
            self.root
                .as_mut()
                .map_or(V::id(), |root| rec(root, &range, 0..n))
        }

        pub fn get_pos(&mut self, node: NonNull<Node>) -> usize {
            unsafe {
                let mut curr = node;
                debug_assert!(curr.as_ref().tag.get_leaf().is_some());
                let mut idx = 0u32;
                while let Some(parent) = curr.as_ref().parent {
                    let branch = parent.as_ref().tag.get_branch().unwrap_unchecked();
                    if ptr::eq(curr.as_ptr(), &*branch.children[1]) {
                        idx += branch.children[0].size;
                    }
                    if branch.inv {
                        idx = parent.as_ref().size - 1 - idx;
                    }
                    curr = parent;
                }
                idx as usize
            }
        }

        pub fn from_iter(n: usize, xs: impl IntoIterator<Item = V>) -> Self {
            fn rec(xs: &mut impl Iterator<Item = V>, range: Range<u32>) -> Box<Node> {
                let Range { start, end } = range;
                debug_assert!(start != end);
                if start + 1 == end {
                    Node::new_leaf(xs.next().unwrap())
                } else {
                    let mid = (start + end) / 2;
                    Node::new_branch([rec(xs, start..mid), rec(xs, mid..end)])
                }
            }
            Self {
                root: (n > 0).then(|| rec(&mut xs.into_iter(), 0..n as u32)),
            }
        }

        pub fn for_each(&mut self, mut visitor: impl FnMut(&V)) {
            self.inorder(|node| {
                node.tag.get_leaf().is_some().then(|| visitor(&node.value));
            });
        }

        pub fn inorder(&mut self, mut visitor: impl FnMut(&mut Box<Node>)) {
            fn rec(node: &mut Box<Node>, visitor: &mut impl FnMut(&mut Box<Node>)) {
                node.push_down();
                match &mut node.tag {
                    Tag::Leaf(_) => visitor(node),
                    Tag::Branch(_) => {
                        let branch = unsafe { node.tag.get_branch_mut().unwrap_unchecked() };
                        rec(&mut branch.children[0], visitor);
                        visitor(node);
                        let branch = unsafe { node.tag.get_branch_mut().unwrap_unchecked() };
                        rec(&mut branch.children[1], visitor);
                    }
                }
            }

            if let Some(root) = &mut self.root {
                rec(root, &mut visitor);
            }
        }

        pub fn reverse(&mut self) {
            self.root
                .as_mut()
                .and_then(|root| root.tag.get_branch_mut())
                .map(|node| node.inv ^= true);
        }

        pub fn reverse_range(self, range: Range<usize>) -> Self {
            let Range { start, end } = range;
            let (left, rest) = self.split(start);
            let (mut mid, right) = rest.split(end - start);
            mid.reverse();
            let rest = mid.merge(right);
            left.merge(rest)
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let q: usize = input.value();

    let xs = 1..=n as u32;
    let mut tree = wbtree::Tree::from_iter(n, xs.map(wbtree::Data::new));

    let mut leaf_ptr = vec![ptr::null(); n + 1];
    tree.inorder(|node| {
        if let wbtree::Tag::Leaf(_) = node.tag {
            // Unsafe as hell
            // Better (but workful) approach would be using ptr everywhere, instead of mixing box
            // and ptr
            unsafe {
                let mut succ = |x: Box<wbtree::Node>| -> Box<_> {
                    let value = x.value.sum;
                    let ptr = Box::into_raw(x);
                    leaf_ptr[value as usize] = ptr;
                    Box::from_raw(ptr)
                };
                let next = succ(std::ptr::read(node));
                std::ptr::write(node, next);
            }
        }
    });

    for _ in 0..q {
        match input.token() {
            "1" => {
                let l: usize = input.value();
                let r: usize = input.value();
                let res = tree.query_range(l - 1..r);
                writeln!(output, "{} {} {}", res.min, res.max, res.sum).unwrap();

                tree = tree.reverse_range(l - 1..r);
            }
            "2" => {
                let l: usize = input.value();
                let r: usize = input.value();
                let x: isize = input.value();
                let res = tree.query_range(l - 1..r);
                writeln!(output, "{} {} {}", res.min, res.max, res.sum).unwrap();

                let s = r - l + 1;
                let x = (x % s as isize + s as isize) as usize % s;

                let (mid, right) = tree.split(r);
                let (left, mid) = mid.split(l - 1);
                let (m0, m1) = mid.split(s - x);
                tree = left.merge(m1).merge(m0).merge(right);
            }
            "3" => {
                let i: usize = input.value();
                let ith = tree.query_range(i - 1..i).sum;
                writeln!(output, "{}", ith).unwrap();
            }
            "4" => {
                let x: u32 = input.value();
                let ptr = NonNull::new(leaf_ptr[x as usize] as *mut wbtree::Node);
                let pos = tree.get_pos(ptr.unwrap());
                writeln!(output, "{}", pos + 1).unwrap();
            }
            _ => panic!(),
        }
    }
    tree.for_each(|node| write!(output, "{:?} ", node.sum).unwrap());

    writeln!(output).unwrap();
}
