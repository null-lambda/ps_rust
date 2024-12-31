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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

pub mod wbtree {
    // Weight-Balanced Binary Tree
    // https://koosaga.com/342

    use std::{cmp::Ordering, ops::Range};

    #[derive(Clone, Copy, Debug)]
    pub struct MinWithIdx {
        pub key: u32,
        size: u32,
        pub idx: u32,
    }

    impl MinWithIdx {
        pub fn new(key: u32) -> Self {
            Self {
                key,
                size: 1,
                idx: 0,
            }
        }

        pub fn id() -> Self {
            Self {
                key: u32::MAX,
                size: 0,
                idx: 0,
            }
        }
        pub fn combine(&self, other: &Self) -> Self {
            let size = self.size + other.size;
            if self.key <= other.key {
                Self {
                    key: self.key,
                    size,
                    idx: self.idx,
                }
            } else {
                Self {
                    key: other.key,
                    size,
                    idx: self.size + other.idx,
                }
            }
        }
        pub fn reverse(&mut self) {
            self.idx = self.size - self.idx - 1;
        }
    }

    type V = MinWithIdx;

    pub type Link = Option<Box<Node>>;

    pub struct Node {
        size: u32,
        value: V,
        pub tag: Tag,
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
        pub fn get_branch(&mut self) -> Option<&mut Branch> {
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
        fn new_leaf(value: V) -> Self {
            Self {
                size: 1,
                value,
                tag: Tag::Leaf(Leaf),
            }
        }

        fn new_branch(children: [Box<Node>; 2]) -> Self {
            let mut node = Self {
                size: 0,                 // Uninit
                value: MinWithIdx::id(), // Uninit
                tag: Tag::Branch(Branch {
                    inv: false,
                    children,
                }),
            };
            node.pull_up();
            node
        }

        fn pull_up(&mut self) {
            self.push_down();
            let Some(branch) = self.tag.get_branch() else {
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
            let Some(branch) = self.tag.get_branch() else {
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

    // fn should_rotate(size_left: u32, size_right: u32) -> bool {
    //   // println!(
    //   // "{}, {}",
    //   // SINGLE_ROT_THRES * (size_left + 1),
    //   // (size_right + 1)
    //   // );

    //   // SINGLE_ROT_THRES * (size_left + 1) < (size_right + 1)
    //   (size_left + 1) * 5 / 2 < (size_right + 1)
    // }
    // fn should_rotate_once(size_left: u32, size_right: u32) -> bool {
    //   println!(
    //       "{}, {}",
    //       SINGLE_ROT_THRES * (size_left + 1),
    //       DOUBLE_ROT_THRES * (size_right + 1)
    //   );
    //   // (size_left + 1) < DOUBLE_ROT_THRES * (size_right + 1)
    //   (size_left + 1) < (size_right + 1) * 3 / 2
    // }
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

        pub fn size(&self) -> usize {
            self.root.as_ref().map_or(0, |h| h.size as usize)
        }

        fn merge_nonempty(left: Box<Node>, right: Box<Node>) -> Box<Node> {
            unsafe {
                let sp = |mut node: Box<Node>| {
                    node.push_down();
                    node.tag.into_branch().unwrap_unchecked().children
                };
                let b = |left, right| Box::new(Node::new_branch([left, right]));
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
            let [left, right] = branch.children;
            let [left, right] = [left, right];
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
                        let branch = node.tag.get_branch().unwrap_unchecked();
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

        pub fn from_iter(n: usize, xs: impl IntoIterator<Item = V>) -> Self {
            fn rec(xs: &mut impl Iterator<Item = V>, range: Range<u32>) -> Box<Node> {
                let Range { start, end } = range;
                debug_assert!(start != end);
                Box::new(if start + 1 == end {
                    Node::new_leaf(xs.next().unwrap())
                } else {
                    let mid = (start + end) / 2;
                    Node::new_branch([rec(xs, start..mid), rec(xs, mid..end)])
                })
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

        pub fn inorder(&mut self, mut visitor: impl FnMut(&Node)) {
            fn rec(node: &mut Node, visitor: &mut impl FnMut(&Node)) {
                node.push_down();
                match &node.tag {
                    Tag::Leaf(_) => visitor(node),
                    Tag::Branch(_) => {
                        let branch = unsafe { node.tag.get_branch().unwrap_unchecked() };
                        rec(branch.children[0].as_mut(), visitor);
                        visitor(&node);
                        let branch = unsafe { node.tag.get_branch().unwrap_unchecked() };
                        rec(branch.children[1].as_mut(), visitor);
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
                .and_then(|root| root.tag.get_branch())
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

    loop {
        let n: usize = input.value();
        if n == 0 {
            break;
        }

        let mut xs: Vec<_> = (0..n as u32).map(|i| (input.value::<u32>(), i)).collect();
        xs.sort_unstable();
        let mut order = vec![0; n];
        for (i, (_, j)) in xs.iter().enumerate() {
            order[*j as usize] = i as u32;
        }

        let xs = order.into_iter().map(wbtree::MinWithIdx::new);

        let mut tree = wbtree::Tree::from_iter(n, xs);

        for i in 0..n {
            let swap_with = tree.query_range(i..n).idx as usize + i;
            write!(output, "{} ", swap_with + 1).unwrap();
            tree = tree.reverse_range(i..swap_with + 1);
            // assert!(tree.check_balance());
        }

        writeln!(output).unwrap();
    }
}
