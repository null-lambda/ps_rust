use std::io::Write;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    pub struct InputAtOnce {
        _buf: &'static str,
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let _buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let _buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(_buf, stat[6])) };
        let iter = _buf.split_ascii_whitespace();
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::new(stdout)
    }
}

#[allow(unused)]
pub mod splay {
    use super::*;

    use std::{
        cmp::Ordering,
        fmt, iter, mem,
        ops::{Bound, Index, IndexMut, Range, RangeBounds},
        ptr,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        Left = 0,
        Right = 1,
    }

    pub use Branch::{Left, Right};

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

    type Link = Option<Box<Node>>;

    pub struct Node {
        children: [Link; 2],
        pub value: u8,
        pub size: u32,
    }

    #[derive(Debug)]
    pub struct Tree {
        pub root: Link,
    }

    impl Node {
        fn new(value: u8) -> Self {
            Self {
                children: [None, None],
                size: 1,
                value,
            }
        }

        fn pull_up(&mut self) {
            self.size = 1;
            for branch in Branch::iter() {
                if let Some(child) = self.children[branch as usize].as_ref() {
                    self.size += child.size;
                }
            }
        }

        fn push_down(&mut self) {}

        fn splay_by<F>(self: &mut Box<Node>, mut cmp: F)
        where
            F: FnMut(&Node) -> Ordering,
        {
            let mut side_nodes = [vec![], vec![]];

            // cmp is called at most once for each nodes
            self.push_down();
            let mut ord = cmp(&self);
            loop {
                let branch = match ord {
                    Ordering::Equal => break,
                    Ordering::Less => Left,
                    Ordering::Greater => Right,
                };

                let Some(mut child) = self.children[branch as usize].take() else {
                    break;
                };
                child.push_down();
                let child_ord = cmp(&child);

                if child_ord == ord {
                    self.children[branch as usize] = child.children[branch.inv() as usize].take();
                    self.pull_up();
                    mem::swap(self, &mut child);
                    self.children[branch.inv() as usize] = Some(child);

                    let Some(next_child) = self.children[branch as usize].take() else {
                        break;
                    };
                    child = next_child;

                    child.push_down();
                    ord = cmp(&child);
                } else {
                    ord = child_ord;
                }
                side_nodes[branch.inv() as usize].push(mem::replace(self, child));
            }

            self.push_down();
            for (branch, nodes) in side_nodes.into_iter().enumerate() {
                self.children[branch] =
                    nodes
                        .into_iter()
                        .rev()
                        .fold(self.children[branch].take(), |acc, mut node| {
                            node.children[Branch::from(branch).inv() as usize] = acc;
                            node.pull_up();
                            Some(node)
                        });
            }
            self.pull_up();
        }
    }

    impl Tree {
        pub fn new() -> Self {
            Self { root: None }
        }

        pub fn size(&self) -> usize {
            self.root.as_ref().map_or(0, |node| node.size) as usize
        }

        pub fn splay_nth(&mut self, mut pos: usize) {
            debug_assert!(pos < self.size());
            self.root.as_mut().map(|x| {
                x.splay_by(|node| {
                    let size_left = node.children[Left as usize]
                        .as_ref()
                        .map_or(0, |x| x.size as usize);
                    let res = pos.cmp(&size_left);
                    match res {
                        Ordering::Less => {}
                        Ordering::Equal => pos = 0,
                        Ordering::Greater => pos -= size_left + 1,
                    };
                    res
                })
            });
        }

        pub fn splay_first(&mut self) {
            self.root.as_mut().map(|x| x.splay_by(|_| Ordering::Less));
        }

        pub fn splay_last(&mut self) {
            self.root
                .as_mut()
                .map(|x| x.splay_by(|_| Ordering::Greater));
        }

        pub fn join(mut left: Self, mut right: Self) -> Self {
            left.splay_last();
            let Some(root) = left.root.as_mut() else {
                return right;
            };
            root.push_down();
            root.children[Right as usize] = right.root.take();
            root.pull_up();
            left
        }

        pub fn split(mut self, branch: Branch, pos: usize) -> (Self, Self) {
            match branch {
                Left => {
                    if pos == self.size() {
                        (self.into(), None.into())
                    } else {
                        self.splay_nth(pos);
                        let mut root = self.root.as_mut().unwrap();
                        root.push_down();
                        let mut left = root.children[Left as usize].take();
                        root.pull_up();
                        (left.into(), self)
                    }
                }
                Right => {
                    if pos == 0 {
                        (None.into(), self.into())
                    } else {
                        self.splay_nth(pos - 1);

                        let mut root = self.root.as_mut().unwrap();
                        root.push_down();
                        let mut right = root.children[Right as usize].take();
                        root.pull_up();

                        (self, right.into())
                    }
                }
            }
        }

        pub fn take_range<R>(&mut self, range: R) -> Self
        where
            R: RangeBounds<usize>,
        {
            let l = match range.start_bound() {
                Bound::Included(&l) => l,
                Bound::Excluded(&l) => l + 1,
                Bound::Unbounded => 0,
            };
            let r = match range.end_bound() {
                Bound::Included(&r) => r + 1,
                Bound::Excluded(&r) => r,
                Bound::Unbounded => self.size(),
            };
            debug_assert!(l <= r && r <= self.size());

            let (left, mid_right) = unsafe { ptr::read(self) }.split(Right, l);
            let (mid, right) = mid_right.split(Left, r - l);
            unsafe { ptr::write(self, Self::join(left, right)) };
            mid
        }
    }

    impl From<Option<Box<Node>>> for Tree {
        fn from(root: Option<Box<Node>>) -> Self {
            Self { root }
        }
    }

    impl FromIterator<u8> for Tree {
        fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
            iter.into_iter()
                .map(|x| Box::new(Node::new(x)))
                .reduce(|acc, mut node| {
                    node.children[Left as usize] = Some(acc);
                    node.pull_up();
                    node
                })
                .into()
        }
    }

    impl fmt::Debug for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "{} {} {}",
                self.children[Left as usize]
                    .as_ref()
                    .map_or("_".to_owned(), |x| format!("({:?})", x)),
                self.value as u8,
                self.children[Right as usize]
                    .as_ref()
                    .map_or("_".to_owned(), |x| format!("({:?})", x)),
            )
        }
    }

    impl fmt::Display for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.children[Left as usize]
                .as_ref()
                .map_or(Ok(()), |x| write!(f, "{}", x))?;
            write!(f, "{:?}", self.value)?;
            self.children[Right as usize]
                .as_ref()
                .map_or(Ok(()), |x| write!(f, "{}", x))
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum VisitType {
        Preorder,
        Inorder,
        Postorder,
    }

    impl Tree {
        pub fn traverse<F>(&self, mut visitor: F)
        where
            F: FnMut(VisitType, &Node),
        {
            fn inner<F>(node: &Link, visitor: &mut F)
            where
                F: FnMut(VisitType, &Node),
            {
                if let Some(node) = node {
                    visitor(VisitType::Preorder, node);
                    inner(&node.children[Left as usize], visitor);
                    visitor(VisitType::Inorder, node);
                    inner(&node.children[Right as usize], visitor);
                    visitor(VisitType::Postorder, node);
                }
            }

            inner(&self.root, &mut visitor);
        }

        pub fn preorder<F>(&self, mut visitor: F)
        where
            F: FnMut(&Node),
        {
            self.traverse(|visit_type, node| {
                if visit_type == VisitType::Preorder {
                    visitor(node);
                }
            });
        }

        pub fn inorder<F>(&self, mut visitor: F)
        where
            F: FnMut(&Node),
        {
            self.traverse(|visit_type, node| {
                if visit_type == VisitType::Inorder {
                    visitor(node);
                }
            });
        }

        pub fn postorder<F>(&self, mut visitor: F)
        where
            F: FnMut(&Node),
        {
            self.traverse(|visit_type, node| {
                if visit_type == VisitType::Postorder {
                    visitor(node);
                }
            });
        }
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    for _ in 0..input.value() {
        let mut s: splay::Tree = input.token().bytes().collect();

        loop {
            let cmd = input.token();
            match cmd {
                "I" => {
                    let r: splay::Tree = input.token().bytes().collect();
                    let x: usize = input.value();
                    let (left, right) = s.split(splay::Right, x);
                    s = splay::Tree::join(left, splay::Tree::join(r, right));
                }
                "P" => {
                    let x: usize = input.value();
                    let y: usize = input.value();

                    let (left_mid, right) = s.split(splay::Right, y + 1);
                    let (left, mid) = left_mid.split(splay::Right, x);

                    let mut buf: Vec<u8> = vec![];
                    mid.inorder(|node| buf.push(node.value));
                    output.write_all(&buf).unwrap();
                    writeln!(output).unwrap();

                    s = splay::Tree::join(left, splay::Tree::join(mid, right));
                }
                "END" => {
                    break;
                }
                _ => panic!(),
            }
        }
    }
}
