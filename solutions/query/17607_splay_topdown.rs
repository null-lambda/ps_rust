use std::io::Write;

use splay::Branch;

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

    type Link = Option<Box<Node>>;

    pub struct Node {
        children: [Link; 2],
        pub value: bool,
        pub size: u32,
        pub count_left: u32,
        pub count_right: u32,
        pub count_inside: u32,
        pub inv: bool,
    }

    #[derive(Debug)]
    pub struct Tree {
        pub root: Link,
    }

    impl Node {
        fn new(value: bool) -> Self {
            Self {
                children: [None, None],
                size: 1,
                value,
                count_left: value as u32,
                count_right: value as u32,
                count_inside: value as u32,
                inv: false,
            }
        }

        fn is_full(&self) -> bool {
            self.size == self.count_inside
        }

        fn pull_up(&mut self) {
            debug_assert!(!self.inv);
            self.size = 1;

            self.children[Left as usize].as_mut().map(|x| x.push_down());
            self.children[Right as usize]
                .as_mut()
                .map(|x| x.push_down());

            match &self.children {
                [Some(left), Some(right)] => {
                    self.size += left.size + right.size;
                    self.count_left = left.count_left;
                    if self.value && left.is_full() {
                        self.count_left += right.count_left + 1;
                    }
                    self.count_right = right.count_right;
                    if self.value && right.is_full() {
                        self.count_right += left.count_right + 1;
                    }
                    self.count_inside = left.count_inside.max(right.count_inside);
                    if self.value {
                        self.count_inside = self
                            .count_inside
                            .max(left.count_right + 1 + right.count_left);
                    }
                }
                [Some(left), None] => {
                    self.size += left.size;
                    self.count_left = left.count_left + (self.value && left.is_full()) as u32;
                    self.count_right = if self.value { 1 + left.count_right } else { 0 };
                    self.count_inside = left.count_inside;
                    if self.value {
                        self.count_inside = self.count_inside.max(left.count_right + 1);
                    }
                }
                [None, Some(right)] => {
                    self.size += right.size;
                    self.count_right = right.count_right + (self.value && right.is_full()) as u32;
                    self.count_left = if self.value { 1 + right.count_left } else { 0 };
                    self.count_inside = right.count_inside;
                    if self.value {
                        self.count_inside = self.count_inside.max(right.count_left + 1);
                    }
                }
                [None, None] => {
                    self.count_left = self.value as u32;
                    self.count_right = self.value as u32;
                    self.count_inside = self.value as u32;
                }
            }
        }

        fn push_down(&mut self) {
            if self.inv {
                self.inv = false;

                self.children.swap(0, 1);
                mem::swap(&mut self.count_left, &mut self.count_right);
                for branch in Branch::iter() {
                    if let Some(child) = self.children[branch as usize].as_mut() {
                        child.inv ^= true;
                    }
                }
            }
        }

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
            // println!("call splay_nth");
            self.root.as_mut().map(|x| {
                x.splay_by(|node| {
                    // println!("call comparator at pos {pos}, {:?}", node);
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

        pub fn merge(mut left: Self, mut right: Self) -> Self {
            left.splay_nth(left.size());
            left.root.as_mut().unwrap().children[Right as usize] = right.root.take();
            left.root.as_mut().unwrap().pull_up();
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

        pub fn join(&mut self, branch: Branch, mut child: Self) {
            let child = child.root;
            match self.root {
                None => self.root = child,
                Some(ref mut root) => {
                    debug_assert!(root.children[branch as usize].is_none());

                    root.push_down();
                    root.children[branch as usize] = child;
                    root.pull_up();
                }
            }
        }

        pub fn join_to(&mut self, branch: Branch, mut parent: Self) {
            mem::swap(self, &mut parent);
            self.join(branch, parent);
        }

        pub fn split_range<R>(&mut self, range: R) -> (Self, Self)
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
            unsafe { ptr::write(self, mid) };
            (left, right)
        }

        pub fn reverse(&mut self) {
            self.root.as_mut().map(|x| x.inv ^= true);
        }
    }

    impl From<Option<Box<Node>>> for Tree {
        fn from(root: Option<Box<Node>>) -> Self {
            Self { root }
        }
    }

    impl FromIterator<bool> for Tree {
        fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
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
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let mut xs: splay::Tree = (0..n).map(|_| input.token() == "1").collect();

    let m: usize = input.value();
    for _ in 0..m {
        let cmd = input.token();
        let l: usize = input.value();
        let r: usize = input.value();

        let (left, right) = xs.split_range(l - 1..=r - 1);
        match cmd {
            "1" => xs.reverse(),
            "2" => {
                let root = xs.root.as_ref().unwrap();
                writeln!(output, "{}", root.count_inside).unwrap();
            }
            _ => panic!(),
        }

        xs.join_to(Branch::Left, right);
        xs.join_to(Branch::Right, left);
    }
}
