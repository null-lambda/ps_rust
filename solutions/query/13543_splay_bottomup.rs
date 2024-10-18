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

const P: u64 = 1 << 32;

static mut COMB: [[u64; 11]; 11] = [[0; 11]; 11];

fn init_comb() {
    let n_max = 10;
    for i in 0..=n_max {
        unsafe {
            COMB[i][0] = 1;
            for j in 1..=i {
                COMB[i][j] = (COMB[i - 1][j - 1] + COMB[i - 1][j]) % P;
            }
        }
    }
}

fn comb(n: usize, k: usize) -> u64 {
    unsafe {
        debug_assert_ne!(COMB[0][0], 0, "Should call init_comb() first");
        COMB[n][k]
    }
}

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut res = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            res = res * base % P;
        }

        base = base * base % P;
        exp >>= 1;
    }
    res
}

#[allow(unused)]
pub mod splay {
    use super::*;

    use std::{
        fmt, iter, mem,
        ops::{Index, IndexMut, Range},
        ptr::{self, NonNull},
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Branch {
        Left = 0,
        Right = 1,
    }

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

    type Link = Option<NonNull<Node>>;
    pub struct Node {
        pub value: u64,
        pub sum: [u64; 11],
        pub count: u32,
        children: Children,
        parent: Link,
    }

    #[derive(Debug)]
    struct Children([Link; 2]);

    impl Children {
        fn get(&self, branch: Branch) -> Option<&Node> {
            self.0[branch as usize]
                .as_ref()
                .map(|x| unsafe { x.as_ref() })
        }

        fn get_mut(&mut self, branch: Branch) -> Option<&mut Node> {
            self.0[branch as usize]
                .as_mut()
                .map(|x| unsafe { x.as_mut() })
        }
    }

    impl Node {
        unsafe fn new(value: u64, parent: Option<NonNull<Node>>) -> Box<Self> {
            Box::new(Node {
                value,
                sum: [value; 11],
                count: 1,
                children: Children([None, None]),
                parent,
            })
        }

        fn attach(mut node: NonNull<Self>, branch: Branch, mut child: Option<NonNull<Self>>) {
            unsafe {
                // debug_assert!(self.children.0[branch as usize] == None);
                debug_assert_ne!(Some(node.as_ptr() as *mut _), child.map(|x| x.as_ptr()));

                node.as_mut().children.0[branch as usize] = child;
                if let Some(mut child) = child.as_mut() {
                    child.as_mut().parent = Some(node);
                }
            }
        }

        fn detach(mut node: NonNull<Self>, branch: Branch) -> Option<NonNull<Self>> {
            unsafe {
                node.as_mut().children.0[branch as usize]
                    .take()
                    .map(|mut child| {
                        child.as_mut().parent = None;
                        child
                    })
            }
        }

        fn is_root(&self) -> bool {
            self.parent.is_none()
        }

        fn branch(&self) -> Branch {
            debug_assert!(!self.is_root());
            unsafe {
                let node: NonNull<Self> = self.into();
                match node
                    .as_ref()
                    .parent
                    .unwrap()
                    .as_ref()
                    .children
                    .get(Branch::Left)
                {
                    Some(child) if ptr::eq(node.as_ptr(), child) => Branch::Left,
                    _ => Branch::Right,
                }
            }
        }

        fn rotate(mut node: NonNull<Self>) {
            unsafe {
                debug_assert!(!node.as_mut().is_root());

                let branch = node.as_mut().branch();

                let mut c = Node::detach(node, branch.inv());
                Node::attach(node.as_mut().parent.unwrap(), branch, c);

                let mut parent = node.as_mut().parent.unwrap();
                if let Some(mut grandparent) = parent.as_mut().parent {
                    Node::attach(grandparent, parent.as_ref().branch(), Some(node));
                } else {
                    node.as_mut().parent = None;
                }
                Node::attach(node, branch.inv(), Some(parent));

                parent.as_mut().update();
                node.as_mut().update();
            }
        }

        fn update(&mut self) {
            self.count = 1;
            for branch in Branch::iter() {
                if let Some(child) = self.children.get(branch) {
                    self.count += child.count;
                }
            }

            self.sum = [0; 11];
            let mut size_left = 1;
            if let Some(child) = &self.children.get(Branch::Left) {
                size_left = child.count + 1;
                for i in 0..=10 {
                    self.sum[i] += child.sum[i];
                    self.sum[i] %= P;
                }
            }
            for i in 0..=10 {
                self.sum[i] = (self.sum[i] + self.value * pow(size_left as u64, i as u64)) % P;
            }
            if let Some(child) = &self.children.get(Branch::Right) {
                for i in 0..=10 {
                    for k in 0..=i {
                        self.sum[i] += child.sum[k] * comb(i as usize, k as usize) % P
                            * pow(size_left as u64, (i - k) as u64);
                        self.sum[i] %= P;
                    }
                }
            }
        }

        pub fn validate_parents(&self) {
            if !cfg!(debug_assertions) {
                return;
            }

            unsafe {
                if let Some(parent) = self.parent {
                    debug_assert_eq!(
                        self as *const _,
                        parent.as_ref().children.get(self.branch()).unwrap() as *const _,
                        "Parent's child pointer does not point to self"
                    );
                }
                for branch in Branch::iter() {
                    if let Some(child) = self.children.get(branch) {
                        debug_assert_eq!(
                            Some(self as *const _),
                            child.parent.map(|x| x.as_ptr() as *const _),
                            "Child's parent pointer does not point to self: {:?} {:?}",
                            self,
                            child
                        );
                        debug_assert_ne!(child as *const _, self as *const _, "Self loop detected");
                    }
                }
            }
        }
    }

    pub struct Tree {
        root: Link,
    }

    impl Tree {
        pub fn into_root(mut self) -> Link {
            self.root.take()
        }

        pub fn size(&self) -> usize {
            unsafe { self.root.map_or(0, |x| x.as_ref().count as usize) }
        }

        fn splay(&mut self, mut node: NonNull<Node>) {
            debug_assert!(self.root.is_some());

            unsafe {
                let mut node_mut = node.as_ptr();
                while let Some(mut parent) = (*node_mut).parent {
                    if let Some(_grandparent) = parent.as_ref().parent {
                        if parent.as_ref().branch() == (*node_mut).branch() {
                            Node::rotate(parent);
                        } else {
                            Node::rotate(node);
                        }
                    }
                    Node::rotate(node);
                }

                self.root = Some(node);
            }
        }

        pub fn splay_nth(&mut self, mut n: usize) {
            debug_assert!(n < self.size(), "Out of Index: {} < {}", n, self.size());

            unsafe {
                let mut node: NonNull<Node> = self.root.unwrap();
                loop {
                    let mut child_count = (node)
                        .as_ref()
                        .children
                        .get(Branch::Left)
                        .map_or(0, |x| x.count as usize);

                    if child_count == n {
                        break;
                    } else if child_count < n {
                        node = node.as_mut().children.0[Branch::Right as usize].unwrap();
                        n -= child_count + 1;
                    } else {
                        node = node.as_mut().children.0[Branch::Left as usize].unwrap();
                    }
                }

                self.splay(node);
            }
        }

        fn split_right(&mut self) -> Option<Tree> {
            unsafe {
                let mut right = Node::detach(self.root?, Branch::Right);
                right.map(|x| Tree { root: Some(x) })
            }
        }

        fn merge(&mut self, rhs: Tree) {
            unsafe {
                if self.root.is_none() {
                    self.root = rhs.into_root();
                    return;
                }
                if rhs.root.is_none() {
                    return;
                }
                let mut node = self.root.unwrap();
                while let Some(mut right) = node.as_mut().children.0[Branch::Right as usize] {
                    node = right;
                }
                self.splay(node);
                Node::attach(self.root.unwrap(), Branch::Right, rhs.into_root());
                self.root.unwrap().as_mut().update();
            }
        }

        pub fn insert_left(&mut self, value: u64, pos: usize) {
            unsafe {
                let new_node = Some(NonNull::new_unchecked(Box::into_raw(Node::new(
                    value, None,
                ))));
                if self.root.is_none() {
                    debug_assert_eq!(pos, 0);
                    self.root = new_node;
                    return;
                }

                if pos == self.size() {
                    let mut p = self.root.unwrap();
                    while let Some(child) = p.as_mut().children.0[Branch::Right as usize] {
                        p = child;
                    }
                    Node::attach(p, Branch::Right, new_node);
                    self.splay(p);
                } else {
                    self.splay_nth(pos);

                    let Some(mut p) = self.root.unwrap().as_mut().children.0[Branch::Left as usize]
                    else {
                        Node::attach(self.root.unwrap(), Branch::Left, new_node);
                        self.root.unwrap().as_mut().update();
                        return;
                    };
                    while let Some(child) = p.as_ref().children.0[Branch::Right as usize] {
                        p = child;
                    }

                    Node::attach(p, Branch::Right, new_node);
                    self.splay(p);
                }
            }
        }

        pub fn remove(&mut self, pos: usize) {
            debug_assert!(pos < self.size());
            unsafe {
                self.splay_nth(pos);
                let mut right = self.split_right();
                let mut left = Node::detach(self.root.unwrap(), Branch::Left);
                let old = mem::replace(&mut self.root, left);
                if let Some(old) = old {
                    drop(Box::from_raw(old.as_ptr()));
                }

                if let Some(mut right) = right {
                    self.merge(right);
                }
            }
        }

        pub fn replace(&mut self, pos: usize, value: u64) {
            debug_assert!(pos < self.size());
            unsafe {
                self.splay_nth(pos);
                self.root.unwrap().as_mut().value = value;
                self.root.unwrap().as_mut().update();
            }
        }

        pub fn query_range<'a>(&'a mut self, range: Range<usize>) -> &'a Node {
            unsafe {
                let Range { start: l, end: r } = range;
                debug_assert!(l < r && r <= self.size());

                if r == self.size() {
                    if l == 0 {
                        &self.root.unwrap().as_ref()
                    } else {
                        self.splay_nth(l - 1);

                        let mut res: &mut Node = self
                            .root
                            .unwrap()
                            .as_mut()
                            .children
                            .get_mut(Branch::Right)
                            .unwrap();

                        res.update();
                        for branch in Branch::iter() {
                            res.children.get_mut(branch).map(|x| x.update());
                        }
                        res.update();

                        res
                    }
                } else {
                    self.splay_nth(r);
                    if l == 0 {
                        &self
                            .root
                            .unwrap()
                            .as_ref()
                            .children
                            .get(Branch::Left)
                            .unwrap()
                    } else {
                        let temp = self.root.unwrap();
                        self.splay_nth(l - 1);
                        let mut right = self.split_right().unwrap();
                        right.splay(temp);
                        let mut right = right.into_root();
                        Node::attach(self.root.unwrap(), Branch::Right, right.take());
                        self.root.unwrap().as_mut().update();

                        self.root
                            .unwrap()
                            .as_ref()
                            .children
                            .get(Branch::Right)
                            .unwrap()
                            .children
                            .get(Branch::Left)
                            .unwrap()
                    }
                }
            }
        }
    }

    impl FromIterator<u64> for Tree {
        fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
            unsafe {
                let mut root: Link = iter
                    .into_iter()
                    .map(|value| NonNull::new_unchecked(Box::into_raw(Node::new(value, None))))
                    .reduce(|mut acc, mut node| {
                        Node::attach(node, Branch::Left, Some(acc));
                        node.as_mut().update();
                        node
                    });
                // println!("{:?}", root.unwrap().as_ref());

                Tree { root }
            }
        }
    }

    impl Drop for Node {
        fn drop(&mut self) {
            unsafe {
                for branch in Branch::iter() {
                    if let Some(mut child) = self.children.0[branch as usize] {
                        let _owned = Box::from_raw(child.as_ptr());
                    }
                }
            }
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            unsafe {
                if let Some(mut root) = self.root {
                    let _owned = Box::from_raw(root.as_ptr());
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VisitType {
        Inorder,
        Preorder,
        Postorder,
    }

    impl Node {
        pub fn traverse(&self, mut visitor: impl FnMut(VisitType, &Node)) {
            pub fn inner(node: &Node, visitor: &mut impl FnMut(VisitType, &Node)) {
                visitor(VisitType::Preorder, node);
                if let Some(left) = node.children.get(Branch::Left) {
                    inner(left, visitor);
                }
                visitor(VisitType::Inorder, node);
                if let Some(right) = node.children.get(Branch::Right) {
                    inner(right, visitor);
                }
                visitor(VisitType::Postorder, node);
            }
            inner(self, &mut visitor);
        }

        pub fn inorder(&self, mut visitor: impl FnMut(&Node)) {
            self.traverse(|visit_type, node| {
                if visit_type == VisitType::Inorder {
                    visitor(node);
                }
            });
        }

        pub fn preorder(&self, mut visitor: impl FnMut(&Node)) {
            self.traverse(|visit_type, node| {
                if visit_type == VisitType::Preorder {
                    visitor(node);
                }
            });
        }

        pub fn postorder(&self, mut visitor: impl FnMut(&Node)) {
            self.traverse(|visit_type, node| {
                if visit_type == VisitType::Postorder {
                    visitor(node);
                }
            });
        }
    }

    impl fmt::Debug for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut res = Ok(());
            self.traverse(|visit_type, node| match visit_type {
                VisitType::Preorder => res = res.and_then(|_| write!(f, "(")),
                VisitType::Inorder => {
                    res = res.and_then(|_| write!(f, "{:?},{:?}", node.value, node.count))
                }
                VisitType::Postorder => res = res.and_then(|_| write!(f, ")")),
            });
            // self.preorder(|node| {
            //     res = res.and_then(|_| {
            //         writeln!(
            //             f,
            //             "{:?}>{:?}>{:?}",
            //             node.parent.map(|x| unsafe { x.as_ref().value }),
            //             node.value,
            //             Branch::iter()
            //                 .map(|b| node.children.get(b).map(|x| x.value))
            //                 .collect::<Vec<_>>()
            //         )
            //     })
            // });
            res?;
            // writeln!(f)
            Ok(())
        }
    }

    impl fmt::Debug for Tree {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unsafe { write!(f, "Tree {:?}", self.root.map(|x| x.as_ref())) }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();

    init_comb();

    let mut tree: splay::Tree = (0..n).map(|_| input.value()).collect();

    for _i_query in 0..input.value() {
        let cmd: u8 = input.value();
        match cmd {
            1 => {
                let p: usize = input.value();
                let x: u64 = input.value();
                tree.insert_left(x, p);
            }
            2 => {
                let p: usize = input.value();
                tree.remove(p);
            }
            3 => {
                let p: usize = input.value();
                let x: u64 = input.value();
                tree.replace(p, x);
                continue;
            }
            4 => {
                let l: usize = input.value();
                let r: usize = input.value();
                let k: usize = input.value();
                let node = tree.query_range(l..r + 1);
                writeln!(output, "{}", node.sum[k]).unwrap();
            }
            _ => panic!(),
        }
    }
}
