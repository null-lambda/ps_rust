use std::io::Write;

use collections::DisjointSet;
use splay::LinkCutTree;

mod buffered_io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        fn value<T: FromStr>(&mut self) -> T
        where
            <T as FromStr>::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    // cheap and unsafe whitespace check
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};

    pub struct LineSyncedInput<R: BufRead> {
        line_buf: Vec<u8>,
        line_cursor: usize,
        inner: R,
    }

    impl<R: BufRead> LineSyncedInput<R> {
        pub fn new(r: R) -> Self {
            Self {
                line_buf: Vec::new(),
                line_cursor: 0,
                inner: r,
            }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.line_buf.len() - self.line_cursor);
            let slice = &self.line_buf[self.line_cursor..self.line_cursor + n];
            self.line_cursor += n;
            slice
        }

        fn eol(&self) -> bool {
            self.line_cursor == self.line_buf.len()
        }

        fn refill_line_buf(&mut self) -> bool {
            self.line_buf.clear();
            self.line_cursor = 0;
            let result = self.inner.read_until(b'\n', &mut self.line_buf).is_ok();
            result
        }
    }

    impl<R: BufRead> InputStream for LineSyncedInput<R> {
        fn token(&mut self) -> &[u8] {
            loop {
                if self.eol() {
                    let b = self.refill_line_buf();
                    if !b {
                        panic!(); // EOF
                    }
                }
                self.take(
                    self.line_buf[self.line_cursor..]
                        .iter()
                        .position(|&c| !is_whitespace(c))
                        .unwrap_or_else(|| self.line_buf.len() - self.line_cursor),
                );

                let idx = self.line_buf[self.line_cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.line_buf.len() - self.line_cursor);
                if idx > 0 {
                    return self.take(idx);
                }
            }
        }

        fn line(&mut self) -> &[u8] {
            if self.eol() {
                self.refill_line_buf();
            }

            self.line_cursor = self.line_buf.len();
            trim_newline(self.line_buf.as_slice())
        }
    }

    pub fn stdin() -> LineSyncedInput<BufReader<Stdin>> {
        LineSyncedInput::new(BufReader::new(std::io::stdin()))
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

pub mod splay {
    use std::{
        fmt::{self, Debug},
        marker::PhantomData,
        num::NonZeroU32,
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
    }

    type Link<'pool> = Option<NodeRef<'pool>>;

    #[derive(Debug, Default)]
    pub struct Node<'pool> {
        pub value: u32,
        pub sum: u32,
        inv: bool,
        children: [Link<'pool>; 2],
        internal_parent: Link<'pool>,
        _marker: PhantomData<&'pool ()>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct NodeRef<'pool> {
        pub idx: NonZeroU32,
        _marker: PhantomData<&'pool mut Node<'pool>>,
    }

    impl Debug for NodeRef<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    #[derive(Debug)]
    pub struct LinkCutTree<'pool> {
        pool: Vec<Node<'pool>>,
    }

    impl<'pool> LinkCutTree<'pool> {
        pub fn new() -> Self {
            let dummy = Node::default();
            Self { pool: vec![dummy] }
        }

        pub fn new_node(&mut self, value: u32, parent: Link<'pool>) -> NodeRef<'pool> {
            let idx = self.pool.len();
            self.pool.push(Node {
                value,
                sum: value,
                inv: false,
                children: [None, None],
                internal_parent: parent,
                _marker: PhantomData,
            });
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
                _marker: PhantomData,
            }
        }

        pub fn add_root(&mut self, value: u32) -> NodeRef<'pool> {
            let root = self.new_node(value, None);
            root
        }

        pub fn get_node<'a>(&'a self, ptr: NodeRef<'pool>) -> &'a Node<'pool> {
            &self.pool[ptr.idx.get() as usize]
        }

        pub fn get_node_mut<'a>(&'a mut self, ptr: NodeRef<'pool>) -> &'a mut Node<'pool> {
            &mut self.pool[ptr.idx.get() as usize]
        }

        pub fn get_node_with_children<'a>(
            &'a mut self,
            ptr: NodeRef<'pool>,
        ) -> (&'a mut Node<'pool>, [Option<&'a mut Node<'pool>>; 2]) {
            unsafe {
                let pool_ptr = self.pool.as_mut_ptr();
                let node = &mut *pool_ptr.add(ptr.idx.get() as usize);
                let children = node
                    .children
                    .map(|child| child.map(|child| &mut *pool_ptr.add(child.idx.get() as usize)));
                (node, children)
            }
        }

        fn push_down(&mut self, node: NodeRef<'pool>) {
            let node = self.get_node_mut(node);
            if node.inv {
                node.inv = false;
                node.children.swap(0, 1);

                let children = node.children;
                for child in children.into_iter().flatten() {
                    self.get_node_mut(child).inv ^= true;
                }
            }
        }

        fn pull_up(&mut self, node: NodeRef<'pool>) {
            let (node, children) = self.get_node_with_children(node);
            node.sum = node.value;
            for child in children.iter().flatten() {
                node.sum += child.sum;
            }
        }

        pub fn get_internal_parent(
            &self,
            node: NodeRef<'pool>,
        ) -> Result<(NodeRef<'pool>, Branch), Option<NodeRef<'pool>>> {
            match self.get_node(node).internal_parent {
                Some(parent) => {
                    if self.get_node(parent).children[Branch::Left as usize] == Some(node) {
                        Ok((parent, Branch::Left)) // parent on a chain
                    } else if self.get_node(parent).children[Branch::Right as usize] == Some(node) {
                        Ok((parent, Branch::Right)) // parent on a chain
                    } else {
                        Err(Some(parent)) // path-parent
                    }
                }
                None => Err(None), // true root
            }
        }

        pub fn is_root(&self, node: NodeRef<'pool>) -> bool {
            self.get_internal_parent(node).is_err()
        }

        fn attach(&mut self, node: NodeRef<'pool>, child: NodeRef<'pool>, branch: Branch) {
            debug_assert_ne!(node, child);
            self.get_node_mut(node).children[branch as usize] = Some(child);
            self.get_node_mut(child).internal_parent = Some(node);
        }

        fn detach(&mut self, node: NodeRef<'pool>, branch: Branch) -> Option<NodeRef<'pool>> {
            let child = self.get_node_mut(node).children[branch as usize].take()?;
            self.get_node_mut(child).internal_parent = None;
            Some(child)
        }

        fn rotate(&mut self, node: NodeRef<'pool>) {
            let (parent, branch) = self
                .get_internal_parent(node)
                .expect("Root shouldn't be rotated");
            let child = self.detach(node, branch.inv());
            if let Some(child) = child {
                self.attach(parent, child, branch);
            } else {
                self.detach(parent, branch);
            }

            match self.get_internal_parent(parent) {
                Ok((grandparent, grandbranch)) => {
                    self.attach(grandparent, node, grandbranch);
                }
                Err(path_parent) => {
                    self.get_node_mut(node).internal_parent = path_parent;
                }
            }
            self.attach(node, parent, branch.inv());

            self.pull_up(parent);
            self.pull_up(node);
        }

        pub fn splay(&mut self, node: NodeRef<'pool>) {
            while let Ok((parent, branch)) = self.get_internal_parent(node) {
                if let Ok((grandparent, grandbranch)) = self.get_internal_parent(parent) {
                    self.push_down(grandparent);
                    self.push_down(parent);
                    self.push_down(node);
                    if branch != grandbranch {
                        self.rotate(node);
                    } else {
                        self.rotate(parent);
                    }
                } else {
                    self.push_down(parent);
                    self.push_down(node);
                }
                self.rotate(node);
            }
            self.push_down(node);
        }

        pub fn access(&mut self, node: NodeRef<'pool>) {
            self.splay(node);
            self.get_node_mut(node).children[Branch::Right as usize] = None;
            while let Some(path_parent) =
                unsafe { self.get_internal_parent(node).unwrap_err_unchecked() }
            {
                self.splay(path_parent);
                self.get_node_mut(path_parent).children[Branch::Right as usize] = Some(node);
                self.splay(node);
            }
        }

        pub fn link(&mut self, parent: NodeRef<'pool>, child: NodeRef<'pool>) {
            debug_assert!(self.is_root(child));
            self.access(child);
            self.access(parent);
            self.attach(child, parent, Branch::Left);
        }

        pub fn cut(&mut self, child: NodeRef<'pool>) {
            self.access(child);
            self.detach(child, Branch::Left);
        }

        pub fn find_root(&mut self, mut node: NodeRef<'pool>) -> NodeRef<'pool> {
            self.access(node);
            while let Some(left) = self.get_node(node).children[Branch::Left as usize] {
                node = left;
                self.push_down(node);
            }
            self.splay(node);
            node
        }

        pub fn get_parent(&mut self, node: NodeRef<'pool>) -> Option<NodeRef<'pool>> {
            self.access(node);
            let mut left = self.get_node(node).children[Branch::Left as usize]?;
            self.push_down(left);
            while let Some(right) = self.get_node(left).children[Branch::Right as usize] {
                left = right;
                self.push_down(left);
            }
            self.splay(left);
            Some(left)
        }

        // pub fn get_depth(&mut self, node: NodeRef<'pool>) -> u32 {
        // self.access(node);
        // self.get_node(node).children[Branch::Left as usize].map_or(0, |c| self.get_node(c).size)
        // }

        pub fn get_lca(&mut self, lhs: NodeRef<'pool>, rhs: NodeRef<'pool>) -> NodeRef<'pool> {
            self.access(lhs);
            self.access(rhs);
            self.splay(lhs);
            self.get_node(lhs).internal_parent.unwrap_or(lhs)
        }

        pub fn reroot(&mut self, node: NodeRef<'pool>) {
            self.access(node);
            self.get_node_mut(node).inv ^= true;
        }

        pub fn query_vertex_path(&mut self, lhs: NodeRef<'pool>, rhs: NodeRef<'pool>) -> u32 {
            let mut res = 0;
            let join = self.get_lca(lhs, rhs);

            self.access(lhs);
            self.splay(join);
            res += self.get_node(join).value;
            if let Some(lhs_node) = self.get_node(join).children[Branch::Right as usize] {
                res += self.get_node(lhs_node).sum;
            }

            self.access(rhs);
            self.splay(join);
            if let Some(rhs_node) = self.get_node(join).children[Branch::Right as usize] {
                res += self.get_node(rhs_node).sum;
            }

            res
        }

        pub fn debug_topo(&self) {
            for i in 1..self.pool.len() {
                let parent = self.pool[i].internal_parent.map(|p| p.idx.get());
                let children = self.pool[i]
                    .children
                    .iter()
                    .map(|c| c.map(|c| c.idx.get()))
                    .collect::<Vec<_>>();
                print!("{:?}>{:?}>{:?} ", parent, i, children);
            }
            println!();
        }
    }
}

mod collections {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.find_root_with_size(u).1
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

fn main() {
    use buffered_io::*;
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();

    let mut tree = LinkCutTree::new();
    // O(alpha(N)) connectivity check + rerooting smaller tree on link
    let mut dset = DisjointSet::new(n);

    let mut nodes = vec![];
    for _ in 0..n {
        nodes.push(tree.add_root(input.value()));
    }

    let q: usize = input.value();
    for _ in 0..q {
        match input.token() {
            b"bridge" => {
                let mut a = input.value::<usize>() - 1;
                let mut b = input.value::<usize>() - 1;
                if !dset.merge(a, b) {
                    writeln!(output, "no").unwrap();
                } else {
                    writeln!(output, "yes").unwrap();
                    if dset.get_size(a) < dset.get_size(b) {
                        std::mem::swap(&mut a, &mut b);
                    }
                    tree.reroot(nodes[b]);
                    tree.link(nodes[a], nodes[b]);
                }
                output.flush().unwrap();
            }
            b"penguins" => {
                let a = input.value::<usize>() - 1;
                let x: u32 = input.value();
                tree.access(nodes[a]);
                let node_a = tree.get_node_mut(nodes[a]);
                node_a.sum = node_a.sum + x - node_a.value;
                node_a.value = x;
            }
            b"excursion" => {
                let a = input.value::<usize>() - 1;
                let b = input.value::<usize>() - 1;
                if dset.find_root(a) == dset.find_root(b) {
                    let ans = tree.query_vertex_path(nodes[a], nodes[b]);
                    writeln!(output, "{}", ans).unwrap();
                } else {
                    writeln!(output, "impossible").unwrap();
                }
                output.flush().unwrap();
            }
            _ => panic!(),
        }
    }
}
