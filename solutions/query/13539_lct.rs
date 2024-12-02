use std::io::Write;

use splay::LinkCutTree;

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
        pub value: (),
        // inv: bool,
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

        pub fn new_node(&mut self, value: (), parent: Link<'pool>) -> NodeRef<'pool> {
            let idx = self.pool.len();
            self.pool.push(Node {
                value,
                // inv: false,
                children: [None, None],
                internal_parent: parent,
                _marker: PhantomData,
            });
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
                _marker: PhantomData,
            }
        }

        pub fn add_root(&mut self, value: ()) -> NodeRef<'pool> {
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
            // let (node, children) = self.get_node_with_children(node);
            // let node_mut = self.get_node_mut(node);
            // if node_mut.inv {
            //     node_mut.inv = false;
            //     node_mut.children.swap(0, 1);

            //     let children = node_mut.children;
            //     for child in children.into_iter().flatten() {
            //         self.get_node_mut(child).inv ^= true;
            //     }
            // }
        }

        fn pull_up(&mut self, node: NodeRef<'pool>) {
            let (node, children) = self.get_node_with_children(node);
            // node.size = 1;
            // for child in children.iter().flatten() {
            //     node.size += child.size;
            // }
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
            // self.reroot(child);
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
            // self.get_node_mut(node).inv ^= true;
        }

        pub fn from_iter(iter: impl IntoIterator<Item = ()>) -> (Self, Option<NodeRef<'pool>>) {
            let mut tree = LinkCutTree::new();
            let mut root = None;
            for value in iter {
                let node = tree.new_node(value, None);
                root = Some(root.map_or(node, |root| {
                    tree.attach(node, root, Branch::Left);
                    tree.pull_up(node);
                    node
                }));
            }
            (tree, root)
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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let m: usize = input.value();

    let mut tree = LinkCutTree::new();
    let mut nodes = vec![];
    for _ in 1..=n {
        nodes.push(tree.add_root(()));
    }

    for _ in 0..m {
        match input.token() {
            "1" => {
                let u = input.value::<usize>() - 1;
                let v = input.value::<usize>() - 1;
                tree.link(nodes[v], nodes[u]);
            }
            "2" => {
                let v = input.value::<usize>() - 1;
                tree.cut(nodes[v]);
            }
            "3" => {
                let u = input.value::<usize>() - 1;
                let v = input.value::<usize>() - 1;
                let join = tree.get_lca(nodes[u], nodes[v]);
                writeln!(output, "{}", join.idx).unwrap();
            }
            _ => panic!(),
        }
    }
}
