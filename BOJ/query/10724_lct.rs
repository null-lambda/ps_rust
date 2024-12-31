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
        pub value: (i32, u32),
        pub max: (i32, u32),
        inv: bool,
        pub children: [Link<'pool>; 2],
        internal_parent: Link<'pool>,
        _marker: PhantomData<&'pool ()>,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

        pub fn add_root(&mut self, value: (i32, u32)) -> NodeRef<'pool> {
            let idx = self.pool.len();
            self.pool.push(Node {
                value,
                max: value,
                inv: false,
                children: [None, None],
                internal_parent: None,
                _marker: PhantomData,
            });
            NodeRef {
                idx: unsafe { NonZeroU32::new(idx as u32).unwrap_unchecked() },
                _marker: PhantomData,
            }
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
            node.max = node.value;
            for child in children.iter().flatten() {
                node.max = node.max.max(child.max);
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
            self.reroot(child);
            self.access(child);
            self.access(parent);
            self.attach(child, parent, Branch::Left);
            self.pull_up(child);
        }

        pub fn cut(&mut self, child: NodeRef<'pool>) {
            self.access(child);
            if self.get_node(child).children[Branch::Left as usize].is_some() {
                self.detach(child, Branch::Left);
                self.pull_up(child);
            }
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

        pub fn query_vertex_path(
            &mut self,
            lhs: NodeRef<'pool>,
            rhs: NodeRef<'pool>,
        ) -> (i32, u32) {
            let mut res = (i32::MIN / 3, u32::MAX);
            let join = self.get_lca(lhs, rhs);

            self.access(lhs);
            self.splay(join);
            res = res.max(self.get_node(join).value);
            if let Some(lhs_node) = self.get_node(join).children[Branch::Right as usize] {
                res = res.max(self.get_node(lhs_node).max);
            }

            self.access(rhs);
            self.splay(join);
            if let Some(rhs_node) = self.get_node(join).children[Branch::Right as usize] {
                res = res.max(self.get_node(rhs_node).max);
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
                let v = self.pool[i].value;
                let max = self.pool[i].max;
                print!("{:?}>{:?}v{:?}m{:?}>{:?} ", parent, i, v, max, children);
            }
            println!();
        }
    }
}

mod tree {
    use std::iter;

    pub fn preorder_edge_lazy<'a, T: Copy>(
        neighbors: &'a [Vec<(usize, T)>],
        node: usize,
        parent: usize,
    ) -> impl Iterator<Item = (usize, usize, T)> + 'a {
        let mut stack = vec![(node, parent, neighbors[node].iter())];
        iter::from_fn(move || {
            stack.pop().map(|(node, parent, mut iter_child)| {
                let (child, weight) = *iter_child.next()?;
                stack.push((node, parent, iter_child));
                if child == parent {
                    return None;
                }
                stack.push((child, node, neighbors[child].iter()));
                Some((child, node, weight))
            })
        })
        .flatten()
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let m: usize = input.value();

        let mut tree = LinkCutTree::new();

        const NEG_INF: i32 = i32::MIN / 3;
        const UNSET: u32 = u32::MAX;
        let mut vert_nodes = vec![];
        for _ in 0..n {
            vert_nodes.push(tree.add_root((NEG_INF, UNSET)));
        }

        let mut neighbors = vec![vec![]; n];
        let mut mst_len = 0u64;
        for u in 1..n {
            let v: usize = input.value();
            let weight: i32 = input.value();
            neighbors[u].push((v, weight));
            neighbors[v].push((u, weight));

            mst_len += weight as u64;
        }

        let mut edges = vec![];
        let mut edge_idx = 0;
        for (u, p, weight) in tree::preorder_edge_lazy(&neighbors, 0, 0) {
            let edge_node = tree.add_root((weight, edge_idx));
            edges.push((p as u32, u as u32, edge_node));
            edge_idx += 1;

            tree.link(vert_nodes[p], edge_node);
            tree.link(edge_node, vert_nodes[u]);
        }

        let mut ans = 0;
        for _ in 0..m {
            let u: u32 = input.value();
            let v: u32 = input.value();
            let weight: i32 = input.value();

            if u != v {
                let (max_weight, ei) =
                    tree.query_vertex_path(vert_nodes[u as usize], vert_nodes[v as usize]);
                if weight < max_weight {
                    let (eu, ev, edge_node) = edges[ei as usize];

                    tree.reroot(edge_node);
                    tree.cut(vert_nodes[eu as usize]);
                    tree.cut(vert_nodes[ev as usize]);

                    edges[ei as usize] = (u, v, edge_node);
                    tree.get_node_mut(edge_node).value.0 = weight;
                    tree.link(vert_nodes[u as usize], edge_node);
                    tree.link(edge_node, vert_nodes[v as usize]);
                }
            }

            ans ^= mst_len;
        }
        writeln!(output, "{}", ans).unwrap();
    }
}
