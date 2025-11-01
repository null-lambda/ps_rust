pub mod link_cut {
    use std::{
        fmt::Debug,
        ops::{Index, IndexMut},
    };

    pub const UNSET: u32 = !0;

    fn as_option(u: u32) -> Option<u32> {
        (u != UNSET).then(|| u)
    }

    // Intrusive node link
    #[derive(Debug)]
    pub struct Link {
        pub inv: bool,
        pub children: [u32; 2],
        pub parent: u32,
    }

    impl Default for Link {
        fn default() -> Self {
            Self {
                inv: false,
                children: [UNSET; 2],
                parent: UNSET,
            }
        }
    }

    pub trait NodeSpec {
        fn link(&self) -> &Link;
        fn link_mut(&mut self) -> &mut Link;

        fn push_down(&mut self, _cs: [Option<&mut Self>; 2]) {}
        fn pull_up(&mut self, _cs: [Option<&mut Self>; 2]) {}
        // TODO: fn reverse(&mut self) {}

        // Pseudo-top tree operation:
        // Attach or detach a chain (splice) from the rake tree of `u`.
        fn attach_virtual(&mut self, _c: &mut Self, _inv: bool) {}
    }

    #[derive(Debug)]
    pub struct LinkCutForest<S> {
        pub nodes: Vec<S>,
    }

    impl<S> Index<u32> for LinkCutForest<S> {
        type Output = S;
        fn index(&self, index: u32) -> &Self::Output {
            &self.nodes[index as usize]
        }
    }

    impl<S> IndexMut<u32> for LinkCutForest<S> {
        fn index_mut(&mut self, index: u32) -> &mut Self::Output {
            &mut self.nodes[index as usize]
        }
    }

    impl<S> FromIterator<S> for LinkCutForest<S> {
        fn from_iter<T: IntoIterator<Item = S>>(iter: T) -> Self {
            Self {
                nodes: Vec::from_iter(iter),
            }
        }
    }

    impl<S: NodeSpec> LinkCutForest<S> {
        pub unsafe fn get_many<const N: usize>(&mut self, us: [u32; N]) -> Option<[&mut S; N]> {
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

            let ptr = self.nodes.as_mut_ptr();
            Some(us.map(|u| unsafe { &mut *ptr.add(u as usize) }))
        }

        pub unsafe fn get_with_children<'a>(
            &'a mut self,
            u: u32,
        ) -> (&'a mut S, [Option<&'a mut S>; 2]) {
            unsafe {
                let pool_ptr = self.nodes.as_mut_ptr();
                let node = &mut *pool_ptr.add(u as usize);
                let children = node
                    .link()
                    .children
                    .map(|c| as_option(c).map(|c| &mut *pool_ptr.add(c as usize)));
                (node, children)
            }
        }

        fn reverse(&mut self, u: u32) {
            let link = self[u].link_mut();
            link.inv ^= true;
            link.children.swap(0, 1);
        }

        fn push_down(&mut self, u: u32) {
            unsafe {
                let link = self[u].link_mut();
                if link.inv {
                    link.inv = false;
                    for c in link.children {
                        if c != UNSET {
                            self.reverse(c);
                        }
                    }
                }

                let (u, cs) = self.get_with_children(u);
                u.push_down(cs);
            }
        }

        pub fn pull_up(&mut self, u: u32) {
            unsafe {
                let (nu, ncs) = self.get_with_children(u);
                nu.pull_up(ncs);
            }
        }

        pub fn attach_virtual(&mut self, u: u32, c: u32, inv: bool) {
            let [u, c] = unsafe { self.get_many([u, c]).unwrap() };
            u.attach_virtual(c, inv);
        }

        pub fn internal_parent(&self, u: u32) -> Option<(u32, u8)> {
            let p = self[u].link().parent;
            if p == UNSET {
                return None;
            }
            let [l, r] = self[p].link().children;
            if u == l {
                Some((p, 0))
            } else if u == r {
                Some((p, 1))
            } else {
                None
            }
        }

        pub fn path_parent(&self, u: u32) -> Option<u32> {
            let p = self[u].link().parent;
            if p == UNSET {
                return None;
            }

            let [l, r] = self[p].link().children;
            if u == l || u == r {
                None
            } else {
                Some(p) // path-parent
            }
        }

        pub fn is_root(&self, u: u32) -> bool {
            self[u].link().parent == UNSET
        }

        fn rotate(&mut self, u: u32) {
            let (p, bp) = self.internal_parent(u).expect("Root shouldn't be rotated");
            let c = std::mem::replace(&mut self[u].link_mut().children[(bp ^ 1) as usize], p);
            self[p].link_mut().children[bp as usize] = c;
            if let Some(c) = as_option(c) {
                self[c].link_mut().parent = p;
            }

            if let Some((g, bg)) = self.internal_parent(p) {
                self[g].link_mut().children[bg as usize] = u;
            }

            self[u].link_mut().parent = self[p].link().parent;
            self[p].link_mut().parent = u;
        }

        pub fn splay(&mut self, u: u32) {
            while let Some((p, _)) = self.internal_parent(u) {
                if let Some((g, _)) = self.internal_parent(p) {
                    self.push_down(g);
                    self.push_down(p);
                    self.push_down(u);

                    let bg = self[g].link().children[1] == p;
                    let bp = self[p].link().children[1] == u;
                    if bp == bg {
                        self.rotate(p); // zig-zig
                    } else {
                        self.rotate(u); // zig-zag
                    }
                    self.rotate(u);

                    self.pull_up(g);
                    self.pull_up(p);
                    self.pull_up(u);
                } else {
                    self.push_down(p);
                    self.push_down(u);

                    self.rotate(u); // zig

                    self.pull_up(p);
                    self.pull_up(u);
                }
            }
            self.push_down(u);
        }

        pub fn access(&mut self, u: u32) {
            self.splay(u);
            let c = std::mem::replace(&mut self[u].link_mut().children[1], UNSET);
            if c != UNSET {
                self.attach_virtual(u, c, false);
                self.pull_up(u);
            }

            while let Some(p) = self.path_parent(u) {
                self.splay(p);

                let old = std::mem::replace(&mut self[p].link_mut().children[1], u);
                self.attach_virtual(p, u, true);
                if old != UNSET {
                    self.attach_virtual(p, old, false);
                }

                self.splay(u);
            }
        }

        pub fn reroot(&mut self, u: u32) {
            self.access(u);
            self.reverse(u);
        }

        pub fn link(&mut self, u: u32, p: u32) -> bool {
            if self.is_connected(u, p) {
                return false;
            }

            self.access(p);
            self.reroot(u);

            self[u].link_mut().parent = p;
            self.attach_virtual(p, u, false);
            self.pull_up(p);
            true
        }

        pub fn cut(&mut self, u: u32, v: u32) -> bool {
            self.reroot(u);
            self.access(v);
            if std::mem::replace(&mut self[v].link_mut().children[0], UNSET) == UNSET {
                return false;
            }

            self[u].link_mut().parent = UNSET;
            self.pull_up(v);
            true
        }

        pub fn find_root(&mut self, mut u: u32) -> u32 {
            self.access(u);
            while let Some(l) = as_option(self[u].link().children[0]) {
                u = l;
                self.push_down(u);
            }
            self.splay(u);
            u
        }

        pub fn is_connected(&mut self, u: u32, v: u32) -> bool {
            self.find_root(u) == self.find_root(v)
        }

        pub fn get_parent(&mut self, u: u32) -> Option<u32> {
            self.access(u);
            let mut l = as_option(self[u].link().children[0])?;
            self.push_down(l);
            while let Some(right) = as_option(self[l].link().children[1]) {
                l = right;
                self.push_down(l);
            }
            self.splay(l);
            Some(l)
        }

        pub fn get_lca(&mut self, u: u32, v: u32) -> u32 {
            self.access(u);
            self.access(v);
            self.splay(u);
            as_option(self[u].link().parent).unwrap_or(u)
        }

        pub fn access_vertex_path(&mut self, u: u32, v: u32) {
            self.reroot(u);
            self.access(v);
            self.splay(u);
        }
    }
}
