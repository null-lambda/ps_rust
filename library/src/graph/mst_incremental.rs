mod mst_incremental {
    // Fast Incremental minimum spanning tree
    // based on a direct implementation of Anti-Monopoly tree
    //
    // ## Reference
    // - Xiangyun Ding, Yan Gu, Yihan Sun.
    // "New Algorithms for Incremental Minimum Spanning Trees and Temporal Graph Applications".
    // [https://arxiv.org/abs/2504.04619]

    const UNSET: u32 = u32::MAX;
    const LINK_BY_STITCH: bool = false;

    #[derive(Clone, Debug)]
    struct Node<T> {
        parent: u32,
        size: i32,
        weight: T,
    }

    // Lazy Anti-Monopoly tree
    #[derive(Clone, Debug)]
    pub struct AMTree<T> {
        nodes: Vec<Node<T>>,
    }

    #[derive(Clone, Debug)]
    pub enum InsertType<T> {
        Connect,
        Replace(T),
    }

    impl<T: Default> InsertType<T> {
        pub fn ok(self) -> Option<T> {
            match self {
                InsertType::Connect => None,
                InsertType::Replace(w) => Some(w),
            }
        }
    }

    impl<T: Ord + Copy + Default> AMTree<T> {
        pub fn new(n_verts: usize) -> Self {
            Self {
                nodes: vec![
                    Node {
                        parent: UNSET,
                        size: 1,
                        weight: T::default() // Dummy
                    };
                    n_verts
                ],
            }
        }

        fn promote(&mut self, u: u32) {
            let p = self.nodes[u as usize].parent;
            let wu = self.nodes[u as usize].weight;
            let g = self.nodes[p as usize].parent;
            let wp = self.nodes[p as usize].weight;

            if wu >= wp && g != UNSET {
                // Shortcut
                self.nodes[u as usize].parent = g;
                self.nodes[p as usize].size -= self.nodes[u as usize].size;
            } else {
                // Rotate
                self.nodes[u as usize].parent = g;
                self.nodes[p as usize].parent = u;
                self.nodes[u as usize].weight = wp;
                self.nodes[p as usize].weight = wu;
                self.nodes[p as usize].size -= self.nodes[u as usize].size;
                self.nodes[u as usize].size += self.nodes[p as usize].size;
            }
        }

        fn perch(&mut self, u: u32) {
            while self.nodes[u as usize].parent != UNSET {
                self.promote(u);
            }
        }

        fn link_by_perch(&mut self, u: u32, v: u32, w: T) -> Option<InsertType<T>> {
            debug_assert!(u != v);
            self.perch(u);
            self.perch(v);
            if self.nodes[u as usize].parent == v {
                let w_old = self.nodes[u as usize].weight;
                if w < w_old {
                    self.nodes[u as usize].weight = w;
                    Some(InsertType::Replace(w_old))
                } else {
                    None
                }
            } else {
                debug_assert!(self.nodes[u as usize].parent == UNSET);
                self.nodes[u as usize].parent = v;
                self.nodes[u as usize].weight = w;
                self.nodes[v as usize].size += self.nodes[u as usize].size;
                Some(InsertType::Connect)
            }
        }

        fn cut_max_path(&mut self, mut u: u32, mut v: u32, w: T) -> Option<InsertType<T>> {
            debug_assert!(u != v);
            let mut w_old = None;
            loop {
                if self.nodes[u as usize].size > self.nodes[v as usize].size {
                    std::mem::swap(&mut u, &mut v);
                }

                let p = self.nodes[u as usize].parent;
                if p == UNSET {
                    // Disconnected
                    return Some(InsertType::Connect);
                }

                w_old = w_old.max(Some((self.nodes[u as usize].weight, u)));
                u = p;
                if u == v {
                    // reached LCA
                    let (w_old, mut t) = w_old.unwrap();
                    if w >= w_old {
                        return None;
                    }

                    // Unlink
                    let p = self.nodes[t as usize].parent;
                    self.nodes[t as usize].parent = UNSET;
                    self.nodes[t as usize].weight = T::default();
                    let delta_size = self.nodes[t as usize].size;

                    t = p;
                    while t != UNSET {
                        self.nodes[t as usize].size -= delta_size;
                        t = self.nodes[t as usize].parent;
                    }

                    return Some(InsertType::Replace(w_old));
                }
            }
        }

        fn link_by_stitch(&mut self, u: u32, v: u32, mut w: T) -> Option<InsertType<T>> {
            debug_assert!(u != v);

            let res = self.cut_max_path(u, v, w);
            if res.is_none() {
                return None;
            }

            let mut u = u;
            let mut v = v;
            let mut delta_size_u = 0i32;
            let mut delta_size_v = 0i32;
            loop {
                while self.nodes[u as usize].parent != UNSET && w >= self.nodes[u as usize].weight {
                    u = self.nodes[u as usize].parent;
                    self.nodes[u as usize].size += delta_size_u;
                }

                while self.nodes[v as usize].parent != UNSET && w >= self.nodes[v as usize].weight {
                    v = self.nodes[v as usize].parent;
                    self.nodes[v as usize].size += delta_size_v;
                }

                if self.nodes[u as usize].size > self.nodes[v as usize].size {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut delta_size_u, &mut delta_size_v);
                }

                let su = self.nodes[u as usize].size;
                delta_size_u -= su;
                delta_size_v += su;
                self.nodes[v as usize].size += su;

                std::mem::swap(&mut self.nodes[u as usize].weight, &mut w);
                u = std::mem::replace(&mut self.nodes[u as usize].parent, v);
                if u == UNSET {
                    loop {
                        v = self.nodes[v as usize].parent;
                        if v == UNSET {
                            return res;
                        }
                        self.nodes[v as usize].size += delta_size_v;
                    }
                }
                self.nodes[u as usize].size += delta_size_u;
            }
        }

        fn upward_calibrate(&mut self, mut u: u32) {
            loop {
                let p = self.nodes[u as usize].parent;
                if p == UNSET {
                    break;
                }

                if self.nodes[u as usize].size * 3 / 2 > self.nodes[p as usize].size {
                    self.promote(u);
                } else {
                    u = p;
                }
            }
        }

        pub fn insert(&mut self, u: u32, v: u32, w: T) -> Option<InsertType<T>> {
            if u == v {
                return None;
            }

            self.upward_calibrate(u);
            self.upward_calibrate(v);

            if LINK_BY_STITCH {
                self.link_by_stitch(u, v, w)
            } else {
                self.link_by_perch(u, v, w)
            }
        }

        pub fn max_path(&mut self, mut u: u32, mut v: u32) -> Option<T> {
            if u == v {
                return None;
            }

            self.upward_calibrate(u);
            self.upward_calibrate(v);

            let mut res = None;
            loop {
                if self.nodes[u as usize].size > self.nodes[v as usize].size {
                    std::mem::swap(&mut u, &mut v);
                }
                res = res.max(Some(self.nodes[u as usize].weight));
                u = self.nodes[u as usize].parent;
                if u == UNSET {
                    return None; // Disconnected
                }
                if u == v {
                    return res;
                }
            }
        }
    }
}
