pub mod dset {
    pub mod potential {
        pub trait Group: Clone {
            fn id() -> Self;
            fn add_assign(&mut self, b: &Self);
            fn sub_assign(&mut self, b: &Self);
        }

        #[derive(Clone, Copy)]
        struct Link(i32); // Represents parent if >= 0, size if < 0

        impl Link {
            fn node(p: u32) -> Self {
                Self(p as i32)
            }

            fn size(s: u32) -> Self {
                Self(-(s as i32))
            }

            fn get(&self) -> Result<u32, u32> {
                if self.0 >= 0 {
                    Ok(self.0 as u32)
                } else {
                    Err((-self.0) as u32)
                }
            }
        }

        pub struct DisjointSet<E> {
            links: Vec<(Link, E)>,
        }

        impl<E: Group + Eq> DisjointSet<E> {
            pub fn with_size(n: usize) -> Self {
                Self {
                    links: (0..n).map(|_| (Link::size(1), E::id())).collect(),
                }
            }

            pub fn find_root_with_size(&mut self, u: usize) -> (usize, E, u32) {
                let (l, w) = &self.links[u];
                match l.get() {
                    Ok(p) => {
                        let mut w_acc = w.clone();
                        let (root, w_to_root, size) = self.find_root_with_size(p as usize);
                        w_acc.add_assign(&w_to_root);
                        self.links[u] = (Link::node(root as u32), w_acc.clone());
                        (root, w_acc, size)
                    }
                    Err(size) => (u, w.clone(), size),
                }
            }

            pub fn find_root(&mut self, u: usize) -> usize {
                self.find_root_with_size(u).0
            }

            pub fn get_size(&mut self, u: usize) -> u32 {
                self.find_root_with_size(u).2
            }

            // Returns true if two sets were previously disjoint
            pub fn merge(&mut self, u: usize, v: usize, mut weight_uv: E) -> Result<bool, ()> {
                let (mut u, mut weight_u, mut size_u) = self.find_root_with_size(u);
                let (mut v, mut weight_v, mut size_v) = self.find_root_with_size(v);
                if u == v {
                    let mut weight_u_expected = weight_uv;
                    weight_u_expected.add_assign(&weight_v);

                    if weight_u == weight_u_expected {
                        return Ok(false);
                    } else {
                        return Err(());
                    }
                }

                if size_u < size_v {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut weight_u, &mut weight_v);
                    std::mem::swap(&mut size_u, &mut size_v);

                    let mut neg = E::id();
                    neg.sub_assign(&weight_uv);
                    weight_uv = neg;
                }

                weight_u.add_assign(&weight_uv);
                weight_v.sub_assign(&weight_u);
                self.links[v] = (Link::node(u as u32), weight_v);
                self.links[u] = (Link::size(size_u + size_v), E::id());
                Ok(true)
            }

            pub fn delta_potential(&mut self, u: usize, v: usize) -> Option<E> {
                let (u, weight_u, _) = self.find_root_with_size(u);
                let (v, weight_v, _) = self.find_root_with_size(v);
                (u == v).then(|| {
                    let mut delta = weight_u.clone();
                    delta.sub_assign(&weight_v);
                    delta
                })
            }
        }
    }
}
