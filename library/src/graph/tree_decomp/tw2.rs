pub mod map {
    use std::{hash::Hash, mem::MaybeUninit};

    pub enum AdaptiveHashSet<K, const STACK_CAP: usize> {
        Small([MaybeUninit<K>; STACK_CAP], usize),
        Large(std::collections::HashSet<K>),
    }

    impl<K, const STACK_CAP: usize> Drop for AdaptiveHashSet<K, STACK_CAP> {
        fn drop(&mut self) {
            match self {
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe { arr[i].assume_init_drop() }
                    }
                }
                _ => {}
            }
        }
    }

    impl<K: Clone, const STACK_CAP: usize> Clone for AdaptiveHashSet<K, STACK_CAP> {
        fn clone(&self) -> Self {
            match self {
                Self::Small(arr, size) => {
                    let mut cloned = std::array::from_fn(|_| MaybeUninit::uninit());
                    for i in 0..*size {
                        cloned[i] = MaybeUninit::new(unsafe { arr[i].assume_init_ref().clone() });
                    }
                    Self::Small(cloned, *size)
                }
                Self::Large(set) => Self::Large(set.clone()),
            }
        }
    }

    impl<K, const STACK_CAP: usize> Default for AdaptiveHashSet<K, STACK_CAP> {
        fn default() -> Self {
            Self::Small(std::array::from_fn(|_| MaybeUninit::uninit()), 0)
        }
    }

    impl<K: Eq + Hash, const STACK_CAP: usize> AdaptiveHashSet<K, STACK_CAP> {
        pub fn len(&self) -> usize {
            match self {
                Self::Small(_, size) => *size,
                Self::Large(set) => set.len(),
            }
        }

        pub fn contains(&self, key: &K) -> bool {
            match self {
                Self::Small(arr, size) => arr[..*size]
                    .iter()
                    .find(|&x| unsafe { x.assume_init_ref() } == key)
                    .is_some(),
                Self::Large(set) => set.contains(key),
            }
        }

        pub fn insert(&mut self, key: K) -> bool {
            if self.contains(&key) {
                return false;
            }
            match self {
                Self::Small(arr, size) if *size < STACK_CAP => {
                    arr[*size] = MaybeUninit::new(key);
                    *size += 1;
                    true
                }
                Self::Small(arr, size) => {
                    let arr =
                        std::mem::replace(arr, std::array::from_fn(|_| MaybeUninit::uninit()));
                    *size = 0;
                    *self = Self::Large(
                        arr.into_iter()
                            .map(|x| unsafe { x.assume_init() })
                            .chain(Some(key))
                            .collect(),
                    );
                    true
                }
                Self::Large(set) => set.insert(key),
            }
        }

        pub fn remove(&mut self, key: &K) -> bool {
            match self {
                Self::Small(_, 0) => false,
                Self::Small(arr, size) => {
                    for i in 0..*size {
                        unsafe {
                            if arr[i].assume_init_ref() == key {
                                *size -= 1;
                                arr[i].assume_init_drop();
                                arr[i] = std::mem::replace(&mut arr[*size], MaybeUninit::uninit());
                                return true;
                            }
                        }
                    }

                    false
                }
                Self::Large(set) => set.remove(key),
            }
        }

        pub fn for_each(&mut self, mut visitor: impl FnMut(&K)) {
            match self {
                Self::Small(arr, size) => {
                    arr[..*size]
                        .iter()
                        .for_each(|x| visitor(unsafe { x.assume_init_ref() }));
                }
                Self::Large(set) => set.iter().for_each(visitor),
            }
        }
    }
}

pub mod tree_decomp {
    // pub type HashSet<T> = std::collections::HashSet<T>;
    pub type HashSet<T> = crate::map::AdaptiveHashSet<T, 6>;

    pub const UNSET: u32 = u32::MAX;

    // Tree decomposition of treewidth 2.
    #[derive(Clone)]
    pub struct TW2 {
        // Perfect elimination ordering in the chordal completion
        pub topological_order: Vec<u32>,
        pub t_in: Vec<u32>,
        pub parents: Vec<[u32; 2]>,
    }

    impl TW2 {
        pub fn from_edges(
            n_verts: usize,
            edges: impl IntoIterator<Item = [u32; 2]>,
        ) -> Option<Self> {
            let mut neighbors = vec![HashSet::default(); n_verts];
            for [u, v] in edges {
                neighbors[u as usize].insert(v);
                neighbors[v as usize].insert(u);
            }

            let mut visited = vec![false; n_verts];
            let mut parents = vec![[UNSET; 2]; n_verts];
            let mut topological_order: Vec<_> = (0..n_verts as u32)
                .filter(|&u| neighbors[u as usize].len() <= 2)
                .inspect(|&u| visited[u as usize] = true)
                .collect();
            let mut t_in = vec![UNSET; n_verts];
            let mut timer = 0;
            let mut root = None;
            while let Some(&u) = topological_order.get(timer) {
                t_in[u as usize] = timer as u32;
                timer += 1;

                match neighbors[u as usize].len() {
                    0 => {
                        if let Some(old_root) = root {
                            parents[old_root as usize][0] = u;
                        }
                        root = Some(u);
                    }
                    1 => {
                        let mut p = UNSET;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| p = v);
                        neighbors[p as usize].remove(&u);

                        parents[u as usize][0] = p;

                        if !visited[p as usize] && neighbors[p as usize].len() <= 2 {
                            visited[p as usize] = true;
                            topological_order.push(p);
                        }
                    }
                    2 => {
                        let mut ps = [UNSET; 2];
                        let mut i = 0;
                        std::mem::take(&mut neighbors[u as usize]).for_each(|&v| {
                            ps[i] = v;
                            i += 1;
                        });
                        let [p, q] = ps;

                        neighbors[p as usize].remove(&u);
                        neighbors[q as usize].remove(&u);

                        neighbors[p as usize].insert(q);
                        neighbors[q as usize].insert(p);

                        parents[u as usize] = [p, q];

                        for w in [p, q] {
                            if !visited[w as usize] && neighbors[w as usize].len() <= 2 {
                                visited[w as usize] = true;
                                topological_order.push(w);
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }

            if topological_order.len() != n_verts {
                return None;
            }
            assert_eq!(root.as_ref(), topological_order.iter().last());

            for u in 0..n_verts {
                let ps = &mut parents[u];
                if ps[1] != UNSET && t_in[ps[0] as usize] > t_in[ps[1] as usize] {
                    ps.swap(0, 1);
                }
            }

            Some(Self {
                parents,
                topological_order,
                t_in,
            })
        }
    }
}
