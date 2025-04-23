pub mod chordal_graph {
    use crate::jagged::{self, Jagged};

    mod linked_list {
        use std::{
            marker::PhantomData,
            num::NonZeroU32,
            ops::{Index, IndexMut},
        };

        #[derive(Debug)]
        pub struct Cursor<T> {
            idx: NonZeroU32,
            _marker: PhantomData<*const T>,
        }

        // Arena-allocated pool of doubly linked lists.
        // Semantically unsafe, as cursors can outlive and access removed elements.
        #[derive(Clone, Debug)]
        pub struct MultiList<T> {
            links: Vec<[Option<Cursor<T>>; 2]>,
            values: Vec<T>,
            freed: Vec<Cursor<T>>,
        }

        impl<T> Clone for Cursor<T> {
            fn clone(&self) -> Self {
                Self::new(self.idx.get() as usize)
            }
        }

        impl<T> Copy for Cursor<T> {}

        impl<T> Cursor<T> {
            fn new(idx: usize) -> Self {
                Self {
                    idx: NonZeroU32::new(idx as u32).unwrap(),
                    _marker: PhantomData,
                }
            }

            pub fn usize(&self) -> usize {
                self.idx.get() as usize
            }
        }

        impl<T> Index<Cursor<T>> for MultiList<T> {
            type Output = T;
            fn index(&self, index: Cursor<T>) -> &Self::Output {
                &self.values[index.usize()]
            }
        }

        impl<T> IndexMut<Cursor<T>> for MultiList<T> {
            fn index_mut(&mut self, index: Cursor<T>) -> &mut Self::Output {
                &mut self.values[index.usize()]
            }
        }

        impl<T: Default> MultiList<T> {
            pub fn new() -> Self {
                Self {
                    links: vec![[None; 2]],
                    values: vec![Default::default()],
                    freed: vec![],
                }
            }

            pub fn next(&self, i: Cursor<T>) -> Option<Cursor<T>> {
                self.links[i.usize()][1]
            }

            pub fn prev(&self, i: Cursor<T>) -> Option<Cursor<T>> {
                self.links[i.usize()][0]
            }

            pub fn singleton(&mut self, value: T) -> Cursor<T> {
                if let Some(idx) = self.freed.pop() {
                    self.links[idx.usize()] = [None; 2];
                    self.values[idx.usize()] = value;
                    idx
                } else {
                    let idx = self.links.len();
                    self.links.push([None; 2]);
                    self.values.push(value);
                    Cursor::new(idx)
                }
            }

            fn link(&mut self, u: Cursor<T>, v: Cursor<T>) {
                self.links[u.usize()][1] = Some(v);
                self.links[v.usize()][0] = Some(u);
            }

            pub fn insert_left(&mut self, i: Cursor<T>, value: T) -> Cursor<T> {
                let v = self.singleton(value);
                if let Some(j) = self.prev(i) {
                    self.link(j, v);
                }
                self.link(v, i);
                v
            }

            pub fn insert_right(&mut self, i: Cursor<T>, value: T) -> Cursor<T> {
                let v = self.singleton(value);
                if let Some(j) = self.next(i) {
                    self.link(v, j);
                }
                self.link(i, v);
                v
            }

            pub fn erase(&mut self, i: Cursor<T>) {
                let l = self.prev(i);
                let r = self.next(i);
                if let Some(l_inner) = l {
                    self.links[l_inner.usize()][1] = r;
                }
                if let Some(r_inner) = r {
                    self.links[r_inner.usize()][0] = l;
                }
                self.links[i.usize()] = [None; 2];

                self.freed.push(i);
            }
        }
    }

    // Lexicographic bfs, O(N)
    pub fn lex_bfs(neighbors: &impl Jagged<u32>) -> (Vec<u32>, Vec<u32>) {
        let n = neighbors.len();

        let mut heads = linked_list::MultiList::new();
        let h_begin = heads.singleton(0u32);
        let h_end = heads.insert_right(h_begin, n as u32);

        let mut bfs: Vec<_> = (0..n as u32).collect();
        let mut t_in: Vec<_> = (0..n as u32).collect();
        let mut visited = vec![false; n];
        let mut owner = vec![h_begin; n];

        let erase_head = |heads: &mut linked_list::MultiList<u32>, h: &mut _| {
            if heads[*h] + 1 != heads[heads.next(*h).unwrap()] {
                heads[*h] += 1;
            } else {
                heads.erase(*h);
                *h = h_end;
            }
        };

        let mut t_split = vec![!0; n];
        for i in 0..n {
            let u = bfs[i] as usize;
            visited[u] = true;
            erase_head(&mut heads, &mut owner[u]);

            for &v in &neighbors[u] {
                let v = v as usize;
                if visited[v] {
                    continue;
                }

                let h = bfs[heads[owner[v]] as usize] as usize;
                t_in.swap(h, v);
                bfs.swap(t_in[h] as usize, t_in[v] as usize);

                if t_split.len() <= owner[v].usize() {
                    t_split.resize(owner[v].usize() + 1, !0);
                }
                let p = if t_split[owner[v].usize()] == i as u32 {
                    heads.prev(owner[v]).unwrap()
                } else {
                    t_split[owner[v].usize()] = i as u32;
                    heads.insert_left(owner[v], t_in[v])
                };
                erase_head(&mut heads, &mut owner[v]);
                owner[v] = p;
            }
        }

        (bfs, t_in)
    }

    // Perfect elimination ordering, reversed.
    pub fn rev_peo(neighbors: &impl Jagged<u32>) -> Option<(Vec<u32>, Vec<u32>)> {
        let n = neighbors.len();
        let (bfs, t_in) = lex_bfs(neighbors);

        let mut successors = vec![];
        for u in 0..n {
            let mut t_prev = None;
            for &v in &neighbors[u] {
                if t_in[v as usize] < t_in[u] {
                    t_prev = t_prev.max(Some(t_in[v as usize]));
                }
            }

            if let Some(t_prev) = t_prev {
                successors.push((t_prev, u as u32));
            }
        }
        let successors = jagged::CSR::from_pairs(n, &successors);

        let mut marker = vec![!0; n];
        for t_prev in 0..n {
            if successors[t_prev].is_empty() {
                continue;
            }

            let prev = bfs[t_prev as usize] as usize;
            for &v in &neighbors[prev] {
                marker[v as usize] = t_prev as u32;
            }

            for &u in &successors[t_prev] {
                for &v in &neighbors[u as usize] {
                    if t_in[v as usize] < t_in[prev] && marker[v as usize] != t_prev as u32 {
                        return None;
                    }
                }
            }
        }

        Some((bfs, t_in))
    }
}
