pub mod debug {
    use std::cell::Cell;

    thread_local! {
        static WORK: Cell<u64> = Cell::new(0);
    }

    pub fn work() {
        WORK.with(|work| work.set(work.get() + 1));
    }

    pub fn get_work() -> u64 {
        WORK.with(|work| work.get())
    }
}

pub mod bitset {
    pub type B = u64;
    pub const BLOCK_BITS: usize = B::BITS as usize;

    #[derive(Clone)]
    pub struct BitVec {
        masks: Vec<B>,
        size: usize,
    }

    impl BitVec {
        pub fn len(&self) -> usize {
            self.size
        }

        pub fn with_size(n: usize) -> Self {
            Self {
                masks: vec![B::default(); n.div_ceil(BLOCK_BITS)],
                size: n,
            }
        }

        pub fn get(&self, i: usize) -> bool {
            assert!(i < self.size);
            let (b, s) = (i / BLOCK_BITS, i % BLOCK_BITS);
            (self.masks[b] >> s) & 1 != 0
        }

        pub fn set(&mut self, i: usize, value: bool) {
            assert!(i < self.size);
            let (b, s) = (i / BLOCK_BITS, i % BLOCK_BITS);
            if !value {
                self.masks[b] &= !(1 << s);
            } else {
                self.masks[b] |= 1 << s;
            }
        }

        pub fn toggle(&mut self, i: usize) {
            assert!(i < self.size);
            let (b, s) = (i / BLOCK_BITS, i % BLOCK_BITS);
            self.masks[b] ^= 1 << s;
        }

        pub fn count_ones(&self) -> u32 {
            self.masks.iter().map(|&m| m.count_ones()).sum()
        }
    }

    impl FromIterator<bool> for BitVec {
        fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
            let iter = iter.into_iter();
            let (lower, upper) = iter.size_hint();
            let mut masks = Vec::with_capacity(upper.unwrap_or(lower).div_ceil(BLOCK_BITS));

            let mut mask = B::default();
            let mut s = 0;
            let mut size = 0;
            for bit in iter {
                if bit {
                    mask |= 1 << s;
                }
                s += 1;

                if s == BLOCK_BITS {
                    masks.push(mask);
                    size += s;
                    mask = 0;
                    s = 0;
                }
            }
            if s != 0 {
                size += s;
                masks.push(mask);
            }

            BitVec { masks, size }
        }
    }

    impl std::fmt::Debug for BitVec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[")?;
            for i in 0..self.size {
                write!(f, "{}", self.get(i) as u8)?;
            }
            write!(f, "]")?;
            Ok(())
        }
    }
}

pub mod matroid_inter {
    pub(crate) type BitVec = crate::bitset::BitVec;

    pub const UNSET: u32 = u32::MAX;

    // An abstract query structure for building an exchange graph.
    // Use lazy or amortized evaluation if possible.
    pub trait ExchangeOracle {
        fn len(&self) -> usize;

        fn load_indep_set(&mut self, indep_set: &BitVec);

        // Test whether I U {i} is independent.
        fn can_insert(&mut self, i: usize) -> bool;

        // Test whether I - {i} + {j} is indepdendent.
        fn can_exchange(&mut self, _i: usize, _j: usize) -> bool {
            unimplemented!()
        }

        // Assuming i in I, visit all exchangable j.
        fn left_exchange(&mut self, indep_set: &BitVec, i: usize, mut visitor: impl FnMut(usize)) {
            if !indep_set.get(i) {
                return;
            }
            for j in 0..self.len() {
                if !indep_set.get(j) && self.can_exchange(i, j) {
                    visitor(j);
                }
            }
        }

        // Assuming j not in I, visit all exchangable j.
        fn right_exchange(&mut self, indep_set: &BitVec, j: usize, mut visitor: impl FnMut(usize)) {
            if indep_set.get(j) {
                return;
            }
            for i in 0..self.len() {
                if indep_set.get(i) && self.can_exchange(i, j) {
                    visitor(i);
                }
            }
        }
    }

    pub fn inter(m1: &mut impl ExchangeOracle, m2: &mut impl ExchangeOracle) -> (BitVec, usize) {
        assert_eq!(m1.len(), m2.len());
        let mut set = BitVec::with_size(m1.len());
        let mut rank = 0;
        while augment(m1, m2, &mut set) {
            rank += 1;
        }
        (set, rank)
    }

    fn ascend_to_root(parent: &[u32], mut u: usize, mut visitor: impl FnMut(usize)) {
        loop {
            visitor(u);

            if u == parent[u] as usize {
                break;
            }
            u = parent[u] as usize;
        }
    }

    fn augment(
        m1: &mut impl ExchangeOracle,
        m2: &mut impl ExchangeOracle,
        indep_set: &mut BitVec,
    ) -> bool {
        let n = m1.len();
        m1.load_indep_set(&indep_set);
        m2.load_indep_set(&indep_set);

        let mut parent = vec![UNSET; n];
        let mut bfs = vec![];
        for i in 0..n {
            if !indep_set.get(i) && m1.can_insert(i) {
                bfs.push(i as u32);
                parent[i] = i as u32;
            }
        }

        let is_dest: Vec<bool> = (0..n)
            .map(|i| !indep_set.get(i) && m2.can_insert(i))
            .collect();
        let mut timer = 0;

        while let Some(u) = bfs.get(timer).map(|&u| u as usize) {
            timer += 1;

            if is_dest[u] {
                ascend_to_root(&parent, u as usize, |u| indep_set.toggle(u));
                return true;
            }

            let mut try_enqueue = |v| {
                super::debug::work();
                if parent[v] == UNSET {
                    parent[v] = u as u32;
                    bfs.push(v as u32);
                }
            };

            m1.left_exchange(&indep_set, u, |v| try_enqueue(v));
            m2.right_exchange(&indep_set, u, |v| try_enqueue(v));
        }

        false
    }
}
