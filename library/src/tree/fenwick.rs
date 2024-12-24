pub mod fenwick_tree {
    pub trait Group {
        type Elem: Clone;
        fn id(&self) -> Self::Elem;
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
    }

    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n_ceil = n.next_power_of_two();
            let data = (0..n_ceil).map(|_| group.id()).collect();
            Self { n, group, data }
        }

        pub fn add(&mut self, mut idx: usize, value: G::Elem) {
            while idx < self.n {
                self.group.add_assign(&mut self.data[idx], value.clone());
                idx |= idx + 1;
            }
        }
        pub fn get(&self, idx: usize) -> G::Elem {
            self.sum_range(idx..idx + 1)
        }

        pub fn sum_prefix(&self, idx: usize) -> G::Elem {
            let mut res = self.group.id();
            let mut r = idx;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::Elem {
            let mut res = self.sum_prefix(range.end);
            self.group
                .sub_assign(&mut res, self.sum_prefix(range.start));
            res
        }
    }
}
