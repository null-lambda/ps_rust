pub trait SliceExt<'a, T: 'a> {
    fn as_slice(&'a self) -> &'a [T];
    fn as_mut_slice(&'a mut self) -> &'a mut [T];

    fn group_indices_by(
        &'a self,
        mut pred: impl 'a + FnMut(&T, &T) -> bool,
    ) -> impl 'a + Iterator<Item = [usize; 2]> {
        let xs = self.as_slice();
        let mut i = 0;
        std::iter::from_fn(move || {
            if i == xs.len() {
                return None;
            }

            let mut j = i + 1;
            while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
                j += 1;
            }
            let res = [i, j];
            i = j;
            Some(res)
        })
    }
    fn group_indices_by_key<K: PartialEq>(
        &'a self,
        mut key: impl 'a + FnMut(&T) -> K,
    ) -> impl 'a + Iterator<Item = [usize; 2]> {
        self.group_indices_by(move |a, b| key(a) == key(b))
    }
    fn group_indices(&'a self) -> impl 'a + Iterator<Item = [usize; 2]>
    where
        T: PartialEq,
    {
        self.group_indices_by(|x, y| x == y)
    }

    fn group_by(
        &'a self,
        pred: impl 'a + FnMut(&T, &T) -> bool,
    ) -> impl 'a + Iterator<Item = &'a [T]> {
        let xs = self.as_slice();
        self.group_indices_by(pred).map(|w| &xs[w[0]..w[1]])
    }
    fn group_by_key<K: PartialEq>(
        &'a self,
        mut key: impl 'a + FnMut(&T) -> K,
    ) -> impl 'a + Iterator<Item = &'a [T]> {
        self.group_by(move |a, b| key(a) == key(b))
    }
    fn groups(&'a self) -> impl 'a + Iterator<Item = &'a [T]>
    where
        T: PartialEq,
    {
        self.group_by(|x, y| x == y)
    }

    fn partition_in_place(
        &'a mut self,
        mut pred: impl FnMut(&T) -> bool,
    ) -> (&'a mut [T], &'a mut [T]) {
        let xs = self.as_mut_slice();
        let n = xs.len();
        let mut i = 0;
        for j in 0..n {
            if pred(&xs[j]) {
                xs.swap(i, j);
                i += 1;
            }
        }
        xs.split_at_mut(i)
    }
}

impl<'a, T: 'a> SliceExt<'a, T> for [T] {
    fn as_slice(&'a self) -> &'a [T] {
        self
    }
    fn as_mut_slice(&'a mut self) -> &'a mut [T] {
        self
    }
}
