#[allow(dead_code)]
pub mod iter {
    fn group_by<T, P, F>(xs: &[T], mut pred: P, mut f: F)
    where
        P: FnMut(&T, &T) -> bool,
        F: FnMut(&[T]),
    {
        let mut i = 0;
        while i < xs.len() {
            let mut j = i + 1;
            while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
                j += 1;
            }
            f(&xs[i..j]);
            i = j;
        }
    }

    pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)>
    where
        I: IntoIterator,
        I::Item: Clone,
        J: IntoIterator,
        J::IntoIter: Clone,
    {
        let j = j.into_iter();
        i.into_iter()
            .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
    }
}
