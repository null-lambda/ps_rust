pub mod iter {
    fn accumulate<T: Clone>(
        iter: impl IntoIterator<Item = T>,
        init: T,
        mut f: impl FnMut(T, T) -> T,
    ) -> impl Iterator<Item = T> {
        std::iter::once(init.clone()).chain(iter.into_iter().scan(init, move |acc, x| {
            *acc = f(acc.clone(), x);
            Some(acc.clone())
        }))
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
