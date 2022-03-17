pub trait IteratorExt: Iterator {
    fn for_each_group_by<P, F>(self, mut pred: P, mut f: F)
    where
        Self: Sized,
        P: FnMut(&Self::Item, &Self::Item) -> bool,
        F: FnMut(&[Self::Item]),
    {
        let mut group = Vec::new();
        let mut it = self.peekable();
        while let Some(x) = it.next() {
            let group_closed = match it.peek() {
                Some(y) => !pred(&x, &y),
                None => true,
            };
            group.push(x);
            if group_closed {
                f(&group[..]);
                group.clear();
            }
        }
    }
}

impl<T: Iterator> IteratorExt for T {}

#[cfg(test)]
mod test {
    use crate::iter::IteratorExt;
    #[test]
    fn tests() {
        (0..7).for_each_group_by(|x, y| x / 3 == y / 3, |group| {});
    }
}
