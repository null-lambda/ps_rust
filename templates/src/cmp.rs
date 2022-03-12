mod cmp {
    use std::cmp::Ordering;

    // x <= y iff x = y
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        #[inline]
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        #[inline]
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        #[inline]
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}