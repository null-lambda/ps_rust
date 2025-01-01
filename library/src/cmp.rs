mod cmp {
    // The equalizer of all things
    use std::cmp::Ordering;

    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            // All values are equal, but Some(_)™ are more equal than others...
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}
