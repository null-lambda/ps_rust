pub mod debug {
    #[cfg(debug_assertions)]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(T);

    #[cfg(not(debug_assertions))]
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct Label<T>(std::marker::PhantomData<T>);

    impl<T> Label<T> {
        #[inline]
        pub fn new_with(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(value())
            }
            #[cfg(not(debug_assertions))]
            {
                Self(Default::default())
            }
        }
    }

    impl<T: std::fmt::Debug> std::fmt::Debug for Label<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            #[cfg(debug_assertions)]
            {
                write!(f, "{:?}", self.0)
            }
            #[cfg(not(debug_assertions))]
            {
                write!(f, "()")
            }
        }
    }
}
