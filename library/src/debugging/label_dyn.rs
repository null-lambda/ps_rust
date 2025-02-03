pub mod debug {
    use std::{fmt::Debug, rc::Rc};

    #[cfg(debug_assertions)]
    #[derive(Clone)]
    pub struct Label(Rc<dyn Debug>);

    #[cfg(not(debug_assertions))]
    #[derive(Clone)]
    pub struct Label;

    impl Label {
        #[inline]
        pub fn new_with<T: Debug + 'static>(value: impl FnOnce() -> T) -> Self {
            #[cfg(debug_assertions)]
            {
                Self(Rc::new(value()))
            }
            #[cfg(not(debug_assertions))]
            {
                Self
            }
        }
    }

    impl Debug for Label {
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

    impl Default for Label {
        fn default() -> Self {
            Self::new_with(|| ())
        }
    }

    impl PartialEq for Label {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl Eq for Label {}

    impl PartialOrd for Label {
        fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
            Some(std::cmp::Ordering::Equal)
        }
    }

    impl Ord for Label {
        fn cmp(&self, _: &Self) -> std::cmp::Ordering {
            std::cmp::Ordering::Equal
        }
    }

    impl std::hash::Hash for Label {
        fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
    }
}
