pub mod global_map {
    use std::any::Any;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    trait DynHashEq {
        fn hash_dyn(&self, state: &mut dyn Hasher);
        fn eq_dyn(&self, other: &dyn DynHashEq) -> bool;
        fn as_any(&self) -> &dyn Any;
    }

    impl<T: Hash + Eq + Any> DynHashEq for T {
        fn hash_dyn(&self, state: &mut dyn Hasher) {
            struct DynHasher<'a>(&'a mut dyn Hasher);

            impl Hasher for DynHasher<'_> {
                fn finish(&self) -> u64 {
                    self.0.finish()
                }

                fn write(&mut self, bytes: &[u8]) {
                    self.0.write(bytes)
                }
            }

            (self.type_id(), self).hash(&mut DynHasher(state));
        }

        fn eq_dyn(&self, other: &dyn DynHashEq) -> bool {
            other
                .as_any()
                .downcast_ref::<T>()
                .is_some_and(|other| self.eq(other))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl PartialEq for dyn DynHashEq {
        fn eq(&self, other: &Self) -> bool {
            self.eq_dyn(other)
        }
    }

    impl Eq for dyn DynHashEq {}

    impl Hash for dyn DynHashEq {
        fn hash<H: Hasher>(&self, state: &mut H) {
            (self.as_any().type_id(), self);
            self.hash_dyn(state);
        }
    }

    thread_local! {
        static CACHE: RefCell<HashMap<Box<dyn DynHashEq>, Box<dyn Any>>> = Default::default();
    }

    pub fn insert<K: 'static + Hash + Eq, V: 'static>(key: K, value: V) {
        CACHE.with(|cache| cache.borrow_mut().insert(Box::new(key), Box::new(value)));
    }

    pub fn with_boxed_any<K: 'static + Hash + Eq, S>(
        key: &K,
        f: impl FnOnce(Option<&Box<dyn Any>>) -> S,
    ) -> S {
        let key: &dyn DynHashEq = key;
        CACHE.with(|cache| f(cache.borrow_mut().get(key)))
    }

    pub fn with<K: 'static + Hash + Eq, V: 'static, S>(
        key: &K,
        f: impl FnOnce(Option<&V>) -> S,
    ) -> S {
        self::with_boxed_any(key, |value| f(value.and_then(|v| v.downcast_ref())))
    }

    pub fn contains_key<K: 'static + Hash + Eq>(key: &K) -> bool {
        self::with_boxed_any(key, |value| value.is_some())
    }

    pub fn get_cloned<K: 'static + Hash + Eq, V: 'static + Clone>(key: &K) -> Option<V> {
        self::with(key, |value| value.cloned())
    }
}
