pub mod flat_ast {
    /// Flat AST with a layer-based recursion scheme.
    ///
    /// # Reference
    /// [Elegant and performant recursion in Rust](https://recursion.wtf/posts/rust_schemes/)
    use std::mem::MaybeUninit;

    // Emulate higher-kinded types with a trait.
    pub trait LayerKind {
        type Layer<T>;
        fn map<A, B>(layer: Self::Layer<A>, f: impl FnMut(A) -> B) -> Self::Layer<B>;
        fn as_ref<A>(layer: &Self::Layer<A>) -> Self::Layer<&A>;
    }

    #[derive(Clone)]
    pub struct FlatAst<K: LayerKind> {
        pub nodes: Vec<K::Layer<u32>>,
    }

    pub const ROOT: u32 = 0;

    impl<K: LayerKind> FlatAst<K> {
        pub fn collapse<A>(self, mut f: impl FnMut(K::Layer<A>) -> A) -> A {
            let mut dp: Vec<MaybeUninit<A>> =
                self.nodes.iter().map(|_| MaybeUninit::uninit()).collect();
            for (u, layer) in self.nodes.into_iter().enumerate().rev() {
                dp[u as usize] = MaybeUninit::new(f(K::map(layer, |c| unsafe {
                    std::mem::replace(&mut dp[c as usize], MaybeUninit::uninit()).assume_init()
                })));
            }

            unsafe { std::mem::replace(&mut dp[0], MaybeUninit::uninit()).assume_init() }
        }

        pub fn expand<A>(seed: A, mut f: impl FnMut(A) -> K::Layer<A>) -> Self {
            let mut nodes = vec![];
            let mut queue = vec![MaybeUninit::new(seed)];
            let mut timer = 0;
            while let Some(seed) = queue.get_mut(timer) {
                timer += 1;
                let seed = unsafe { std::mem::replace(seed, MaybeUninit::uninit()).assume_init() };
                let layer = K::map(f(seed), |c| {
                    queue.push(MaybeUninit::new(c));
                    (nodes.len() + queue.len() - timer) as u32
                });
                nodes.push(layer);
            }

            Self { nodes }
        }
    }
}
