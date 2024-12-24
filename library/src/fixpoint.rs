// Y combinator for recursive closures
trait FnLike<A, B> {
    fn call(&self, x: A) -> B;
}

impl<A, B, F: Fn(A) -> B> FnLike<A, B> for F {
    fn call(&self, x: A) -> B {
        self(x)
    }
}

fn fix<A, B, F: Fn(&dyn FnLike<A, B>, A) -> B>(f: F) -> impl Fn(A) -> B {
    struct FixImpl<A, B, F: Fn(&dyn FnLike<A, B>, A) -> B>(F, std::marker::PhantomData<(A, B)>);

    impl<A, B, F: Fn(&dyn FnLike<A, B>, A) -> B> FnLike<A, B> for FixImpl<A, B, F> {
        fn call(&self, x: A) -> B {
            (self.0)(self, x)
        }
    }

    let fix = FixImpl(f, std::marker::PhantomData);
    move |x| fix.call(x)
}
