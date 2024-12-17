// Y combinator for recursive closures
fn fix<A, B>(f: impl Fn(&dyn Fn(A) -> B, A) -> B) -> impl Fn(A) -> B {
    move |x: A| f(&|arg| fix(&f)(arg), x)
}

fn fix_mut<A, B>(f: Rc<RefCell<impl FnMut(&mut dyn FnMut(A) -> B, A) -> B>>) -> impl FnMut(A) -> B {
    move |x: A| {
        let mut g = fix_mut(Rc::clone(&f));
        (*f).borrow_mut()(&mut g, x)
    }
}
