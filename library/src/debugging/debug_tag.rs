// Used for debugging node topology in graphs and Trees.
#[cfg(debug_assertions)]
type Tag = u32; // Unique id in debug mode

#[cfg(not(debug_assertions))]
type Tag = (); // No-op in release mode

fn gen_tag() -> Tag {
    #[cfg(debug_assertions)]
    {
        use std::cell::Cell;
        thread_local! {
            static COUNTER: Cell<u32> = Default::default();
        }
        return COUNTER.with(|counter| {
            let idx = counter.get();
            counter.set(idx + 1);
            idx
        });
    }
}

#[derive(Default, Debug)]
pub struct Node {
    is_left: bool,
    data: UnclosedParens,

    tag: Tag,

    size: u32,
    link: Link,
}
