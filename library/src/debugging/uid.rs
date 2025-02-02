pub mod debug {
    // Used for debugging node topology in graphs and Trees.
    #[cfg(debug_assertions)]
    type UID = u32; // Unique id in debug mode

    #[cfg(not(debug_assertions))]
    type UID = (); // No-op in release mode

    fn gen_tag() -> UID {
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
}
