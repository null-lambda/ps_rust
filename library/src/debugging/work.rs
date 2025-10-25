pub mod debug {
    use std::cell::Cell;

    thread_local! {
        static WORK: Cell<u64> = Cell::new(0);
    }

    pub fn work() {
        WORK.with(|work| work.set(work.get() + 1));
    }

    pub fn get_work() -> u64 {
        WORK.with(|work| work.get())
    }
}
