mod mem_reserved {
    use std::mem::MaybeUninit;

    pub struct Stack<T> {
        pos: Box<[MaybeUninit<T>]>,
        len: usize,
    }

    impl<T> Stack<T> {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                pos: (0..capacity).map(|_| MaybeUninit::uninit()).collect(),
                len: 0,
            }
        }

        #[must_use]
        pub fn push(&mut self, value: T) -> Option<()> {
            if self.len == self.pos.len() {
                return None;
            }
            unsafe { self.push_unchecked(value) };
            Some(())
        }

        pub unsafe fn push_unchecked(&mut self, value: T) {
            *self.pos.get_unchecked_mut(self.len) = MaybeUninit::new(value);
            self.len += 1;
        }

        pub fn pop(&mut self) -> Option<T> {
            self.len = self.len.checked_sub(1)?;
            Some(unsafe { self.pos.get_unchecked(self.len).assume_init_read() })
        }
    }
}

mod mem_static {
    use std::mem::MaybeUninit;

    pub struct Stack<T, const N: usize> {
        pos: [MaybeUninit<T>; N],
        len: usize,
    }

    impl<T, const N: usize> Stack<T, N> {
        pub fn new() -> Self {
            Self {
                pos: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
            }
        }

        #[must_use]
        pub fn push(&mut self, value: T) -> Option<()> {
            if self.len == N {
                return None;
            }
            unsafe { self.push_unchecked(value) };
            Some(())
        }

        pub unsafe fn push_unchecked(&mut self, value: T) {
            *self.pos.get_unchecked_mut(self.len) = MaybeUninit::new(value);
            self.len += 1;
        }

        pub fn pop(&mut self) -> Option<T> {
            self.len = self.len.checked_sub(1)?;
            Some(unsafe { self.pos.get_unchecked(self.len).assume_init_read() })
        }
    }
}
