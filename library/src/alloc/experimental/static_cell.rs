#[macro_use]
mod mem_static {
    /// Note: If you can apply maximum optimization flags to the compiler (e.g. opt-level=3),
    /// prefer using vectors or stack-allocated arrays instead of static memory allocations.
    ///
    /// A convenient wrapper for static allocation.
    /// Provides the largest performance boost in cases of heavy pointer chasing.
    use core::{
        cell::UnsafeCell,
        sync::atomic::{AtomicBool, Ordering},
    };

    pub struct UnsafeStaticCell<T> {
        value: UnsafeCell<T>,
        lock: AtomicBool,
    }

    impl<T> UnsafeStaticCell<T> {
        pub const fn new(value: T) -> Self {
            Self {
                value: UnsafeCell::new(value),
                lock: AtomicBool::new(false),
            }
        }

        pub unsafe fn lock(&self) -> Option<&mut T> {
            match self
                .lock
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            {
                Ok(_) => Some(unsafe { &mut *self.value.get() }),
                Err(_) => None,
            }
        }

        pub unsafe fn unlock(&self) {
            self.lock.store(false, Ordering::Release);
        }
    }

    unsafe impl<T> Sync for UnsafeStaticCell<T> {}
    unsafe impl<T> Send for UnsafeStaticCell<T> {}

    macro_rules! read_once_static {
        ($ty:ty, $value:expr) => {
            #[allow(unused_unsafe)]
            unsafe {
                static INSTANCE: crate::mem_static::UnsafeStaticCell<$ty> =
                    crate::mem_static::UnsafeStaticCell::new($value);
                INSTANCE.lock().unwrap()
            }
        };
    }
}
