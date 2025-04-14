pub mod rc_acyclic {
    // Shared rc pointers without weak references.
    use std::{cell::Cell, mem::MaybeUninit, ops::Deref, ptr::NonNull};

    #[allow(non_camel_case_types)]
    pub type ucount = u32;

    pub struct RcInner<T: ?Sized> {
        strong_count: Cell<ucount>,
        value: T,
    }

    pub struct Rc<T: ?Sized> {
        ptr: NonNull<RcInner<T>>,
        _marker: std::marker::PhantomData<Box<T>>,
    }

    impl<T> Rc<T> {
        #[inline]
        pub fn new(value: T) -> Self {
            let inner = RcInner {
                value,
                strong_count: Cell::new(1),
            };
            let inner = Box::new(inner);
            unsafe {
                Self {
                    ptr: NonNull::new_unchecked(Box::leak(inner)),
                    _marker: Default::default(),
                }
            }
        }

        #[inline]
        pub fn strong_count(&self) -> usize {
            unsafe { self.ptr.as_ref().strong_count.get() as usize }
        }
    }

    impl<T: ?Sized> Drop for Rc<T> {
        fn drop(&mut self) {
            let inner = unsafe { self.ptr.as_ref() };
            inner
                .strong_count
                .set(inner.strong_count.get().wrapping_sub(1));

            if inner.strong_count.get() == 0 {
                unsafe {
                    let _ = Box::from_raw(self.ptr.as_ptr());
                }
            }
        }
    }

    impl<T: ?Sized> Clone for Rc<T> {
        #[inline]
        fn clone(&self) -> Self {
            let inner = unsafe { self.ptr.as_ref() };
            inner
                .strong_count
                .set(inner.strong_count.get().wrapping_add(1));
            Self {
                ptr: self.ptr,
                _marker: Default::default(),
            }
        }
    }

    impl<T: ?Sized> Deref for Rc<T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &T {
            unsafe { &self.ptr.as_ref().value }
        }
    }

    impl<T: ?Sized + Clone> Rc<T> {
        pub fn make_mut(&mut self) -> &mut T {
            if Rc::strong_count(self) != 1 {
                let mut buffer: Box<MaybeUninit<RcInner<T>>> = Box::new(MaybeUninit::uninit());
                unsafe {
                    std::ptr::write(
                        buffer.as_mut_ptr(),
                        RcInner {
                            value: T::clone(self),
                            strong_count: Cell::new(1),
                        },
                    );
                    let ptr = Box::into_raw(buffer) as *mut RcInner<T>;
                    *self = Rc {
                        _marker: Default::default(),
                        ptr: NonNull::new_unchecked(ptr),
                    };
                }
            }
            &mut unsafe { self.ptr.as_mut() }.value
        }

        pub fn try_unwrap(self) -> Result<T, Self> {
            if Rc::strong_count(&self) == 1 {
                unsafe {
                    let inner = Box::from_raw(self.ptr.as_ptr());
                    std::mem::forget(self);
                    Ok(inner.value)
                }
            } else {
                Err(self)
            }
        }

        pub fn unwrap_or_clone(self) -> T {
            Self::try_unwrap(self).unwrap_or_else(|this| (*this).clone())
        }
    }

    impl<T: ?Sized + std::fmt::Display> std::fmt::Display for Rc<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            std::fmt::Display::fmt(&**self, f)
        }
    }

    impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for Rc<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            std::fmt::Debug::fmt(&**self, f)
        }
    }
}
