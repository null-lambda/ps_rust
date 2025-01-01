#[macro_use]
mod mem_static {
    /// Note: If you can apply maximum optimization flags to the compiler (e.g. opt-level=3),
    /// prefer using vectors or stack-allocated arrays instead of static memory allocations.
    ///
    /// A wrapper around a statically allocated array (`static mut [T; N]`), providing a static bump allocator.
    /// This provides a safer interface than raw `static mut`, but does not guarantee UB-freeness.
    /// Use at your own risk.
    ///
    /// # Example
    ///
    /// ```rust
    /// const N_MAX: usize = 100_000;
    /// static_bump!(ZSTBump: [(); 1000]);
    /// static_bump!(U32Bump: [u32; N_MAX + 1]);
    /// let slice: &mut [u32] = U32Bump::take_slice_with(n, |i| (i as u32)).unwrap();
    /// ```
    use core::mem::MaybeUninit;

    pub trait StaticBump<T> {
        fn take_slice_uninit(n: usize) -> Option<&'static mut [MaybeUninit<T>]>;
        fn take_slice_with(n: usize, mut f: impl FnMut(usize) -> T) -> Option<&'static mut [T]> {
            let slice = Self::take_slice_uninit(n)?;
            for i in 0..n {
                slice[i] = core::mem::MaybeUninit::new(f(i));
            }
            let slice = unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, n) };
            Some(slice)
        }
    }

    pub struct StaticBumpImpl<const N: usize, T> {
        buffer: [MaybeUninit<T>; N],
        cursor: usize,
    }

    impl<const N: usize, T> StaticBumpImpl<N, T> {
        pub const fn new() -> Self {
            Self {
                // buffer: [const { MaybeUninit::uninit() }; N], // rustc >= 1.79
                buffer: unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() },
                cursor: 0,
            }
        }

        pub fn take_slice_uninit(
            &'static mut self,
            n: usize,
        ) -> Option<&'static mut [MaybeUninit<T>]> {
            let cursor = self.cursor;
            self.cursor += n;
            if self.cursor > N {
                return None;
            }
            let buffer = self.buffer.as_mut_ptr();
            let slice = unsafe { core::ptr::slice_from_raw_parts_mut(buffer.add(cursor), n) };
            unsafe { slice.as_mut() }
        }
    }

    #[macro_export]
    macro_rules! static_bump {
        ($name:ident: [$type:ty; $buf_size:expr]) => {
            struct $name(core::marker::PhantomData<&'static mut mem_static::StaticBumpImpl<$buf_size, $type>>);

            impl $name {
                pub fn __get_instance() -> *mut crate::mem_static::StaticBumpImpl<$buf_size, $type> {
                    thread_local! {
                        static INSTANCE: core::cell::UnsafeCell<crate::mem_static::StaticBumpImpl<$buf_size, $type>> = const {
                            core::cell::UnsafeCell::new(crate::mem_static::StaticBumpImpl::new())
                        };
                    }
                    INSTANCE.with(|instance| instance.get())
                }
           }

            impl crate::mem_static::StaticBump<$type> for $name {
                fn take_slice_uninit(n: usize) -> Option<&'static mut [core::mem::MaybeUninit<$type>]>
                {
                    let instance = Self::__get_instance();
                    unsafe { (*instance).take_slice_uninit(n) }
                }
            }
        }
    }
}
