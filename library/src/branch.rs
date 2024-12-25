pub mod branch {
    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[cold]
    #[inline(always)]
    pub fn cold() {}

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        if !b {
            cold();
        }
        b
    }

    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        if b {
            cold();
        }
        b
    }
}
