use std::io::Write;

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

mod unsafe_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::mem::ManuallyDrop;
    use std::os::unix::io::FromRawFd;

    use crate::branch;

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    extern "C" {
        fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }

    const BUFFER_SIZE: usize = 1 << 18;

    pub struct UnsafeIntScanner {
        buffer_start: *mut u8,
        buffer_end: *mut u8,
        cursor: *mut u8,
    }

    impl UnsafeIntScanner {
        unsafe fn refill(&mut self) {
            if branch::unlikely(self.cursor >= self.buffer_end.offset(-64)) {
                let mut dest = self.buffer_start;
                while self.cursor != self.buffer_end {
                    *dest = *self.cursor;
                    dest = dest.offset(1);
                    self.cursor = self.cursor.offset(1);
                }
                let len = read(0, dest, self.buffer_end.offset_from(dest) as usize);
                if len <= 0 {
                    panic!();
                }
                *dest.offset(len as isize) = 0;
                self.cursor = self.buffer_start;
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn u32_simd(&mut self) -> u32 {
            let block = _mm_loadu_si128(self.cursor as *const i8 as *const __m128i);
            let mask_lower = _mm_cmpgt_epi8(_mm_set1_epi8(b'0' as i8), block);
            // let mask_upper = _mm_cmplt_epi8(_mm_set1_epi8(b'9' as i8 + 1), block);
            // let mask = _mm_or_si128(mask_lower, mask_upper);
            let mask = mask_lower;

            let len = _mm_movemask_epi8(mask).trailing_zeros();
            self.cursor = self.cursor.offset(len as isize);

            let mut block = _mm_loadu_si128(self.cursor.offset(-16) as *const i8 as *const __m128i);
            block = _mm_max_epi8(block, _mm_set1_epi8(b'0' as i8));
            block = _mm_sub_epi8(block, _mm_set1_epi8(b'0' as i8));

            let mut mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            mask = _mm_cmpgt_epi8(_mm_set1_epi8(len as i8), mask);
            block = _mm_and_si128(block, mask);

            let mul10 = _mm_set_epi8(1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10);
            let mul100 = _mm_set_epi16(1, 100, 1, 100, 1, 100, 1, 100);
            let mul10000 = _mm_set_epi16(0, 0, 0, 0, 1, 10000, 1, 10000);
            let mut acc = _mm_maddubs_epi16(block, mul10);
            acc = _mm_madd_epi16(acc, mul100);
            acc = _mm_packus_epi32(acc, acc);
            acc = _mm_madd_epi16(acc, mul10000);

            let mut recovered = [0u32; 4];
            _mm_storeu_si128(recovered.as_mut_ptr() as *mut __m128i, acc);
            let x = recovered[1] as u32;

            self.cursor = self.cursor.offset(1);
            x
        }

        pub fn u32(&mut self) -> u32 {
            unsafe {
                self.refill();
                self.u32_simd()

                // let mut x = 0u32;
                // while *self.cursor < b'0' {
                //     self.cursor = self.cursor.offset(1);
                // }
                // while *self.cursor >= b'0' {
                //     x = x * 10 + (*self.cursor - b'0') as u32;
                //     self.cursor = self.cursor.offset(1);
                // }
                // self.cursor = self.cursor.offset(1);
                // x
            }
        }
    }

    pub fn stdin_int() -> UnsafeIntScanner {
        let buf = Vec::<u8>::with_capacity(BUFFER_SIZE);
        let mut buf = ManuallyDrop::new(buf);
        let capacity = buf.capacity();
        let buf = buf.as_mut_ptr();
        let buffer_start = unsafe { buf.offset(64) };
        let buffer_end = unsafe { buf.offset(capacity as isize) };

        UnsafeIntScanner {
            buffer_start,
            buffer_end,
            cursor: buffer_end,
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn write_num(w: &mut Vec<u8>, c: u32) {
    if c / 10 > 0 {
        write_num(w, c / 10);
    }
    w.push((c % 10) as u8 + b'0');
}

fn main() {
    let mut input = unsafe_io::stdin_int();
    let mut output = unsafe_io::stdout();
    let mut buf = Vec::new();

    for _ in 0..input.u32() {
        let a: u32 = input.u32();
        let b: u32 = input.u32();
        unsafe { write_num(&mut buf, a + b) };
        buf.push(b'\n');
    }

    output.write_all(&buf).unwrap();
}
