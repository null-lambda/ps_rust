pub mod simd {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn prefix_sum_exact_i32(xs: &mut [i32]) {
        assert_eq!(xs.len() % 4, 0);

        let mut carry = _mm_set1_epi32(0);
        for block in xs.chunks_exact_mut(4) {
            let mut x = _mm_loadu_si128(block.as_mut_ptr() as *const __m128i); // x=[a,b,c,d]

            x = _mm_add_epi32(x, _mm_slli_si128(x, 4)); // ... + [0,a,b,c] = [a,a+b,b+c,c+d]
            x = _mm_add_epi32(x, _mm_slli_si128(x, 8)); // ... + [0,0,a,a+b] = [a,a+b,a+b+c,b+c+d]
            x = _mm_add_epi32(carry, x); // add carry from the previous block

            carry = _mm_set1_epi32(_mm_extract_epi32(x, 3)); // carry = [b+c+d,b+c+d,b+c+d,b+c+d]

            _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, x);
        }
    }
}
