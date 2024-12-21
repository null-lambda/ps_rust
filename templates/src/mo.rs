pub mod mo {
    pub fn even_odd_order(n: usize) -> impl Fn(u32, u32) -> (u32, i32) {
        assert!(n > 0);
        let bucket_size = (n as f64).sqrt() as u32;
        move |l, r| {
            let k = l / bucket_size;
            let l = if k % 2 == 0 { r as i32 } else { -(r as i32) };
            (k, l)
        }
    }

    // Mo's algorithm with space filling curve
    // https://codeforces.com/blog/entry/61203
    // https://codeforces.com/blog/entry/115590
    // Note: use sort_with_cached_key instead of sort_unstable
    //       to avoid recomputing the Hilbert order.
    pub fn hilbert_order(n: usize) -> impl Fn(u32, u32) -> i64 {
        assert!(n > 0);
        let log2n_ceil = usize::BITS - 1 - n.next_power_of_two().leading_zeros();

        fn inner(mut x: u32, mut y: u32, mut exp: u32) -> i64 {
            let mut res = 0;
            let mut sign = 1;
            let mut rot = 0;

            while exp > 0 {
                let w_half = 1 << exp - 1;
                let quadrant = match (x < w_half, y < w_half) {
                    (true, true) => (rot + 0) % 4,
                    (false, true) => (rot + 1) % 4,
                    (false, false) => (rot + 2) % 4,
                    (true, false) => (rot + 3) % 4,
                };
                rot = match quadrant {
                    0 => (rot + 3) % 4,
                    1 => (rot + 0) % 4,
                    2 => (rot + 0) % 4,
                    3 => (rot + 1) % 4,
                    _ => unsafe { std::hint::unreachable_unchecked() },
                };

                x &= !w_half;
                y &= !w_half;

                let square_area_half = 1 << 2 * exp - 2;
                res += sign * quadrant as i64 * square_area_half;
                if quadrant == 0 || quadrant == 3 {
                    res += sign * (square_area_half - 1);
                    sign = -sign;
                };

                exp -= 1;
            }
            res
        }

        move |l, r| {
            debug_assert!(l < n as u32);
            debug_assert!(r < n as u32);
            inner(l, r, log2n_ceil as u32)
        }
    }
}
