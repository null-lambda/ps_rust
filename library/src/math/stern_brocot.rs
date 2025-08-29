mod stern_brocot {
    pub type T = u32;

    fn partition_point_unbounded(mut pred: impl FnMut(T) -> bool) -> T {
        // Exponential search
        let mut left = 0;
        let mut right = 0;
        while pred(right) {
            left = right;
            right = right * 2 + 1;
            if right == T::MAX {
                return right;
            }
        }

        // Binary search
        while left < right {
            let mid = left + (right - left) / 2;
            if pred(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }

    // Let Q_N = { p/q: 1 <= p, 1 <= q <= N },
    //     P = { r in Q_N : pred(r) }.
    //
    // Find max (S U { 0/1 }) and min ((Q_N \ S) U { 1/0 }).
    pub fn partition_point(max_denom: T, mut pred: impl FnMut((T, T)) -> bool) -> [(T, T); 2] {
        assert!(max_denom >= 1);
        let mut l = (0, 1);
        let mut r = (1, 0);
        let mut m = (l.0 + r.0, l.1 + r.1);

        loop {
            let k = partition_point_unbounded(|k| {
                m.1 + l.1 * k <= max_denom && !pred((m.0 + l.0 * k, m.1 + l.1 * k))
            });
            if k > 0 {
                r = (m.0 + l.0 * (k - 1), m.1 + l.1 * (k - 1));
                m = (l.0 + r.0, l.1 + r.1);
                if m.1 > max_denom {
                    break;
                }
            }

            let k = partition_point_unbounded(|k| {
                m.1 + r.1 * (k + 1) <= max_denom && pred((m.0 + r.0 * (k + 1), m.1 + r.1 * (k + 1)))
            });
            l = (m.0 + r.0 * k, m.1 + r.1 * k);
            m = (l.0 + r.0, l.1 + r.1);
            if m.1 > max_denom {
                break;
            }
        }

        [l, r]
    }

    pub fn find_path_rle(mut x: (T, T)) -> Vec<(bool, T)> {
        assert!(x.0 != 0 && x.1 != 0);

        let mut is_right = true;
        let mut path = vec![];
        while x.1 != 0 {
            let q = x.0 / x.1;
            if q != 0 {
                path.push((is_right, q));
            }
            x = (x.1, x.0 % x.1);
            is_right ^= true;
        }

        let tail = &mut path.last_mut().unwrap().1;
        *tail -= 1;
        if *tail == 0 {
            path.pop();
        }

        path
    }

    pub fn locate_by_rle(path: impl IntoIterator<Item = (bool, T)>) -> (T, T) {
        let mut l = (0, 1);
        let mut r = (1, 0);
        let mut m = (l.0 + r.0, l.1 + r.1);

        for (is_right, len) in path {
            if !is_right {
                r = (m.0 + l.0 * (len - 1), m.1 + l.1 * (len - 1));
            } else {
                l = (m.0 + r.0 * (len - 1), m.1 + r.1 * (len - 1));
            }
            m = (l.0 + r.0, l.1 + r.1);
        }
        m
    }
}
