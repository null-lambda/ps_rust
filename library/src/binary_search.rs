use std::cmp::Ordering;

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn partition_point_f64<P>(
    mut left: f64,
    mut right: f64,
    eps: f64,
    mut max_iter: u32,
    mut pred: P,
) -> f64
where
    P: FnMut(f64) -> bool,
{
    while right - left > eps && max_iter > 0 {
        let mid = left + (right - left) / 2.0;
        if pred(mid) {
            left = mid;
        } else {
            right = mid;
        }
        max_iter -= 1;
    }
    left
}

fn binary_search_by<F>(mut left: u32, mut right: u32, mut f: F) -> Result<u32, u32>
where
    F: FnMut(u32) -> Ordering,
{
    let mut size;
    while left < right {
        size = right - left;
        let mid = left + size / 2;

        let cmp = f(mid);
        match cmp {
            Ordering::Less => {
                left = mid + 1;
            }
            Ordering::Greater => {
                right = mid;
            }
            Ordering::Equal => {
                return Ok(mid);
            }
        }
    }
    Err(left)
}

fn ternary_search<F, K>(mut left: i64, mut right: i64, mut f: F) -> i64
where
    K: Ord,
    F: FnMut(&i64) -> K,
{
    while right - left > 3 {
        let m1 = left + (right - left) / 3;
        let m2 = right - (right - left) / 3;
        if f(&m1) <= f(&m2) {
            right = m2;
        } else {
            left = m1;
        }
    }
    (left..=right).min_by_key(|&x| f(&x)).unwrap()
}
