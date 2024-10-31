use std::cmp::Ordering;

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
