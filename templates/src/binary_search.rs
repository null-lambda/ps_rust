use std::cmp::Ordering;

// adoption from std crate
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

// adoption from std crate
fn partition_point<P>(left: u32, right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    binary_search_by(left, right, |x| {
        if pred(x) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    })
    .unwrap_or_else(|i| i)
}
