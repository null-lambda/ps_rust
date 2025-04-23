fn next_permutation<T: Ord>(arr: &mut [T]) -> bool {
    match arr.windows(2).rposition(|w| w[0] < w[1]) {
        Some(i) => {
            let j = i + arr[i + 1..].partition_point(|x| &arr[i] < x);
            arr.swap(i, j);
            arr[i + 1..].reverse();
            true
        }
        None => {
            arr.reverse();
            false
        }
    }
}

fn combinations_u32(n: usize, r: usize) -> impl Iterator<Item = u32> {
    assert!(n <= 31);
    std::iter::successors(Some((1u32 << r) - 1), move |&mask| {
        let mut next = mask;
        let lsb = next & next.wrapping_neg();
        let r = next + lsb;
        if r >= 1 << n {
            return None;
        }
        next = (((next ^ r) >> 2) / lsb) | r;
        Some(next)
    })
}

fn nonzero_bits_u32(n: u32) -> impl Iterator<Item = u32> {
    std::iter::successors(Some(n), |&mask| Some(mask & (mask - 1)))
        .take_while(|&mask| mask != 0)
        .map(|mask| u32::BITS - 1 - (mask & mask.wrapping_neg()).leading_zeros())
}
