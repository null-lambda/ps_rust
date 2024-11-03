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
