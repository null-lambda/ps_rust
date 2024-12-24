// Compute row minimum C(i) = min_j A(i, j)
// where opt(i) = argmin_j A(i, j) is monotonically increasing.
//
// Arguments:
// - naive: Compute opt(i) for a given range of j, with an implicit evaluation of C(i).
fn dnc_row_min(
    naive: &mut impl FnMut(usize, Range<usize>) -> usize,
    i: Range<usize>,
    j: Range<usize>,
) {
    if i.start >= i.end {
        return;
    }
    let i_mid = i.start + i.end >> 1;
    let j_opt = naive(i_mid, j.clone());
    dnc_row_min(naive, i.start..i_mid, j.start..j_opt + 1);
    dnc_row_min(naive, i_mid + 1..i.end, j_opt..j.end);
}
