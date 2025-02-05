fn kmp<'a: 'c, 'b: 'c, 'c, T: PartialEq>(
    s: impl IntoIterator<Item = T> + 'a,
    pattern: &'b [T],
) -> impl Iterator<Item = usize> + 'c {
    // Build a jump table
    let mut jump_table = vec![0];
    let mut i_prev = 0;
    for i in 1..pattern.len() {
        while i_prev > 0 && pattern[i] != pattern[i_prev] {
            i_prev = jump_table[i_prev - 1];
        }
        if pattern[i] == pattern[i_prev] {
            i_prev += 1;
        }
        jump_table.push(i_prev);
    }

    // Search patterns
    let mut j = 0;
    s.into_iter().enumerate().filter_map(move |(i, c)| {
        while j == pattern.len() || j > 0 && pattern[j] != c {
            j = jump_table[j - 1];
        }
        if pattern[j] == c {
            j += 1;
        }
        (j == pattern.len()).then(|| i + 1 - pattern.len())
    })
}
