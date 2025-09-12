pub mod linalg_gf2 {
    pub type T = u64;

    pub fn xor_basis(rows: &mut Vec<T>, mut select: impl FnMut(usize)) {
        let mut rank = 0;
        for i in 0..rows.len() {
            let mut x = rows[i];
            for j in 0..rank {
                x = x.min(x ^ rows[j]);
            }
            if x != 0 {
                rows[rank] = x;
                select(i);
                rank += 1;
            }
        }

        rows.truncate(rank);
    }

    pub fn bit_reversed_rref(rows: &mut Vec<T>) {
        xor_basis(rows, |_| {});
        rows.sort_unstable_by_key(|&x| std::cmp::Reverse(x));

        for i in 0..rows.len() {
            for p in 0..i {
                rows[p] = rows[p].min(rows[p] ^ rows[i]);
            }
        }
    }
}
