fn gen_factorials<T: algebra::Field + Clone + From<u32> + std::fmt::Debug>(
    n_bound: u32,
) -> (Vec<T>, Vec<T>) {
    assert!(n_bound >= 1);

    let mut fac = vec![T::one()];
    for i in 1..=n_bound {
        fac.push(fac[i as usize - 1].clone() * T::from(i));
    }

    let mut ifac = vec![T::one(); n_bound as usize + 1];
    ifac[n_bound as usize] = fac[n_bound as usize].inv();
    for i in (2..=n_bound).rev() {
        ifac[i as usize - 1] = ifac[i as usize].clone() * T::from(i);
    }

    (fac, ifac)
}

fn comb_lucas<'a, T: algebra::Field + Clone + std::fmt::Debug>(
    p: u64,
    fac_mod_p: &'a [T],
    ifac_mod_p: &'a [T],
) -> impl 'a + Fn(u64, u64) -> T {
    move |mut m: u64, mut n: u64| {
        let mut acc = T::one();
        while m > 0 {
            let x = m % p;
            let y = n % p;
            if x < y {
                return T::zero();
            }
            acc *= fac_mod_p[x as usize].clone()
                * &ifac_mod_p[y as usize]
                * &ifac_mod_p[(x - y) as usize];

            m /= p;
            n /= p;
        }
        acc
    }
}
