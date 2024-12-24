const P: u64 = 1_000_000_007;

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: u64) -> u64 {
    pow(n, P - 2)
}

fn crt(a1: u64, m1: u64, a2: u64, m2: u64) -> Option<(u64, u64)> {
    let (d, x, _y) = egcd(m1, m2);
    debug_assert!(d == 1);
    let m = m1 * m2;
    let a2_m_a1 = ((a2 as i64 - a1 as i64) % m as i64 + m as i64) as u64 % m;
    if a2_m_a1 % d != 0 {
        return None;
    }
    let mut x = ((x % m as i64) + m as i64) as u64 % m;
    x = (a2_m_a1 / d % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}
