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
