fn isqrt(n: u64) -> u64 {
    if n <= 4_503_599_761_588_223u64 {
        return (n as f64).sqrt() as u64;
    }

    let mut x = n;
    loop {
        let next_x = (x + n / x) / 2;
        if next_x >= x {
            return x;
        }
        x = next_x;
    }
}
