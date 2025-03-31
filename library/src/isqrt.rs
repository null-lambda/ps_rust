fn isqrt(x: u64) -> u64 {
    debug_assert!(x <= 4_503_599_761_588_223u64);
    (x as f64).sqrt() as u64
}
