fn isqrt(x: i64) -> i64 {
    debug_assert!(0 <= x && x <= 4_503_599_761_588_223i64.pow(2));
    (x as f64).sqrt() as i64
}
