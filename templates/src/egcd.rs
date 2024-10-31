// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: u64, b: u64) -> (u64, i64, i64) {
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - (q as i64) * x.1));
        y = (y.1, (y.0 - (q as i64) * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}