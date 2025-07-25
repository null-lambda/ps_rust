fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

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

// Find (d, x, y) satisfying d = gcd(abs(a), abs(b)) and a * x + b * y = d
fn egcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    let (d, x, y) = egcd(a.abs() as u64, b.abs() as u64);
    (d as i64, x as i64 * a.signum(), y as i64 * b.signum())
}

fn crt(a1: u64, m1: u64, a2: u64, m2: u64) -> Option<(u64, u64)> {
    let (d, x, _y) = egcd(m1, m2);
    let m = m1 / d * m2;
    let da = ((a2 as i64 - a1 as i64) % m as i64 + m as i64) as u64 % m;
    if da % d != 0 {
        return None;
    }
    let mut x = ((x % m as i64) + m as i64) as u64 % m;
    x = (da / d % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}
