// Manacher's algorithm
fn palindrome_radius<T: Eq>(s: &[T]) -> Vec<usize> {
    let n = s.len();
    let mut i = 0;
    let mut radius = 0;
    let mut rs = vec![];
    while i < n {
        while i >= (radius + 1)
            && i + (radius + 1) < n
            && s[i - (radius + 1)] == s[i + (radius + 1)]
        {
            radius += 1;
        }
        rs.push(radius);

        let mut mirrored_center = i;
        let mut max_mirrored_radius = radius;
        i += 1;
        radius = 0;
        while max_mirrored_radius > 0 {
            mirrored_center -= 1;
            max_mirrored_radius -= 1;
            if rs[mirrored_center] == max_mirrored_radius {
                radius = max_mirrored_radius;
                break;
            }
            rs.push(rs[mirrored_center].min(max_mirrored_radius));
            i += 1;
        }
    }
    rs
}
