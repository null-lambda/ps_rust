fn yesno(b: bool) -> &'static str {
    if b { "Yes" } else { "No" }
}

fn join<'a>(iter: impl 'a + IntoIterator<Item = impl std::fmt::Display>) -> String {
    use std::fmt::Write;
    let mut iter = iter.into_iter();
    let mut res = String::new();
    if let Some(x) = iter.next() {
        write!(&mut res, "{x}").unwrap();
        for x in iter {
            write!(&mut res, " {x}").unwrap();
        }
    }
    res
}
