fn guess_frac<M: Field + Eq + Clone + From<u32> + Into<u32>>(bound: u32) -> impl Fn(M) -> String {
    use std::collections::HashMap;

    let mut table = HashMap::<u32, String>::new();

    table.insert(0, "0".into());
    for p in 1..=bound {
        for q in 1..=bound {
            let r = M::from(p).clone() / M::from(q);
            table
                .entry(r.clone().into())
                .or_insert_with(|| format!("{p}/{q}"));
            table
                .entry((-r).into())
                .or_insert_with(|| format!("-{p}/{q}"));
        }
    }

    move |x: M| {
        table
            .get(&(x.into()))
            .cloned()
            .unwrap_or_else(|| "?".into())
    }
}
