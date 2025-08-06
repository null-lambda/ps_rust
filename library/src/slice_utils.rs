fn group_indices_by<'a, T>(
    xs: &'a [T],
    mut pred: impl 'a + FnMut(&T, &T) -> bool,
) -> impl 'a + Iterator<Item = [usize; 2]> {
    let mut i = 0;
    std::iter::from_fn(move || {
        if i == xs.len() {
            return None;
        }

        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        let res = [i, j];
        i = j;
        Some(res)
    })
}

fn group_by<'a, T>(
    xs: &'a [T],
    pred: impl 'a + FnMut(&T, &T) -> bool,
) -> impl 'a + Iterator<Item = &'a [T]> {
    group_indices_by(xs, pred).map(|w| &xs[w[0]..w[1]])
}

fn group_by_key<'a, T, K: PartialEq>(
    xs: &'a [T],
    mut key: impl 'a + FnMut(&T) -> K,
) -> impl 'a + Iterator<Item = &'a [T]> {
    group_by(xs, move |a, b| key(a) == key(b))
}

fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let n = xs.len();
    let mut i = 0;
    for j in 0..n {
        if pred(&xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    xs.split_at_mut(i)
}
