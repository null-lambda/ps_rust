use std::io::Write;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

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

fn group_indices_by_key<'a, T, K: PartialEq>(
    xs: &'a [T],
    mut key: impl 'a + FnMut(&T) -> K,
) -> impl 'a + Iterator<Item = [usize; 2]> {
    group_indices_by(xs, move |a, b| key(a) == key(b))
}

fn group_indices<'a, T: PartialEq>(xs: &'a [T]) -> impl 'a + Iterator<Item = [usize; 2]> {
    group_indices_by(xs, |x, y| x == y)
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

fn groups<'a, T: PartialEq>(xs: &'a [T]) -> impl 'a + Iterator<Item = &'a [T]> {
    group_by(xs, |x, y| x == y)
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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut s = input.token().as_bytes().to_vec();

    let mut ans = s.len();
    {
        let mut it = group_indices(&s).enumerate();
        while let Some((k, [i, j])) = it.next() {
            if j - i >= 2 && k >= 1 {
                drop(it);
                s.truncate(i + 1);
                break;
            }
        }
    }

    s.dedup();
    ans = ans.min(s.len());

    writeln!(output, "{}", ans).unwrap();
}
