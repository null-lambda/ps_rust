use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let n: usize = input.value();

    let xs_orig: Vec<i32> = (0..2 * n).map(|_| input.value()).collect();
    let mut xs: Vec<(i32, u32)> = (0..2 * n).map(|i| (xs_orig[i], i as u32)).collect();
    xs.sort_unstable();

    let dummy: u32 = 2 * n as u32;
    const UNSET: u32 = std::u32::MAX;
    let mut next = vec![dummy; 2 * n + 1];
    let mut prev = vec![dummy; 2 * n + 1];
    let mut pq: BinaryHeap<Reverse<(i32, u32, u32)>> = BinaryHeap::new();

    next[dummy as usize] = xs[0].1;
    prev[dummy as usize] = xs[2 * n - 1].1;
    for t in xs.windows(2) {
        let ((xi, i), (xj, j)) = (t[0], t[1]);
        next[i as usize] = j;
        prev[j as usize] = i;
        pq.push(Reverse((xj - xi, i, j)));
    }

    let mut paired_to: Vec<u32> = vec![UNSET; n * 2];
    while let Some(Reverse((_d, i, j))) = pq.pop() {
        if (i < n as u32) == (j < n as u32) {
            continue;
        } else if paired_to[i as usize] == UNSET && paired_to[j as usize] == UNSET {
            paired_to[i as usize] = j;
            paired_to[j as usize] = i;
            prev[next[j as usize] as usize] = prev[i as usize];
            next[prev[i as usize] as usize] = next[j as usize];

            if prev[i as usize] != dummy && next[j as usize] != dummy {
                let d =
                    (xs_orig[next[j as usize] as usize] - xs_orig[prev[i as usize] as usize]).abs();
                pq.push(Reverse((d, prev[i as usize], next[j as usize])));
            }
        } else {
            continue;
        }

        // pop link i-j from the link
        //
    }

    if paired_to.iter().any(|&x| x == UNSET) {
        writeln!(output, "-1").unwrap();
        return;
    }
    for i in 0..n {
        writeln!(output, "{} {}", i + 1, paired_to[i] % n as u32 + 1).unwrap();
    }
}
