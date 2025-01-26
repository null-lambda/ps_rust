use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

#[derive(Clone, Debug)]
struct CardSet {
    w_max: u32,
    h_max: u32,
    count: u32,
    total_area: u64,
}

impl CardSet {
    fn new(w_max: u32, h_max: u32, count: u32) -> Self {
        Self {
            w_max,
            h_max,
            count,
            total_area: w_max as u64 * h_max as u64 * count as u64,
        }
    }

    fn union_with(&mut self, other: &Self) {
        self.w_max = self.w_max.max(other.w_max);
        self.h_max = self.h_max.max(other.h_max);
        self.count += other.count;
        self.total_area += other.total_area;
    }

    fn wasted_area(&self) -> u64 {
        self.w_max as u64 * self.h_max as u64 * self.count as u64 - self.total_area
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let mut cards = vec![];
    for _ in 0..n {
        let w: u32 = input.value();
        let h: u32 = input.value();
        let c: u32 = input.value();
        cards.push(CardSet::new(w, h, c));
    }

    let mut sums = vec![CardSet::new(0, 0, 0); 1 << n];
    for mask in 1usize..1 << n {
        let lsb = mask.trailing_zeros() as usize;
        sums[mask] = sums[mask ^ (1 << lsb)].clone();
        sums[mask].union_with(&cards[lsb]);
    }

    let costs_base = sums
        .into_iter()
        .map(|set| set.wasted_area())
        .collect::<Vec<_>>();
    let mut costs = costs_base.clone();
    for _ in 0..k - 1 {
        let prev = costs.clone();
        for mask in 0..1 << n {
            let mut submask = mask;
            while submask > 0 {
                costs[mask] = costs[mask].min(prev[submask] + costs_base[mask ^ submask]);
                submask = (submask - 1) & mask;
            }
        }
    }

    let ans = costs[(1 << n) - 1];
    writeln!(output, "{}", ans).unwrap();
}
