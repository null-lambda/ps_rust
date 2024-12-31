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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let cost: Vec<u64> = (0..n).map(|_| input.value()).collect();
    let value: Vec<u64> = (0..n).map(|_| input.value()).collect();
    let k: u64 = input.value();

    let n_owned: usize = input.value();
    let mut base_budget = 0;
    for _ in 0..n_owned {
        base_budget += cost[input.value::<usize>()];
    }

    // Meet in the middle
    let (cost_left, cost_right) = cost.split_at(n / 2);
    let (value_left, value_right) = value.split_at(n / 2);

    fn gen_comb<'a>(cost: &'a [u64], value: &'a [u64]) -> impl Iterator<Item = (u64, u64)> + 'a {
        (0..1 << cost.len()).map(|state| {
            let mut c = 0;
            let mut v = 0;
            for j in 0..cost.len() {
                if (state >> j) & 1 == 1 {
                    c += cost[j];
                    v += value[j];
                }
            }
            (v, c)
        })
    }

    let comb_left = gen_comb(cost_left, value_left);
    let mut comb_right = gen_comb(cost_right, value_right).collect::<Vec<_>>();
    comb_right.sort_unstable_by_key(|&(v, c)| (c, v));

    let mut res = vec![(0, 0)];
    let mut prev_value = 0;
    for (v, c) in comb_right {
        if v > prev_value {
            res.push((v, c));
            prev_value = v;
        }
    }
    comb_right = res;

    // Maximize sum of cost, subject to the constraint that sum of value >= k
    let mut min_cost = u64::MAX;
    for (v1, c1) in comb_left {
        let i = comb_right.partition_point(|(v2, _)| v1 + v2 < k);
        if i == comb_right.len() {
            continue;
        }

        let (_, c2) = comb_right[i];
        min_cost = min_cost.min(c1 + c2);
    }
    if min_cost == u64::MAX {
        writeln!(output, "-1").unwrap();
        return;
    }

    writeln!(output, "{}", min_cost.max(base_budget) - base_budget).unwrap();
}
