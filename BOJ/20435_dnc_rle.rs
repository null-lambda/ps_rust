use std::{
    cmp::Ordering::{Equal, Greater, Less},
    collections::{HashMap, HashSet},
    io::Write,
};

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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let k: usize = input.value();
    let n: usize = input.value();
    let _m: usize = input.value();
    let rle: Vec<(u32, u32)> = (0..n)
        .map(|_| {
            let count: u32 = input.value();
            let value: u32 = input.value();
            (value, count)
        })
        .collect();
    let rle = merge_rle(rle);

    fn merge_rle(rle: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
        let mut prev_value = 0;
        let mut count_acc = 0;
        let mut res = vec![];

        for (value, count) in rle {
            if value == prev_value {
                count_acc += count;
            } else {
                if count_acc > 0 {
                    res.push((prev_value, count_acc));
                }
                prev_value = value;
                count_acc = count;
            }
        }
        if count_acc > 0 {
            res.push((prev_value, count_acc));
        }

        res
    }

    fn split_rle_at(rle: Vec<(u32, u32)>, idx: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
        let mut len_acc = 0;
        let mut rle = rle.into_iter();
        let mut left = vec![];
        let mut right = vec![];
        while let Some((value, count)) = rle.next() {
            match (len_acc + count).cmp(&idx) {
                Less => {
                    left.push((value, count));
                    len_acc += count;
                }
                Equal => {
                    left.push((value, count));
                    break;
                }
                Greater => {
                    left.push((value, idx - len_acc));
                    right.push((value, count - (idx - len_acc)));
                    break;
                }
            }
        }
        right.extend(rle);
        (left, right)
    }

    fn dnc(rle: Vec<(u32, u32)>, size: u32) -> (HashMap<u32, u32>, u32) {
        assert_eq!(size, rle.iter().map(|(_, count)| count).sum());
        assert!(rle.iter().all(|(_, count)| *count > 0));
        assert!(size > 0);
        if rle.len() == 1 {
            let (value, _) = rle[0];
            return ([(value, 0)].into(), 1);
        }
        assert!(size >= 2);

        let mid = size / 2;
        let (left, right) = split_rle_at(rle, size / 2);
        let ((left, left_full), (right, right_full)) = (dnc(left, mid), dnc(right, size - mid));

        let mut full_cost = left_full + right_full;
        let mut cost_without_single_value = HashMap::new();
        for &value_omitted in left.keys().chain(right.keys()).collect::<HashSet<_>>() {
            let left_cost = left
                .get(&value_omitted)
                .copied()
                .map_or(left_full, |x| left_full.min(x));
            let right_cost = right
                .get(&value_omitted)
                .copied()
                .map_or(right_full, |x| right_full.min(x));
            cost_without_single_value.insert(value_omitted, left_cost + right_cost);
        }

        full_cost = full_cost.min(cost_without_single_value.values().min().copied().unwrap() + 1);
        (cost_without_single_value, full_cost)
    }

    let (_, min_cost) = dnc(rle, 1 << k);
    writeln!(output, "{}", min_cost).unwrap();
}
