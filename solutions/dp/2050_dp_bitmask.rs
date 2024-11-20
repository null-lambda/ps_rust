use std::{io::Write, str};

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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)>
where
    I: IntoIterator,
    I::Item: Clone,
    J: IntoIterator,
    J::IntoIter: Clone,
{
    let j = j.into_iter();
    i.into_iter()
        .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    for _ in 0..input.value() {
        let n: usize = input.value();
        let num_max = 100;
        let mut tiles = vec![[false; 4]; num_max + 1];
        for _ in 0..n {
            let t = input.token();
            let (&color, rest) = t.as_bytes().split_last().unwrap();
            let color = match color {
                b'r' => 0,
                b'y' => 1,
                b'g' => 2,
                b'b' => 3,
                _ => panic!(),
            };
            let num: usize = str::from_utf8(rest).unwrap().parse().unwrap();
            tiles[num][color] = true;
        }

        // DP configuration:
        // set consecutive-num identical-color group (horizontal group) as a dp value,
        // identical-num distinct-color group (vertical group) as bitmask.
        const NEG_INF: i32 = i32::MIN / 3;
        let mut dp = vec![NEG_INF; 256];
        dp[0] = 0;
        let decompose = |x| {
            let mask = (1 << 2) - 1;
            [x & mask, (x >> 2) & mask, (x >> 4) & mask, (x >> 6) & mask]
        };
        let mut ans = 0i32;
        for num in 1..=num_max {
            let dp_prev = dp;
            dp = vec![NEG_INF; 256];

            'outer: for (prev_state, next_state) in product(0..256, 0..256) {
                let prev_group_len = decompose(prev_state);
                let next_group_len = decompose(next_state);
                let mut acc = dp_prev[prev_state];
                for color in 0..4 {
                    match (
                        (prev_group_len[color], next_group_len[color]),
                        tiles[num][color],
                    ) {
                        // Disconnect horizontal group
                        ((_, 0), _) | ((0, 1) | (1, 2), true) => {}
                        // Add new horizontal group with length == 3
                        ((2, 3), true) => acc += ((num - 2) + (num - 1) + num) as i32,
                        // Extend a horizontal group with length >= 3
                        ((3, 3), true) => acc += num as i32,
                        _ => continue 'outer,
                    }
                }
                dp[next_state] = dp[next_state].max(acc);
            }

            'outer: for (prev_state, next_state) in product(0..256, 0..256) {
                let prev_group_len = decompose(prev_state);
                let next_group_len = decompose(next_state);

                let mut acc = dp_prev[prev_state];
                match (prev_group_len, next_group_len, tiles[num]) {
                    (_, [0, 0, 0, 0], [true, true, true, true]) => acc += num as i32 * 4,
                    ([r, _, _, _], [s, 0, 0, 0], [b, true, true, true])
                    | ([_, r, _, _], [0, s, 0, 0], [true, b, true, true])
                    | ([_, _, r, _], [0, 0, s, 0], [true, true, b, true])
                    | ([_, _, _, r], [0, 0, 0, s], [true, true, true, b]) => {
                        acc += num as i32 * 3;
                        match ((r, s), b) {
                            ((_, 0), _) | ((0, 1) | (1, 2), true) => {}
                            ((2, 3), true) => acc += ((num - 2) + (num - 1) + num) as i32,
                            ((3, 3), true) => acc += num as i32,
                            _ => continue 'outer,
                        };
                    }
                    _ => continue 'outer,
                }
                dp[next_state] = dp[next_state].max(acc);
            }

            ans = ans.max(*dp.iter().max().unwrap());
        }
        writeln!(output, "{}", ans).unwrap();
    }
}
