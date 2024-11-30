use std::{cmp::Ordering, iter};

fn constrain_lifetime<'a, A: ?Sized + 'a, B, F>(f: F) -> F
where
    F: Fn(&'a A) -> B + 'a,
{
    f
}

#[derive(PartialEq, Eq, Clone)]
struct RevVec<T>(Vec<T>);

impl<T: PartialOrd> PartialOrd for RevVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.iter().rev().partial_cmp(other.0.iter().rev())
    }
}

fn main() {
    let s = std::io::read_to_string(std::io::stdin()).unwrap();
    let s = s.trim().as_bytes();
    let (s1, rest) = s.split_at(s.iter().position(|&x| x == b'+').unwrap());
    let rest = &rest[1..];
    let (s2, s3) = rest.split_at(rest.iter().position(|&x| x == b'=').unwrap());
    let s3 = &s3[1..];

    let n1 = s1.len();
    let n2 = s2.len();
    let n3 = s3.len();
    let n_max = n1.max(n2).max(n3);

    let map_candidates = constrain_lifetime(|s: &[u8]| {
        (0..s.len())
            .rev()
            .map(|i| match s[i] {
                b'0'..=b'9' => {
                    let b = s[i] - b'0';
                    b..=b
                }
                b'?' if s.len() >= 2 && i == 0 => 1..=9,
                b'?' => 0..=9,
                _ => panic!(),
            })
            .chain(iter::repeat(0..=0))
            .take(n_max)
    });
    let s1 = map_candidates(s1);
    let s2 = map_candidates(s2);
    let s3 = map_candidates(s3);

    let mut dp = [None, None];
    dp[0] = Some((RevVec(vec![]), RevVec(vec![]), RevVec(vec![])));
    for ((ds1, ds2), ds3) in s1.zip(s2).zip(s3) {
        let prev = dp;
        dp = [None, None];

        for prev_carry in 0..2 {
            let Some((x3_prev, x1_prev, x2_prev)) = &prev[prev_carry as usize] else {
                continue;
            };
            for d1 in ds1.clone() {
                for d2 in ds2.clone() {
                    let sum = d1 + d2 + prev_carry;
                    let (d3, carry) = (sum % 10, sum / 10);
                    if !ds3.contains(&d3) {
                        continue;
                    }

                    let mut x1 = x1_prev.clone();
                    x1.0.push(d1);

                    let mut x2 = x2_prev.clone();
                    x2.0.push(d2);

                    let mut x3 = x3_prev.clone();
                    x3.0.push(d3);

                    let new = Some((x3, x1, x2));
                    if new > dp[carry as usize] {
                        dp[carry as usize] = new;
                    }
                }
            }
        }
    }

    if let Some((x3, x1, x2)) = &dp[0] {
        for &d in x1.0[..n1].iter().rev() {
            print!("{}", (d + b'0') as char);
        }
        print!("+");
        for &d in x2.0[..n2].iter().rev() {
            print!("{}", (d + b'0') as char);
        }
        print!("=");
        for &d in x3.0[..n3].iter().rev() {
            print!("{}", (d + b'0') as char);
        }
    } else {
        print!("-1");
    }
}
