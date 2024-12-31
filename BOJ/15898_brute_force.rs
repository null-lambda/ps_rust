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

fn parse_coeff(s: &str) -> i8 {
    match s {
        "R" => 7,
        "B" => 5,
        "G" => 3,
        "Y" => 2,
        "W" => 0,
        _ => panic!(),
    }
}

pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)> + Clone
where
    I: IntoIterator,
    I::Item: Clone,
    I::IntoIter: Clone,
    J: IntoIterator,
    J::IntoIter: Clone,
{
    let j = j.into_iter();
    i.into_iter()
        .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
}

fn rot(s: Vec<i8>) -> Vec<i8> {
    let mut res = vec![0; s.len()];
    for i in 0..4 {
        for j in 0..4 {
            res[j * 4 + 3 - i] = s[i * 4 + j];
        }
    }
    res
}

fn rots(mut s: Vec<i8>) -> Vec<Vec<i8>> {
    let mut res = vec![];
    for _ in 0..4 {
        res.push(s.clone());
        s = rot(s);
    }
    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut effects: Vec<Vec<Vec<i8>>> = vec![];
    let mut coeffs: Vec<Vec<Vec<i8>>> = vec![];
    for _ in 0..n {
        effects.push(rots((0..16).map(|_| input.value()).collect()));
        coeffs.push(rots((0..16).map(|_| parse_coeff(input.token())).collect()));
    }

    let mut ans = 0u32;
    let samples =
        product(product(0..n, 0..n), 0..n).filter(|&((i, j), k)| i != j && j != k && k != i);
    let modifiers = || product(product(0..2, 0..2), 0..4);
    let sampled_modifiers = || product(product(modifiers(), modifiers()), modifiers());
    for ((i, j), k) in samples {
        for ((mi, mj), mk) in sampled_modifiers() {
            let mut combined_effect = vec![0i8; 25];
            let mut combined_coeffs = vec![0i8; 25];
            for (p, ((dy, dx), r)) in [(i, mi), (j, mj), (k, mk)] {
                for (y, x) in product(0..4, 0..4) {
                    combined_effect[(dy + y) * 5 + (dx + x)] =
                        (combined_effect[(dy + y) * 5 + (dx + x)] + effects[p][r][y * 4 + x])
                            .min(9)
                            .max(0);
                    let c = coeffs[p][r][y * 4 + x];
                    if c != 0 {
                        combined_coeffs[(dy + y) * 5 + (dx + x)] = c;
                    }
                }
            }

            let dot = (0..25)
                .map(|i| combined_effect[i] as u32 * combined_coeffs[i] as u32)
                .sum::<u32>();
            ans = ans.max(dot);
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
