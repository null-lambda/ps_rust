use std::{
    collections::HashMap,
    io::{BufRead, Write},
    iter,
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

const CSV: &str = r#"name,index,kodama,hikari_main,hikari_optional,interval_1,interval_2,nozomi,initial,terminal,special,dist_cost,dist_real
Tokyo,0,1,1,,,,1,1,1,,0,0
Shinagawa,1,1,1,,,,1,1,,,6.8,6.8
Shin-Yokohama,2,1,1,,1,,1,1,,,28.8,25.5
Odawara,3,1,,1,1,,,,,,83.9,76.7
Atami,4,1,,1,1,,,,,,104.6,95.4
Mishima,5,1,,1,1,,,1,,1,120.7,111.3
Shin-Fuji,6,1,,,1,,,,,,146.2,135
Shizuoka,7,1,,1,,,,1,1,,180.2,167.4
Kakegawa,8,1,,,1,,,,,,229.3,211.3
Hamamatsu,9,1,,1,1,,,1,1,,257.1,238.9
Toyohashi,10,1,,1,1,,,,,,293.6,274.2
Mikawa-Anjo,11,1,,,1,,,,,,336.3,312.8
Nagoya,12,1,1,,1,2,1,1,1,1,366,342
Gifu-Hashima,13,1,,1,,2,,,,1,396.3,367.1
Maibara,14,1,,1,,2,,,,1,445.9,408.2
Kyoto,15,1,1,,,2,1,,,,513.6,476.3
Shin-Osaka,16,1,1,,,,1,1,1,,552.6,515.4"#;

type Table = Vec<Row>;
type Row = HashMap<String, String>;
fn parse_csv(s: &str) -> Table {
    let rows: Vec<Vec<_>> = s
        .lines()
        .map(|line| line.split(',').map(str::to_string).collect())
        .collect();
    let header = rows[0].clone();

    let mut rows_grouped = vec![];
    for i in 1..rows.len() {
        rows_grouped.push(
            header
                .iter()
                .cloned()
                .zip(rows[i].iter().cloned())
                .collect(),
        );
    }
    rows_grouped
}

struct Context {
    table: Table,
}

impl Context {
    fn new() -> Self {
        Self {
            table: parse_csv(CSV),
        }
    }
}

impl Context {
    fn get_by_name(&self, name: &str) -> Option<&Row> {
        self.table.iter().find(|row| row["name"] == name)
    }

    fn get_by_index(&self, index: u8) -> Option<&Row> {
        self.table
            .iter()
            .find(|row| row["index"].parse::<u8>().unwrap() == index)
    }
}

enum ErrorCode {
    InvalidStation = 200,
    DuplicateStation = 300,
    DirectionChange = 400,
    InvalidPattern = 500,
    DuplicateTier = 600,
}

mod tier {
    pub const KODAMA: &str = "kodama";
    pub const HIKARI: &str = "hikari";
    pub const NOZOMI: &str = "nozomi";
}

fn check_kodama(cx: &Context, xs: &[String]) -> Option<String> {
    let indices: Vec<u8> = xs
        .iter()
        .map(|x| cx.get_by_name(x).unwrap()["index"].parse().unwrap())
        .collect();
    let min = *indices.iter().min().unwrap();
    let max = *indices.iter().max().unwrap();
    let expected: Vec<_> = (min..=max)
        .filter(|&i| cx.get_by_index(i).unwrap()["kodama"] == "1")
        .collect();
    if indices != expected && indices.iter().rev().copied().collect::<Vec<_>>() != expected {
        return None;
    }

    Some(tier::KODAMA.to_string())
}

fn check_hikari(cx: &Context, xs: &[String]) -> Option<String> {
    // row.get(KODAMA).unwrap() == &1
    None
}

fn check_nozomi(cx: &Context, xs: &[String]) -> Option<String> {
    // row.get(KODAMA).unwrap() == &1
    None
}

fn verify(cx: &Context, xs: &[String]) -> Result<String, ErrorCode> {
    for x in xs {
        if cx.get_by_name(x).is_none() {
            return Err(ErrorCode::InvalidStation);
        }
    }

    let mut xs_cloned = xs.to_vec();
    xs_cloned.sort_unstable();
    xs_cloned.dedup();
    if xs.len() != xs_cloned.len() {
        return Err(ErrorCode::DuplicateStation);
    }

    let indices: Vec<u8> = xs
        .iter()
        .map(|x| cx.get_by_name(x).unwrap()["index"].parse().unwrap())
        .collect();

    let asc = indices.windows(2).all(|w| w[0] < w[1]);
    let desc = indices.windows(2).all(|w| w[0] > w[1]);
    if !asc && !desc {
        return Err(ErrorCode::DirectionChange);
    }

    let tiers: Vec<String> = [
        check_kodama(cx, xs),
        check_hikari(cx, xs),
        check_nozomi(cx, xs),
    ]
    .into_iter()
    .flatten()
    .collect();

    if tiers.len() == 0 {
        return Err(ErrorCode::InvalidPattern);
    }
    if tiers.len() > 1 {
        return Err(ErrorCode::DuplicateTier);
    }

    Ok(tiers.into_iter().next().unwrap())
}

fn main() {
    let input = std::io::BufReader::new(std::io::stdin());
    let mut input = input.lines().flatten();
    let mut output = simple_io::stdout();

    let cx = Context::new();

    for _ in 0..input.next().unwrap().parse::<usize>().unwrap() {
        let n: usize = input.next().unwrap().parse().unwrap();

        let mut path = vec![];
        for _ in 0..n {
            let s = input.next().unwrap();
            let mut s = s.as_str();

            let postfix = " station";
            if s.ends_with(postfix) {
                let (left, _) = s.split_at(s.len() - postfix.len());
                s = left;
            }
            path.push(s.to_string());
        }

        match verify(&cx, &path) {
            Ok(tier) => writeln!(output, "{}", tier).unwrap(),
            Err(code) => writeln!(output, "ERROR {}", code as u32).unwrap(),
        }
    }
}
