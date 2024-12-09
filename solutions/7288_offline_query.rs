use std::{
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

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    Enqueue = 0,
    Query = 1,
}
use Event::*;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut prefixes = HashMap::<&str, &str>::new();
    let mut prefixes_inv = HashMap::<&str, &str>::new();
    for _ in 0..n {
        let prefix = input.token();
        let area_name = input.token();
        prefixes.insert(area_name, prefix);
        prefixes_inv.insert(prefix, area_name);
    }

    let r: usize = input.value();
    let mut updates = vec![];
    for _ in 0..r {
        let year: u32 = input.value();
        let op: u8 = input.value();
        let area_name = input.token();
        let arg = input.token();
        updates.push((year, op, area_name, arg));
    }
    updates.sort_unstable_by_key(|t| t.0);

    let mut events = vec![];
    let mut query_numbers = vec![];
    for i in 0.. {
        let start: u32 = input.value();
        let end: u32 = input.value();
        let phone_number = input.token().as_bytes().to_vec();
        if start == 0 && end == 0 {
            break;
        }

        query_numbers.push(phone_number);
        events.push((start, Enqueue, i));
        events.push((end, Query, i));
    }
    events.sort_unstable();

    let mut ans = vec![None; query_numbers.len()];
    let mut query_area_name = vec![None; query_numbers.len()];
    let mut active = HashMap::<&str, HashSet<usize>>::new();
    let mut updates = updates.into_iter().peekable();
    for (year, event, i_query) in events {
        while let Some((_, op, area_name, arg)) = updates.next_if(|(y, _, _, _)| *y <= year) {
            match op {
                1 => {
                    let i = arg.parse::<usize>().unwrap() - 1;
                    for target in active.get(area_name).into_iter().flatten() {
                        let word = &mut query_numbers[*target];
                        let left = &word[..i];
                        let mid = &word[i..i + 1];
                        let right = &word[i..];
                        *word = [left, mid, right].concat();
                    }
                }
                2 => {
                    let i = arg.parse::<usize>().unwrap() - 1;
                    for target in active.get(area_name).into_iter().flatten() {
                        let word = &mut query_numbers[*target];
                        word.swap(i, i + 1);
                    }
                }
                3 => {
                    let old_prefix = prefixes.get(area_name).unwrap();
                    assert!(prefixes_inv.remove(old_prefix).is_some());
                    assert!(prefixes.remove(area_name).is_some());

                    let new_prefix = arg;
                    prefixes.insert(area_name, new_prefix);
                    prefixes_inv.insert(new_prefix, area_name);
                }
                _ => panic!(),
            }
        }

        match event {
            Enqueue => {
                let mut area_name = None;
                let mut prefix_len = 0;
                let word = query_numbers[i_query].as_slice();
                for i in 1..=6.min(word.len()) {
                    if let Some(s) = prefixes_inv.get(std::str::from_utf8(&word[..i]).unwrap()) {
                        area_name = Some(s);
                        prefix_len = i;
                        break;
                    }
                }
                let area_name = area_name.unwrap();
                let rest = &word[prefix_len..];
                query_area_name[i_query] = Some(area_name.to_owned());
                query_numbers[i_query] = rest.to_vec();
                active.entry(area_name).or_default().insert(i_query);
            }
            Query => {
                let area_code = prefixes
                    .get(query_area_name[i_query].unwrap())
                    .unwrap()
                    .to_owned();
                let rest = query_numbers[i_query].clone();
                ans[i_query] = Some((area_code, rest.clone()));
                active
                    .get_mut(query_area_name[i_query].unwrap())
                    .unwrap()
                    .remove(&i_query);
            }
        }
    }

    for a in ans {
        let a = a.unwrap();
        writeln!(output, "{}{}", a.0, std::str::from_utf8(&a.1).unwrap()).unwrap();
    }
}
