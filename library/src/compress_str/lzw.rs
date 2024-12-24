pub mod lzw {
    use std::{collections::HashMap, iter};

    const MAX_BITS: u32 = 20;
    const CLEAR_CODE: u32 = 256;
    const TERMINAL_CODE: u32 = 257;
    const INC_BITS_CODE: u32 = 258;

    pub fn encode<'a>(s: &'a [u8]) -> Vec<u8> {
        let mut dict: HashMap<Vec<u8>, u32> = (0..256).map(|i| (vec![i as u8], i)).collect();
        let mut next_code: u32 = 259;
        let mut n_bits = u32::BITS - next_code.leading_zeros();

        let mut res = vec![];
        let mut push_code = |n_bits: u32, code: u32| {
            res.extend((0..n_bits).map(|i| (code >> (n_bits - i - 1)) & 1 == 1));
        };

        let mut pattern = vec![];
        for &c in s {
            let mut next_pattern = pattern.clone();
            next_pattern.push(c);
            if dict.contains_key(&next_pattern) {
                pattern = next_pattern;
            } else {
                push_code(n_bits, dict[&pattern]);
                dict.insert(next_pattern, next_code);
                next_code += 1;
                let n_bits_prev = n_bits;
                n_bits = u32::BITS - next_code.leading_zeros();

                pattern = vec![c];
                if n_bits_prev != n_bits {
                    if n_bits > MAX_BITS {
                        if !pattern.is_empty() {
                            push_code(n_bits_prev, dict[&pattern]);
                        }
                        push_code(n_bits_prev, CLEAR_CODE);
                        dict = (0..256).map(|i| (vec![i as u8], i)).collect();
                        next_code = 259;
                        n_bits = u32::BITS - next_code.leading_zeros();
                        pattern = vec![];
                    } else {
                        push_code(n_bits_prev, INC_BITS_CODE);
                    }
                }
            }
        }

        if !pattern.is_empty() {
            push_code(n_bits, dict[&pattern]);
        }

        push_code(n_bits, TERMINAL_CODE);

        res.extend((0..(8 - res.len() % 8) % 8).map(|_| false));
        res.chunks(8)
            .map(|chunk| {
                let mut acc = 0;
                for &b in chunk {
                    acc = acc * 2 + b as u8 as u32;
                }
                acc as u8
            })
            .collect()
    }

    pub fn decode<'a>(s: &'a [u8]) -> Vec<u8> {
        let mut dict: HashMap<u32, Vec<u8>> = (0..256).map(|i| (i, vec![i as u8])).collect();
        let mut next_code: u32 = 259;
        let mut n_bits = u32::BITS - next_code.leading_zeros();

        let mut pattern = vec![];
        let mut res = vec![];

        let mut bits = s
            .iter()
            .flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1));

        loop {
            let mut c = 0;
            for _ in 0..n_bits {
                c = c * 2 + bits.next().unwrap() as u32;
            }

            match c {
                TERMINAL_CODE => break,
                INC_BITS_CODE => n_bits += 1,
                CLEAR_CODE => {
                    dict = (0..256).map(|i| (i, vec![i as u8])).collect();
                    next_code = 259;
                    n_bits = u32::BITS - next_code.leading_zeros();
                    pattern = vec![];
                }
                _ => {
                    if pattern.is_empty() {
                        pattern = dict[&c].clone();
                        res.push(pattern[0]);
                        continue;
                    }

                    let mut next_pattern;
                    if let Some(x) = dict.get(&c) {
                        next_pattern = x.clone();
                    } else {
                        next_pattern = pattern.clone();
                        next_pattern.push(pattern[0]);
                    }

                    res.extend(next_pattern.iter().cloned());
                    dict.insert(
                        next_code,
                        pattern
                            .iter()
                            .chain(Some(&next_pattern[0]))
                            .copied()
                            .collect(),
                    );
                    next_code += 1;
                    pattern = next_pattern;
                }
            }
        }
        res
    }
}
