use std::{collections::HashMap, io::Write};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> Option<T>
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().ok()
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

fn xorshift32(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

fn xorshift32_nth(mut x: u32, n: u32) -> u32 {
    for _ in 0..n {
        x = xorshift32(x);
    }
    x
}

fn inv_xor_shl(x: u32, shift: u32) -> u32 {
    let mut res = 0;
    for i in (0..32).step_by(shift as usize) {
        res ^= x << i;
    }
    res
}

fn inv_xor_shr(x: u32, shift: u32) -> u32 {
    let mut res = 0;
    for i in (0..32).step_by(shift as usize) {
        res ^= x >> i;
    }
    res
}

fn inv_xorshift32(x: u32) -> u32 {
    let x = inv_xor_shl(x, 5);
    let x = inv_xor_shr(x, 17);
    inv_xor_shl(x, 13)
}

fn xorshift32_nth_mat(n: u32) -> [u32; 32] {
    let mut res = [0; 32];
    for i in 0..32 {
        res[i] = xorshift32_nth(1 << i, n);
    }
    res
}

fn apply_mat(mat: [u32; 32], x: u32) -> u32 {
    let mut res = 0;
    for i in 0..32 {
        if (x >> i) & 1 == 1 {
            res ^= mat[i];
        }
    }
    res
}

fn precomputed_block_mat() -> [u32; 32] {
    [
        437906867, 506273532, 1246454606, 2241393224, 958817800, 3472911366, 3064280302,
        1337743425, 1789918472, 2952294299, 3262575055, 2871278968, 471368716, 2032964697,
        2822943001, 3102833472, 2478598240, 9866905, 2120828680, 56646574, 1601371120, 2641177728,
        173376804, 1225692501, 204540805, 2983620998, 753404461, 3824765745, 3148149924,
        3672362573, 560093186, 3007903540,
    ]
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let x: u32 = input.value().unwrap();
    let t: u32 = input.value().unwrap();
    let p: u32 = 4_294_967_295;

    let block_size = ((p as f32).sqrt().ceil() as u32).max(2);
    // let block_mat = xorshift32_nth_mat(block_size);
    let block_mat = precomputed_block_mat();

    let mut right_dict: HashMap<_, _> = Default::default();
    let mut right = t;
    for i in 0..block_size {
        right_dict.entry(right).or_insert(i);
        right = inv_xorshift32(right);
    }

    let mut log = None;
    let mut left = x;
    for u in 0..p.div_ceil(block_size) {
        if let Some(v) = right_dict.get(&left) {
            log = Some(u * block_size + v);
            break;
        }
        left = apply_mat(block_mat, left);
    }

    writeln!(output, "{}", log.unwrap() + 1).unwrap();
}
