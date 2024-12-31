use std::io::Write;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    pub struct InputAtOnce {
        _buf: &'static str,
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let _buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let _buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(_buf, stat[6])) };
        let iter = _buf.split_ascii_whitespace();
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::new(stdout)
    }
}

fn encode_freq(word: &str) -> [u128; 2] {
    let mut freq = [0u8; 26];
    for b in word.bytes() {
        freq[(b - b'A') as usize] += 1;
    }

    let mut mask = [0; 2];
    for i in 0..26 {
        let (block, j) = (i / 16, i % 16);
        mask[block] |= !(!0 << (freq[i] as u128)) << j * 8;
    }
    mask
}

fn encode_unique(word: &str) -> u32 {
    let mut mask = 0;
    for b in word.bytes() {
        mask |= 1 << (b - b'A');
    }
    mask
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let mut words = vec![];
    loop {
        let word = input.token();
        if word == "-" {
            break;
        }

        words.push((encode_freq(word), encode_unique(word)));
    }

    loop {
        let word = input.token();
        if word == "#" {
            break;
        }

        let resource_freq = encode_freq(word);
        let resource_unique = encode_unique(word);
        let mut acc = [0u32; 26];

        for &(freq, unique) in &words {
            if resource_freq[0] & freq[0] != freq[0] || resource_freq[1] & freq[1] != freq[1] {
                continue;
            }

            for i in 0..26 {
                if (unique >> i) & 1 != 0 {
                    acc[i] += 1;
                }
            }
        }

        let mut min = u32::MAX;
        let mut max = 0;
        for i in 0..26 {
            if (resource_unique >> i) & 1 == 0 {
                continue;
            }
            min = min.min(acc[i]);
            max = max.max(acc[i]);
        }

        for i in 0..26 {
            if (resource_unique >> i) & 1 == 0 || acc[i] != min {
                continue;
            }
            write!(output, "{}", (i as u8 + b'A') as char).unwrap();
        }
        write!(output, " {} ", min).unwrap();

        for i in 0..26 {
            if (resource_unique >> i) & 1 == 0 || acc[i] != max {
                continue;
            }
            write!(output, "{}", (i as u8 + b'A') as char).unwrap();
        }
        write!(output, " {}\n", max).unwrap();
    }
}
