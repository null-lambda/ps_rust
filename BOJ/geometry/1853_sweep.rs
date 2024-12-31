mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let mut read_polygon = || -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
        let n = input.value();
        let points: Vec<(i32, i32)> = (0..n).map(|_| (input.value(), input.value())).collect();

        let i_min = points
            .iter()
            .enumerate()
            .min_by_key(|(_, &(_, z))| z)
            .unwrap()
            .0;
        let i_max = points
            .iter()
            .enumerate()
            .max_by_key(|(_, &(_, z))| z)
            .unwrap()
            .0;

        assert!(n >= 3);
        assert!(i_min != i_max);
        if i_min < i_max {
            let left = points[i_min..=i_max].to_vec();
            let right = points[i_max..]
                .iter()
                .chain(points[..=i_min].iter())
                .rev()
                .copied()
                .collect();
            (left, right)
        } else {
            let right = points[i_max..=i_min].iter().rev().copied().collect();
            let left = points[i_min..]
                .iter()
                .chain(points[..=i_max].iter())
                .copied()
                .collect();
            (left, right)
        }
    };

    let (xl, xr) = read_polygon();
    let (yl, yr) = read_polygon();
    let lines = [xl, xr, yl, yr];

    #[derive(Debug, Clone, Copy)]
    enum LineType {
        XL = 0,
        XR = 1,
        YL = 2,
        YR = 3,
    }
    let line_types: [LineType; 4] = [LineType::XL, LineType::XR, LineType::YL, LineType::YR];

    let mut z_ordering: Vec<(LineType, i32)> = line_types
        .iter()
        .zip(lines.iter())
        .flat_map(|(&line_type, points)| {
            points[1..points.len()]
                .iter()
                .map(move |&(_, z)| (line_type, z))
        })
        .collect();
    z_ordering.sort_unstable_by_key(|&(_line_type, z)| z);

    /*
    for i in 0..4 {
        println!("lines[{}]={:?}", i, lines[i]);
    }
    println!("{:?}", z_ordering);
    */

    let mut dx_prev = 0.0f64;
    let mut dy_prev = 0.0f64;
    let mut z_prev = lines[LineType::XL as usize][0].1;
    let mut total = 0.0f64;
    let mut lines_idx = [0, 0, 0, 0];

    let mut intersection = [0.0f64; 4];
    // optimization(d): factor out common divisors
    for (line_type, z) in z_ordering {
        line_types.iter().for_each(|&lt| {
            let line = &lines[lt as usize];
            let idx = lines_idx[lt as usize];
            // println!("line[{:?}][{}]: {:?}", lt, idx, (line[idx] ,line[idx+1]));

            let (x1, z1) = line[idx];
            let (x2, z2) = *line.get(idx + 1).unwrap_or(&(x1, z1));
            let (t, t1) = (z2 - z1, z - z1);
            intersection[lt as usize] = if t != 0 {
                (((t - t1) * x1 + t1 * x2) as f64) / (t as f64)
            } else {
                x2 as f64
            };
        });

        let dx = intersection[LineType::XR as usize] - intersection[LineType::XL as usize];
        let dy = intersection[LineType::YR as usize] - intersection[LineType::YL as usize];

        let dz = z - z_prev;
        if dz != 0 {
            let d_vol = (dz as f64) * (dx * (2.0 * dy + dy_prev) + dx_prev * (2.0 * dy_prev + dy));
            total += d_vol;

            /* println!(
                "z={}: {:?}, {:?} -> {:?}, {} | vol={}",
                z,
                lines_idx,
                (dx_prev, dy_prev),
                (dx, dy),
                dz,
                d_vol
            ); */
        }

        lines_idx[line_type as usize] += 1;
        dx_prev = dx;
        dy_prev = dy;
        z_prev = z;
    }
    total *= 1.0 / 6.0;

    writeln!(output_buf, "{}", total).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
