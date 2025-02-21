use std::io::Write;

use std::{collections::HashMap, hash::Hash};

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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

fn kmp<'a: 'c, 'b: 'c, 'c, T: PartialEq>(
    s: impl IntoIterator<Item = T> + 'a,
    pattern: &'b [T],
) -> impl Iterator<Item = usize> + 'c {
    // Build a jump table
    let mut jump_table = vec![0];
    let mut i_prev = 0;
    for i in 1..pattern.len() {
        while i_prev > 0 && pattern[i] != pattern[i_prev] {
            i_prev = jump_table[i_prev - 1];
        }
        if pattern[i] == pattern[i_prev] {
            i_prev += 1;
        }
        jump_table.push(i_prev);
    }

    // Search patterns
    let mut j = 0;
    s.into_iter().enumerate().filter_map(move |(i, c)| {
        while j == pattern.len() || j > 0 && pattern[j] != c {
            j = jump_table[j - 1];
        }
        if pattern[j] == c {
            j += 1;
        }
        (j == pattern.len()).then(|| i + 1 - pattern.len())
    })
}

fn sorted2<T: Ord>([a, b]: [T; 2]) -> [T; 2] {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}

fn transpose(p: [i32; 2]) -> [i32; 2] {
    [p[1], p[0]]
}

fn signed_area(ps: &[[i32; 2]]) -> i64 {
    let n = ps.len();
    let mut acc = 0;
    for i in 0..n {
        let p = ps[i].map(|x| x as i64);
        let q = ps[(i + 1) % n].map(|x| x as i64);
        acc += p[0] * q[1] - p[1] * q[0];
    }
    acc /= 2;
    acc
}

type PolygonCode = Vec<i32>;

fn encode_poly(ps: &[[i32; 2]]) -> PolygonCode {
    let n = ps.len();
    let mut code = vec![];
    for i in 0..n {
        let prev = ps[i];
        let p = ps[(i + 1) % n];
        let next = ps[(i + 2) % n];

        let d_prev = [p[0] - prev[0], p[1] - prev[1]];
        let d = [next[0] - p[0], next[1] - p[1]];

        let s = if d_prev[0] as i64 * d[1] as i64 - d_prev[1] as i64 * d[0] as i64 > 0 {
            0
        } else {
            -1
        };
        code.push(s);
        code.push(d[0].abs().max(d[1].abs()));
    }
    code
}

fn eq_cyclic(ps: &PolygonCode, qs: &PolygonCode) -> bool {
    if ps.len() != qs.len() {
        return false;
    }
    if ps.is_empty() {
        return true;
    }

    let ps_ext = ps.iter().chain(&ps[..ps.len() - 1]).copied();
    kmp(ps_ext, &qs).next().is_some()
}

fn eq_diheral(ps: &mut PolygonCode, qs: &PolygonCode) -> bool {
    eq_cyclic(ps, qs) || {
        ps.reverse();
        eq_cyclic(ps, qs)
    }
}

fn solve_horizontal(ps: &[[i32; 2]], target_area: i64) -> Option<[[i32; 2]; 2]> {
    let n = ps.len();

    // Step 1. Find the splitting line
    let mut events = vec![];
    for i in 0..n {
        let p = ps[i];
        let q = ps[(i + 1) % n];
        if p[1] == q[1] {
            let delta_section = q[0] - p[0];
            events.push((p[1], delta_section));
        }
    }
    events.sort_unstable();
    debug::with(|| println!("{:?}", events));

    let mut y_prev = 0;
    let mut section = 0i32;
    let mut area_prefix = 0i64;
    let mut y_split = None;
    for (y, dx) in events {
        let dy = y - y_prev;
        area_prefix += dy as i64 * section as i64;
        debug::with(|| {
            println!(
                "dy={} dx={} section={} area_prefix={}",
                dy, dx, section, area_prefix
            )
        });

        let da = area_prefix - target_area;
        if da >= 0 {
            if da % section as i64 == 0 {
                y_split = Some(y - (da / section as i64) as i32);
            }
            break;
        }

        section += dx;
        y_prev = y;
    }

    let y_split = y_split?;

    // Step 2. split the polygon into two parts
    debug::with(|| println!("Step 2"));
    let mut x_splits = vec![];
    for i in 0..n {
        let prev = ps[(i + n - 1) % n];
        let p = ps[i];
        let q = ps[(i + 1) % n];
        let next = ps[(i + 2) % n];
        if p[1] == q[1]
            && p[1] == y_split
            && (p[1] - prev[1]).signum() * (next[1] - p[1]).signum() == 1
        {
            x_splits.push((sorted2([p[0], q[0]]), i as u32));
        } else if p[1].min(q[1]) < y_split && y_split < p[1].max(q[1]) {
            x_splits.push(([p[0], p[0]], i as u32));
        }
    }
    x_splits.sort_unstable();

    println!("x_splits :{:?}", x_splits);

    if x_splits.len() != 2 {
        return None;
    }

    let mut lower = vec![];
    let mut upper = vec![];
    for i in 0..n {
        let prev = ps[(i + n - 1) % n];
        let p = ps[i];
        let next = ps[(i + 1) % n];

        if p[1] == y_split {
            if prev[1] < y_split || next[1] < y_split {
                lower.push(p);
            } else {
                upper.push(p);
            }
            continue;
        }

        if p[1] < y_split {
            lower.push(p);
        } else {
            upper.push(p);
        }
        if p[1].min(next[1]) < y_split && y_split < p[1].max(next[1]) {
            let x = p[0];
            lower.push([x, y_split]);
            upper.push([x, y_split]);
        }
    }

    // Step 3. Test congruence
    let mut lower = encode_poly(&lower);
    let upper = encode_poly(&upper);

    debug::with(|| println!("{:?}", lower));
    debug::with(|| println!("{:?}", upper));

    if !eq_diheral(&mut lower, &upper) {
        return None;
    }

    Some([[x_splits[0].0[1], y_split], [x_splits[1].0[0], y_split]])
}

fn render_to_svg(ps: &[[i32; 2]]) {
    use std::fmt::Write;
    use std::fs::File;

    if ps.len() < 2 {
        return;
    }

    let (min_x, max_x, min_y, max_y) = ps.iter().fold(
        (i32::MAX, i32::MIN, i32::MAX, i32::MIN),
        |(min_x, max_x, min_y, max_y), &[x, y]| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    );

    let width = (max_x - min_x) as f64 * 1.1;
    let height = (max_y - min_y) as f64 * 1.1;
    let offset_x = min_x as f64 - (width - (max_x - min_x) as f64) / 2.0;
    let offset_y = min_y as f64 - (height - (max_y - min_y) as f64) / 2.0;

    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='{} {} {} {}' width='100%' height='100%'>",
        offset_x, offset_y, width, height
    )
    .unwrap();

    for ps_window in ps.windows(2) {
        let &[x1, y1] = &ps_window[0];
        let &[x2, y2] = &ps_window[1];
        let thickness = ((width + height) / 500.0).max(0.3);
        writeln!(
            svg,
            "    <line x1='{}' y1='{}' x2='{}' y2='{}' stroke='black' stroke-width='{}'/>",
            x1, y1, x2, y2, thickness
        )
        .unwrap();
    }

    svg.push_str("</svg>");

    let mut file = File::create("./dbg/poly.svg").expect("Unable to create file");
    file.write_all(svg.as_bytes())
        .expect("Unable to write data");
}
fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut ps: Vec<[i32; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();

    debug::with(|| render_to_svg(&ps));

    let mut res = || {
        let mut a = signed_area(&ps);
        if a < 0 {
            ps.reverse();
            a = -a;
        }
        if a % 2 == 1 {
            return None;
        }

        solve_horizontal(&ps, a / 2).or_else(|| {
            ps.iter_mut().for_each(|p| *p = transpose(*p));
            ps.reverse();
            solve_horizontal(&ps, a / 2).map(|res| res.map(transpose))
        })
    };

    if let Some([p, q]) = res() {
        writeln!(output, "{} {} {} {}", p[0], p[1], q[0], q[1]).unwrap();
    } else {
        writeln!(output, "NO").unwrap();
    }
}
