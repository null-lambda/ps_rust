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

use std::io::{BufReader, Read /*, Write*/};
use std::ops::*;

fn cross<T: Mul<Output=T> + Sub<Output=T> + Copy>((x1, y1): (T, T), (x2, y2): (T, T)) -> T {
    x1 * y2 - x2 * y1
}

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

    // let mut output_buf = Vec::<u8>::new();

    // goal:
    // https://www.acmicpc.net/problem/1046
    let (n, m) = {
        let mut line = input.line();
        (line.value(), line.value())
    };
    let grid: Vec<Vec<u8>> = (0..n).map(|_| input.line()[0..m].to_vec()).collect();

    // upscale grid coordinates by 2,
    // and set origin to the light source
    let (source_x, source_y) = (0..n)
        .flat_map(|j| (0..m).map(move |i| (i, j)))
        .find(|&(i, j)| grid[j][i] == b'*')
        .map(|(i, j)| (2 * i as i32 + 1, 2 * j as i32 + 1))
        .unwrap();

    type Point<T> = (T, T);
    type Seg<T> = (Point<T>, Point<T>);
    let coord_transform = |x, y| (2 * (x as i32) - source_x, 2 * (y as i32) - source_y);

    let mut wall_count = 0;

    let mut segments: Vec<Seg<i32>> = vec![];
    let mut add_seg = |p: Point<i32>, q: Point<i32>| match cross(p, q).cmp(&0) {
        Ordering::Greater => {
            segments.push((p, q));
        }
        Ordering::Less => {
            segments.push((q, p));
        }
        Ordering::Equal => {}
    };

    let (bx0, by0) = coord_transform(0, 0);
    let (bx1, by1) = coord_transform(m, n);
    (bx0..bx1)
        .step_by(2)
        .for_each(|x| add_seg((x, by0), (x + 2, by0)));
    (bx0..bx1)
        .step_by(2)
        .for_each(|x| add_seg((x, by1), (x + 2, by1)));
    (by0..by1)
        .step_by(2)
        .for_each(|y| add_seg((bx0, y), (bx0, y + 2)));
    (by0..by1)
        .step_by(2)
        .for_each(|y| add_seg((bx1, y), (bx1, y + 2)));

    for j in 0..n {
        for i in 0..m {
            if grid[j][i] == b'#' {
                wall_count += 1;
                let (x1, y1) = coord_transform(i, j);
                let (x2, y2) = (x1 + 2, y1 + 2);
                if x1 > bx0 {
                    add_seg((x1, y1), (x1, y2));
                }
                if x2 < bx1 {
                    add_seg((x2, y1), (x2, y2));
                }
                if y1 > by0 {
                    add_seg((x1, y1), (x2, y1));
                }
                if y2 < by1 {
                    add_seg((x1, y2), (x2, y2));
                }
            }
        }
    }
    segments.sort_unstable();
    segments.dedup();

    use std::cmp::Ordering;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum EventType {
        Start,
        End,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Event<'a>(EventType, &'a Seg<i32>);
    impl<'a> Event<'a> {
        fn unwrap(&self) -> Point<i32> {
            match self.0 {
                EventType::Start => self.1 .0,
                EventType::End => self.1 .1,
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct OrderedSeg(Seg<i32>);

    impl PartialOrd for OrderedSeg {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            let taxi_norm = |(x, y): Point<i32>| x.abs() + y.abs();
            let norm = |(p, q)| taxi_norm(p) + taxi_norm(q);
            match norm(self.0).cmp(&norm(other.0)) {
                Ordering::Equal => None,
                ord => Some(ord),
            }
        }
    }

    impl Ord for OrderedSeg {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other)
                .unwrap_or_else(|| self.0.cmp(&other.0))
        }
    }

    impl std::fmt::Debug for OrderedSeg {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "({} {},{} {})",
                self.0 .0 .0, self.0 .0 .1, self.0 .1 .0, self.0 .1 .1
            )
        }
    }

    use std::collections::BTreeSet;
    let mut events = Vec::<Event>::new();
    let mut current_segs = BTreeSet::<OrderedSeg>::new();
    let mut result = Vec::<Point<f64>>::new();

    for seg in &segments {
        events.push(Event(EventType::Start, seg));
        events.push(Event(EventType::End, seg));
        let &((x, y), _) = seg;
        if x == y && x < 0 {
            // current_segs.insert(OrderedSeg(*seg));
        }
    }

    events.sort_unstable_by(|&e1, &e2| {
        // order by ccw angle, starting from -135 deg
        let ((x1, y1), (x2, y2)) = (e1.unwrap(), e2.unwrap());
        (e1.0 == EventType::End && x1 == y1 && x1 < 0)
            .cmp(&(e2.0 == EventType::End && x2 == y2 && x2 < 0))
            .then_with(|| (x1 < y1).cmp(&(x2 < y2)))
            .then_with(|| (x1 == y1 && 0 < x1).cmp(&(x2 == y2 && 0 < x2)))
            .then_with(|| 0.cmp(&cross((x1, y1), (x2, y2))))
            .then_with(|| (e1.0 == EventType::Start).cmp(&(e2.0 == EventType::Start)))
    });

    // println!("{:?}", events);

    fn ray_intersection(ray: Point<i32>, line: Seg<i32>) -> Point<f64> {
        let (p, q) = line;
        // assume that the line is parallel to the coordinate axis
        if p.0 == q.0 {
            let ix = p.0 as f64;
            let iy = (p.0 * ray.1) as f64 / ray.0 as f64;
            (ix, iy)
        } else {
            let iy = p.1 as f64;
            let ix = (p.1 * ray.0) as f64 / ray.1 as f64;
            (ix, iy)
        }
    }

    let mut debug_limit = 500;
    for event in &events {
        if USE_GRAPHICS {
            debug_limit -= 1;
            if debug_limit <= 0 {
                break;
            }
        }

        let p = event.unwrap();
        let &Event(end_type, &seg) = event;
        //print!("at {:?} {:?}: ", end_type, seg);

        if end_type == EventType::End {
            current_segs.remove(&OrderedSeg(seg));
        }

        let cast_point = |(x, y): Point<i32>| (x as f64, y as f64);
        if let Some(&OrderedSeg(front_seg)) = current_segs.iter().next() {
            if OrderedSeg(seg) < OrderedSeg(front_seg) {
                let intersection = ray_intersection(p, front_seg);
                match end_type {
                    EventType::Start => {
                        // println!("cut end of {:?} by {:?}", front_seg, p);
                        // println!(" ({:?} {:?})", front_seg.0, intersection);
                        //result.push(cast_point(front_seg.0));
                        result.push(intersection);
                        result.push(cast_point(p));
                    }
                    EventType::End => {
                        // println!("cut start of {:?} by {:?}", front_seg, p);
                        // println!(" ({:?} {:?})", intersection, front_seg.1);
                        result.push(cast_point(p));
                        result.push(intersection);
                    }
                }
            } else {
                // println!("shaded by {:?}", front_seg);
            }
        } else {
            // println!("point {:?}", p);
            result.push(cast_point(p));
        }
        //  println!("\t{:?}", current_segs);

        if end_type == EventType::Start {
            current_segs.insert(OrderedSeg(seg));
        }
    }
    result.dedup();

    let mut light_area: f64 = result[..].windows(2).map(|t| cross(t[0], t[1]).abs()).sum();
    light_area *= 0.5;
    // downscale by 1/4
    light_area *= 0.25;
    let room_area = ((by1 - by0) * (bx1 - bx0)) as f64 * 0.25;
    let shadow_area = room_area - light_area - wall_count as f64;
    println!("{}", shadow_area);

    use std::fmt::Write;
    const USE_GRAPHICS: bool = false;
    let mut svg_buffer = String::new();

    if USE_GRAPHICS {
        writeln!(svg_buffer, r#"<circle cx="0" cy="0" r="0.15" fill="red"/>"#).unwrap();
        events
            .iter()
            .enumerate()
            .for_each(|(i, &Event(event_type, &seg))| {
                if event_type != EventType::Start {
                    return;
                }
                let ((x1, y1), (x2, y2)) = seg;
                writeln!(
                    svg_buffer,
                    r#"<path d="M{} {}L{} {}" stroke="black" stroke-width="0.1"/>\
                <text x="{}" y="{}" font-size="0.4">{}</text>"#,
                    x1,
                    y1,
                    (x1 as f32 * 0.2 + x2 as f32 * 0.8),
                    (y1 as f32 * 0.2 + y2 as f32 * 0.8),
                    (x1 + x2) as f32 * 0.5 - 0.2,
                    (y1 + y2) as f32 * 0.5 + 0.2,
                    i
                )
                .unwrap();
            });
        result.iter().enumerate().for_each(|(i, &(x, y))| {
            writeln!(
                svg_buffer,
                r#"<text x="{}" y="{}" font-size="0.3">v{}({} {})</text>\
                <circle cx="{}" cy="{}" r="0.15" fill="blue"/>"#,
                x + 0.2,
                y + 0.2,
                i,
                x,
                y,
                x,
                y
            )
            .unwrap();
        });

        let svg_buffer = format!(
            r#"<svg width="60%" height="60%" viewBox="{} {} {} {}"> {} </svg>"#,
            bx0 - 2,
            by0 - 2,
            bx1 - bx0 + 4,
            by1 - by0 + 4,
            svg_buffer
        );
        std::fs::write("shadow.html", svg_buffer).unwrap();
    }

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
