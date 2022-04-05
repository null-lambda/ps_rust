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
            token.parse().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while matches!(s.last(), Some(b'\n' | b'\r' | 0)) {
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
                .map_or_else(|| self.len(), |idx| idx + 1);
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

use std::ops::Range;

struct SegTree {
    n: usize,
    n_segs: Vec<i32>,
    sum: Vec<i32>,
    seg_len: Vec<i32>,
}

impl SegTree {
    fn new(xs: &[i32]) -> Self {
        use std::iter::repeat;
        let n = xs.len();
        let n_segs = vec![0; 2 * n];
        let sum = vec![0; 2 * n];
        let mut seg_len: Vec<i32> = repeat(0)
            .take(n)
            .chain(xs.windows(2).map(|t| t[1] - t[0]))
            .collect();
        seg_len.resize(2 * n, 0);

        for i in (1..n).rev() {
            seg_len[i] = seg_len[i << 1] + seg_len[i << 1 | 1];
        }
        Self {
            n,
            n_segs,
            sum,
            seg_len,
        }
    }

    fn apply(&mut self, i: usize, value: i32) {
        self.n_segs[i] += value;
    }

    fn update_sum(&mut self, i: usize) {
        self.sum[i] = if self.n_segs[i] > 0 {
            self.seg_len[i]
        } else if i < self.n {
            self.sum[i << 1] + self.sum[i << 1 | 1usize]
        } else {
            0
        };
    }

    // add on interval [left, right)
    fn add_range(&mut self, range: Range<usize>, value: i32) {
        let Range { mut start, mut end } = range;
        debug_assert!(end <= self.n);
        start += self.n;
        end += self.n;
        let (mut update_left, mut update_right) = (false, false);
        while start < end {
            if update_left {
                self.update_sum(start - 1);
            }
            if update_right {
                self.update_sum(end);
            }
            if start & 1 != 0 {
                self.apply(start, value);
                self.update_sum(start);
                update_left = true;
            }
            if end & 1 != 0 {
                self.apply(end - 1, value);
                self.update_sum(end - 1);
                update_right = true;
            }

            start = (start + 1) >> 1;
            end = end >> 1;
        }

        start -= 1;
        while end > 0 {
            if update_left {
                self.update_sum(start);
            }
            if update_right && !(update_left && start == end) {
                self.update_sum(end);
            }
            start >>= 1;
            end >>= 1;
        }
    }

    fn sum(&self) -> i32 {
        self.sum[1]
    }
}

macro_rules! enum_int {
    ($(#[$meta:meta])* enum $name:ident {
        $($vname:ident $(= $val:expr)?,)*
    }) => {
        $(#[$meta])*  enum $name {
            $($vname $(= $val)?,)*
        }

        impl std::convert::TryFrom<usize> for $name {
            type Error = ();

            fn try_from(v: usize) -> Result<Self, Self::Error> {
                match v {
                    $(x if x == $name::$vname as usize => Ok($name::$vname),)*
                    _ => Err(()),
                }
            }
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::cmp::Reverse;
    use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
    use std::convert::TryFrom;
    use std::convert::TryInto;

    enum_int! {
        #[derive(Debug, Clone, Copy)]
        enum Direction {
            Right = 0,
            Up = 1,
            Left = 2,
            Down = 3,
        }
    }

    #[derive(Debug)]
    struct Projectile {
        x: i32,
        y: i32,
        dir: Direction,
        duration: i32,
    }

    enum_int! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        enum CollisionType {
            RightUp = 0,
            RightDown = 1,
            LeftUp = 2,
            LeftDown = 3,
            RightLeft = 4,
            UpDown = 5,
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum LineDirection {
        ToRight,
        ToLeft,
    }

    let width: i32 = input.value();
    let height: i32 = input.value();
    let n_projectiles: usize = input.value();
    let mut projectiles: Vec<Projectile> = Vec::with_capacity(n_projectiles);
    let mut lines: [HashMap<i32, BTreeMap<i32, (LineDirection, usize)>>; 6] = Default::default();

    for idx in 0..n_projectiles {
        let x: i32 = input.value();
        let y: i32 = input.value();
        let dir: Direction = input.value::<usize>().try_into().unwrap();
        projectiles.push(Projectile {
            x,
            y,
            dir,
            duration: i32::MAX,
        });

        let mut insert = |collision_type, line_pos, pos, line_dir| {
            lines[collision_type as usize]
                .entry(line_pos)
                .or_default()
                .insert(pos, (line_dir, idx))
        };
        use {CollisionType::*, LineDirection::*};
        match dir {
            Direction::Right => {
                insert(RightLeft, y, x, ToRight);
                insert(RightUp, x - y, 2 * x, ToRight);
                insert(RightDown, x + y, 2 * x, ToRight);
            }
            Direction::Left => {
                insert(RightLeft, y, x, ToLeft);
                insert(LeftUp, x + y, 2 * x, ToLeft);
                insert(LeftDown, x - y, 2 * x, ToLeft);
            }
            Direction::Down => {
                insert(UpDown, x, y, ToRight);
                insert(LeftDown, x - y, 2 * x, ToRight);
                insert(RightDown, x + y, 2 * x, ToLeft);
            }
            Direction::Up => {
                insert(UpDown, x, y, ToLeft);
                insert(LeftUp, x + y, 2 * x, ToRight);
                insert(RightUp, x - y, 2 * x, ToLeft);
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Event {
        time: i32,
        x_left: i32,
        x_right: i32,
        collision_type: CollisionType,
        line_pos: i32,
        idx_left: usize,
        idx_right: usize,
    }

    let mut events = BinaryHeap::<Reverse<Event>>::new();
    for (collision_type, lines_inner) in lines.iter().enumerate() {
        for (&line_pos, line) in lines_inner.iter() {
            if line.len() <= 1 {
                continue;
            }
            /*
            println!(
                "{:?} => {:?}",
                (CollisionType::try_from(collision_type), line_pos),
                line
            );
            */
            let mut it = line.iter().peekable();
            while let Some(projectile) = it.next() {
                if let (
                    (&x_left, &(LineDirection::ToRight, idx_left)),
                    Some(&(&x_right, &(LineDirection::ToLeft, idx_right))),
                ) = (projectile, it.peek())
                {
                    events.push(Reverse(Event {
                        time: x_right - x_left,
                        collision_type: collision_type.try_into().unwrap(),
                        line_pos,
                        x_left,
                        x_right,
                        idx_left,
                        idx_right,
                    }));
                    it.next();
                }
            }
        }
    }

    let mut current_collisions = HashSet::<usize>::new();
    while let Some(Reverse(event)) = events.pop() {
        let Event {
            time,
            collision_type,
            line_pos,
            idx_left,
            idx_right,
            x_left,
            x_right,
        } = event;
        let line = lines[collision_type as usize].get_mut(&line_pos).unwrap();

        let active1 = projectiles[idx_left].duration == i32::MAX;
        let active2 = projectiles[idx_right].duration == i32::MAX;
        if active1 && active2 {
            current_collisions.insert(idx_left);
            current_collisions.insert(idx_right);
            // println!("{:?}", (event, active1, active2, collision_type, line_pos));
        }

        let erase1 = !active1 || active2;
        let erase2 = !active2 || active1;
        if erase1 || erase2 {
            let new_left = if erase1 {
                line.range(..x_left).rev().next()
            } else {
                line.get_key_value(&x_left)
            };
            let new_right = if erase2 {
                line.range(x_right + 1..).next()
            } else {
                line.get_key_value(&x_right)
            };
            if let (
                Some((&x_left, &(LineDirection::ToRight, idx_left))),
                Some((&x_right, &(LineDirection::ToLeft, idx_right))),
            ) = (new_left, new_right)
            {
                events.push(Reverse(Event {
                    time: x_right - x_left,
                    collision_type,
                    line_pos,
                    x_left,
                    x_right,
                    idx_left,
                    idx_right,
                }));
            }

            if erase1 {
                line.remove(&x_left);
            }
            if erase2 {
                line.remove(&x_right);
            }
        }

        if !matches!(events.peek(), Some(Reverse(event_next)) if time == event_next.time) {
            // println!("{:?}", current_collisions);

            current_collisions.drain().for_each(|idx| {
                projectiles[idx].duration = time / 2;
            });
        }
    }
    // println!("{:?}", projectiles);

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum SweepEventType {
        Remove,
        Add,
    }

    let mut segment_events = Vec::new();
    let mut xs = Vec::new();
    for Projectile {
        x,
        y,
        dir,
        duration,
    } in projectiles
    {
        let mut rect = |x_start: i32, x_end: i32, y_start: i32, y_end: i32| {
            xs.push(x_start - 1);
            xs.push(x_end);
            segment_events.push((SweepEventType::Add, x_start - 1, x_end, y_start - 1));
            segment_events.push((SweepEventType::Remove, x_start - 1, x_end, y_end));
        };

        if duration == i32::MAX {
            match dir {
                Direction::Down => rect(x, x, y, height),
                Direction::Up => rect(x, x, 1, y),
                Direction::Right => rect(x, width, y, y),
                Direction::Left => rect(1, x, y, y),
            };
        } else {
            match dir {
                Direction::Down => rect(x, x, y, y + duration),
                Direction::Up => rect(x, x, y - duration, y),
                Direction::Right => rect(x, x + duration, y, y),
                Direction::Left => rect(x - duration, x, y, y),
            };
        }
    }

    xs.sort_unstable();
    xs.dedup();

    segment_events.sort_unstable_by_key(|&(.., y)| y);
    // println!("{:?}", segment_events);

    let mut segtree = SegTree::new(&xs);
    let mut result: u64 = 0;
    let mut segment_events = segment_events.into_iter().peekable();

    while let Some((event_type, x_start, x_end, y)) = segment_events.next() {
        let i_start = xs.binary_search(&x_start).unwrap();
        let i_end = xs.binary_search(&x_end).unwrap();

        segtree.add_range(
            i_start..i_end,
            match event_type {
                SweepEventType::Add => 1,
                SweepEventType::Remove => -1,
            },
        );

        match segment_events.peek() {
            Some(&(.., y_next)) if y != y_next => {
                result += (y_next - y) as u64 * segtree.sum();
            }
            _ => {}
        }
    }

    writeln!(output_buf, "{}", result).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
