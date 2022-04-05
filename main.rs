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

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::collections::*;
    use std::iter::*;
    use std::mem::swap;

    let mut height: usize = input.value();
    let mut width: usize = input.value();
    let mut obstacle: Vec<bool> = (0..height)
        .flat_map(|_| input.token()[..width].to_vec())
        .map(|c| c == b'#')
        .collect();

    if width > height {
        obstacle = (0..width)
            .flat_map(|i| obstacle[i..].iter().step_by(width).take(height).copied())
            .collect();
        swap(&mut width, &mut height);
    }

    let width_u32 = width as u32;

    // connection state
    #[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct State(u32);
    mod flag {
        pub const HORIZONTAL: u32 = 0b01;
        pub const VERTICAL: u32 = 0b10;
    }

    impl State {
        fn empty() -> Self {
            Self(0)
        }

        #[inline]
        fn get(self, i: u32) -> u32 {
            (self.0 >> (i * 3)) & 0b111
        }

        // state of last inserted cell
        fn get_flag(self) -> u32 {
            (self.0 >> 30) & 0b11
        }

        #[inline]
        fn set(self, i: u32, value: u32) -> Self {
            debug_assert!(value < 8);
            let mask = !(0b111 << (i * 3));
            Self(self.0 & mask | (value << (i * 3)))
        }

        fn normalize(self, width: u32) -> Self {
            let mut value_map = [0; 8];
            let mut n_values = 0;
            Self(
                (0..width)
                    .rev()
                    .map(|i| self.get(i) as usize)
                    .map(|value| {
                        if value == 0 {
                            0
                        } else if value_map[value] == 0 {
                            n_values += 1;
                            value_map[value] = n_values;
                            n_values
                        } else {
                            value_map[value]
                        }
                    })
                    .fold(0, |acc, value| (acc << 3) + value)
                    | (self.0 & (0b11 << 30)),
            )
        }

        #[inline]
        fn count_normalized(self, width: u32) -> u32 {
            debug_assert!(self == self.normalize(width));
            (0..width).map(|i| self.get(i)).max().unwrap()
        }

        #[inline]
        fn insert(mut self, width: u32, component: u32, flag: u32) -> Self {
            debug_assert!(component < 8 && flag < 4);
            if self.get_flag() & flag::VERTICAL == 0 {
                self = self.set(0, 0);
            }
            let mask = (1 << ((width - 1) * 3)) - 1;
            Self(((self.0 & mask) << 3) | component | (flag << 30))
        }
    }

    use std::fmt;
    impl fmt::Debug for State {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            (0..10)
                .rev()
                .map(|i| self.get(i))
                .try_for_each(|value| write!(f, "{}", value))?;
            write!(f, "|{:02b}", self.get_flag())?;
            Ok(())
        }
    }

    type Cache = HashMap<State, u32>;
    let [mut dp_prev, mut dp]: [Cache; 2] = Default::default();
    dp_prev.insert(State::empty().set(width_u32 - 1, 1), 1);

    const P: u32 = 10007;
    for i in 0..width * height {
        for (&prev_state, &prev_value) in &dp_prev {
            let is_left_end = i % width == 0;
            let is_right_end = i % width == width - 1;
            let component_up = prev_state.get(width_u32 - 1);
            let component_left = if is_left_end || prev_state.get_flag() & flag::HORIZONTAL == 0 {
                0
            } else {
                prev_state.get(0)
            };

            let add_pipe =
                |component: u32, flag: u32| Some(prev_state.insert(width_u32, component, flag));
            let add_disconnected = |component, flag| {
                (component_up == 0 || (0..width_u32 - 1).any(|i| prev_state.get(i) == component_up))
                    .then(|| prev_state.insert(width_u32, component, flag))
            };
            let blank = || add_disconnected(0, 0);
            let merge = || {
                (component_left != component_up).then(|| {
                    let state = prev_state.insert(width_u32, component_up, 0);
                    (1..width_u32)
                        .filter(|&i| state.get(i) == component_left)
                        .fold(state, |acc, i| acc.set(i, component_up))
                })
            };

            let mut update = |state: Option<State>| {
                state.map(|state| {
                    dp.entry(state.normalize(width_u32))
                        .and_modify(|x| {
                            *x = *x + prev_value;
                            if *x >= P {
                                *x -= P;
                            }
                        })
                        .or_insert(prev_value)
                });
            };

            match (obstacle[i], component_left, component_up) {
                (true, 0, 0) => update(blank()),
                (true, ..) => {}
                (false, 0, 0) => {
                    update(blank());
                    if !is_right_end {
                        update(add_disconnected(
                            prev_state.count_normalized(width_u32) + 1,
                            flag::HORIZONTAL | flag::VERTICAL,
                        ));
                    }
                }
                (false, 0, _) => {
                    update(add_pipe(component_up, flag::VERTICAL));
                    if !is_right_end {
                        update(add_pipe(component_up, flag::HORIZONTAL));
                    }
                }
                (false, _, 0) => {
                    update(add_pipe(component_left, flag::VERTICAL));
                    if !is_right_end {
                        update(add_pipe(component_left, flag::HORIZONTAL));
                    }
                }
                (false, _, _) => update(merge()),
            }
        }

        // println!("{:?}: {:?}", i, dp);

        swap(&mut dp, &mut dp_prev);
        dp.clear();
    }

    let result = dp_prev
        .get(&State::empty().insert(width_u32, 1, flag::VERTICAL))
        .copied()
        .unwrap_or(0);
    writeln!(output_buf, "{:?}", result).unwrap();
    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
