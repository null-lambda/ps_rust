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
    use std::mem::swap;

    let mut height: usize = input.value();
    let mut width: usize = input.value();
    let mut grid: Vec<i32> = (0..width * height).map(|_| input.value()).collect();

    if height < width {
        grid = (0..width)
            .flat_map(|i| grid[i..].iter().step_by(width).take(height).copied())
            .collect();
        swap(&mut width, &mut height);
    }

    let width_u32 = width as u32;

    // connection state
    #[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct State(u32);

    impl State {
        fn empty() -> Self {
            Self(0)
        }

        #[inline]
        fn get(self, i: u32) -> u32 {
            (self.0 >> (i * 3)) & 0b111
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
                    .fold(0, |acc, value| (acc << 3) + value),
            )
        }

        #[inline]
        fn count_normalized(self, width: u32) -> u32 {
            debug_assert!(self == self.normalize(width));
            (0..width).map(|i| self.get(i)).max().unwrap()
        }

        #[inline]
        fn insert(self, width: u32, value: u32) -> Self {
            debug_assert!(value < 8);
            let mask = (1 << ((width - 1) * 3)) - 1;
            Self(((self.0 & mask) << 3) | value)
        }

        #[inline]
        fn checked_insert(self, width: u32, value: u32) -> Option<Self> {
            debug_assert!(value < 8);
            let component_erased = self.get(width - 1);
            (component_erased == 0 || (0..width - 1).any(|i| self.get(i) == component_erased))
                .then(|| self.insert(width, value))
        }
    }

    use std::fmt;
    impl fmt::Debug for State {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            (0..10)
                .rev()
                .map(|i| self.get(i))
                .try_for_each(|value| write!(f, "{}", value))
        }
    }

    let states: Vec<_> = (0..width_u32)
        .map(|idx_newline| {
            fn dfs_states(
                width: u32,
                idx_newline: u32,
                state: State,
                i: u32,
                f: &mut impl FnMut(State),
            ) {
                if i >= width {
                    debug_assert!(state == state.normalize(width));
                    f(state);
                    return;
                }

                for value in 1..=7.min(state.normalize(width).count_normalized(width) + 1) {
                    for len in 1..=width - i {
                        let state_next =
                            (0..len).fold(state, |state, _| state.insert(width, value));
                        let len2_min = if i + len + idx_newline == width { 0 } else { 1 };
                        for len2 in len2_min..=width - i - len + 1 {
                            let state_next = (0..len2.min(width - i - len))
                                .fold(state_next, |state, _| state.insert(width, 0));
                            dfs_states(width, idx_newline, state_next, i + len + len2, f);
                        }
                    }
                }
            }
            let mut states = vec![];
            for i in 0..=width_u32 {
                dfs_states(width_u32, idx_newline, State::empty(), i, &mut |state| {
                    states.push(state.normalize(width_u32));
                });
            }
            states.sort_unstable();
            states.dedup();
            states
        })
        .collect();

    type Cache = HashMap<State, i32>;
    let mut dp: [Cache; 2] = Default::default();
    dp[1].insert(State::empty(), 0);

    let mut result = 0;
    for i in 0..width * height {
        // bypass borrow checker
        let (dp0, dp1) = dp.split_at_mut(1);
        let (dp_prev, dp) = if i % 2 == 0 {
            (&dp1[0], &mut dp0[0])
        } else {
            (&dp0[0], &mut dp1[0])
        };

        dp.clear();
        for (&prev_state, &prev_value) in dp_prev {
            let update = |dp: &mut Cache, state: Option<State>, value| {
                state.map(|state| {
                    dp.entry(state)
                        .and_modify(|x| *x = (*x).min(value))
                        .or_insert(value)
                });
            };

            // case 1: insert blank cell
            update(
                dp,
                prev_state
                    .checked_insert(width_u32, 0)
                    .map(|state| state.normalize(width_u32)),
                prev_value,
            );

            // case 2: select next cell
            let component_left = if i % width > 0 { prev_state.get(0) } else { 0 };
            let component_up = prev_state.get(width_u32 - 1);
            let state = (match (component_left, component_up) {
                (0, 0) => prev_state.checked_insert(width_u32, 7),
                (0, _) => Some(prev_state.insert(width_u32, component_up)),
                (_, 0) => Some(prev_state.insert(width_u32, component_left)),
                (_, _) => {
                    let mut state = prev_state.insert(width_u32, component_left);
                    for i in 0..width_u32 {
                        if state.get(i) == component_up {
                            state = state.set(i, component_left);
                        }
                    }
                    Some(state)
                }
            })
            .map(|state| state.normalize(width_u32));
            update(dp, state, prev_value + grid[i]);
        }
        /*
        println!("{:?} {:?}", (i / width, i % width), dp[i % window]);
        println!(
            "{:?}",
            dp[i % window]
                .iter()
                .filter_map(|(&state, &value)| (state.count_normalized(width_u32) <= 1)
                    .then(|| (value, state)))
                .min()
                .unwrap()
        );
        println!();
        */
        dp.iter().for_each(|(&state, &value)| {
            if state.count_normalized(width_u32) <= 1 {
                result = result.min(value);
            }
        });
    }
    writeln!(output_buf, "{:?}", result).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
