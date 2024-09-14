mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
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
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
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

    pub trait ReadValue<T> {
        fn value(&mut self) -> T;
    }

    impl<T: FromStr, I: InputStream> ReadValue<T> for I
    where
        T::Err: Debug,
    {
        #[inline]
        fn value(&mut self) -> T {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }
}

// https://www.acmicpc.net/problem/2580
// sudoku
use std::io::{BufReader, Read};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

mod sudoku {
    struct Env {
        neighbors: Vec<Vec<u32>>,
        candidate_buf: Vec<[bool; 10]>,
    }

    fn elim_candidates(env: &mut Env, board: &Vec<u32>) {
        for cell in 0..81 {
            if board[cell] == 0 {
                for i in 1..=9 {
                    env.candidate_buf[cell][i] = true;
                }

                for neighbor in &env.neighbors[cell] {
                    if board[*neighbor as usize] != 0 {
                        env.candidate_buf[cell][board[*neighbor as usize] as usize] = false;
                    }
                }
            }
        }
    }

    pub fn solve(board: &Vec<u32>) -> Option<Vec<u32>> {
        fn num_candidates(env: &Env, cell: usize) -> u32 {
            env.candidate_buf[cell][1..=9]
                .iter()
                .map(|x| *x as u32)
                .sum()
        }

        let mut env = Env {
            neighbors: (0..81)
                .map(|cell| {
                    let (x, y) = (cell % 9, cell / 9);
                    (0..81)
                        .filter_map(|neighbor| {
                            let (nx, ny) = (neighbor % 9, neighbor / 9);
                            if x == nx || y == ny || (x / 3, y / 3) == (nx / 3, ny / 3) {
                                Some(neighbor)
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect(),
            candidate_buf: (0..81).map(|_| [false; 10]).collect(),
        };

        fn dfs(env: &mut Env, board: &mut Vec<u32>) -> bool {
            elim_candidates(env, &board);
            match (0..81)
                .filter(|&cell| board[cell] == 0)
                .map(|cell| (cell, num_candidates(&env, cell)))
                .min_by_key(|(_, n)| n.clone())
            {
                None => true,
                Some((_, 0)) => false,
                Some((target_cell, _)) => {
                    for val in 1..=9 {
                        if env.candidate_buf[target_cell][val as usize] {
                            board[target_cell] = val;
                            if dfs(env, board) {
                                return true;
                            }
                        }
                    }
                    board[target_cell] = 0;
                    false
                }
            }
        }

        let mut board = board.clone();
        if dfs(&mut env, &mut board) {
            Some(board)
        } else {
            None
        }
    }
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let board: Vec<u32> = (0..81).map(|_| input.value()).collect();
    let board = sudoku::solve(&board).unwrap();

    let board_str = (0..9)
        .map(|y| {
            (0..9)
                .map(|x| board[x + 9 * y].to_string())
                .collect::<Vec<String>>()
                .join(" ")
        })
        .collect::<Vec<String>>()
        .join("\n");
    println!("{}", board_str);
}
