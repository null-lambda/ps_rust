use std::collections::HashSet;

#[allow(dead_code)]
mod fast_io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        fn value<T: FromStr>(&mut self) -> T
        where
            <T as FromStr>::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    // cheap and unsafe whitespace check
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

    use std::io::{BufRead, BufReader, BufWriter, Read, Stdin, Stdout};

    pub struct InputAtOnce {
        buf: Box<[u8]>,
        cursor: usize,
    }

    impl<'a> InputAtOnce {
        pub fn new(buf: Box<[u8]>) -> Self {
            Self { buf, cursor: 0 }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.buf.len() - self.cursor);
            let slice = &self.buf[self.cursor..self.cursor + n];
            self.cursor += n;
            slice
        }
    }

    impl<'a> InputStream for InputAtOnce {
        fn token(&mut self) -> &[u8] {
            self.take(
                self.buf[self.cursor..]
                    .iter()
                    .position(|&c| !is_whitespace(c))
                    .expect("no available tokens left"),
            );
            self.take(
                self.buf[self.cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.buf.len() - self.cursor),
            )
        }

        fn line(&mut self) -> &[u8] {
            let line = self.take(
                self.buf[self.cursor..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map(|idx| idx + 1)
                    .unwrap_or_else(|| self.buf.len() - self.cursor),
            );
            trim_newline(line)
        }
    }

    pub struct LineSyncedInput<R: BufRead> {
        line_buf: Vec<u8>,
        line_cursor: usize,
        inner: R,
    }

    impl<R: BufRead> LineSyncedInput<R> {
        pub fn new(r: R) -> Self {
            Self {
                line_buf: Vec::new(),
                line_cursor: 0,
                inner: r,
            }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.line_buf.len() - self.line_cursor);
            let slice = &self.line_buf[self.line_cursor..self.line_cursor + n];
            self.line_cursor += n;
            slice
        }

        fn eol(&self) -> bool {
            self.line_cursor == self.line_buf.len()
        }

        fn refill_line_buf(&mut self) -> bool {
            self.line_buf.clear();
            self.line_cursor = 0;
            let result = self.inner.read_until(b'\n', &mut self.line_buf).is_ok();
            result
        }
    }

    impl<R: BufRead> InputStream for LineSyncedInput<R> {
        fn token(&mut self) -> &[u8] {
            loop {
                if self.eol() {
                    let b = self.refill_line_buf();
                    if !b {
                        panic!(); // EOF
                    }
                }
                self.take(
                    self.line_buf[self.line_cursor..]
                        .iter()
                        .position(|&c| !is_whitespace(c))
                        .unwrap_or_else(|| self.line_buf.len() - self.line_cursor),
                );

                let idx = self.line_buf[self.line_cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.line_buf.len() - self.line_cursor);
                if idx > 0 {
                    return self.take(idx);
                }
            }
        }

        fn line(&mut self) -> &[u8] {
            if self.eol() {
                self.refill_line_buf();
            }

            self.line_cursor = self.line_buf.len();
            trim_newline(self.line_buf.as_slice())
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let mut reader = BufReader::new(std::io::stdin().lock());
        let mut buf: Vec<u8> = vec![];
        reader.read_to_end(&mut buf).unwrap();
        let buf = buf.into_boxed_slice();
        InputAtOnce::new(buf)
    }

    // pub fn stdin_buf() -> LineSyncedInput<BufReader<StdinLock<'static>>> {
    //     LineSyncedInput::new(BufReader::new(std::io::stdin().lock()))
    // }

    // no lock
    pub fn stdin_buf() -> LineSyncedInput<BufReader<Stdin>> {
        LineSyncedInput::new(BufReader::new(std::io::stdin()))
    }

    pub fn stdout() -> Stdout {
        std::io::stdout()
    }

    pub fn stdout_buf() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

// use std::
type Num = i16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Accessory {
    HR,
    RE,
    CO,
    EX,
    DX,
    HU,
    CU,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AccessoryParseError;

impl std::str::FromStr for Accessory {
    type Err = AccessoryParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "HR" => Ok(Accessory::HR),
            "RE" => Ok(Accessory::RE),
            "CO" => Ok(Accessory::CO),
            "EX" => Ok(Accessory::EX),
            "DX" => Ok(Accessory::DX),
            "HU" => Ok(Accessory::HU),
            "CU" => Ok(Accessory::CU),
            _ => Err(AccessoryParseError),
        }
    }
}

#[derive(Debug, Clone)]
struct Stat {
    atk: Num,
    def: Num,
    hp: Num,
    hp_max: Num,
    exp: Num,
    lv: Num,
}

impl Stat {
    fn heal(&mut self, amount: Num) {
        self.hp = (self.hp + amount).min(self.hp_max);
    }

    fn heal_full(&mut self) {
        self.hp = self.hp_max;
    }

    fn gain_true_dmg(&mut self, dmg: Num) -> bool {
        self.hp -= dmg;
        self.hp <= 0
    }
}

#[derive(Debug, Clone)]
enum Cell {
    Chest(ChestContent),
    Monster {
        boss: bool,
        name: String,
        stat: Stat,
    },
    Wall,
    Spike,
    Empty,
}

#[derive(Debug, Clone)]
enum ChestContent {
    Atk(Num),
    Def(Num),
    Accessory(Accessory),
}

#[derive(Debug, Clone)]
struct Player {
    stat: Stat,
    modifiers: Modifiers,
}

#[derive(Debug, Clone, Default)]
struct Modifiers {
    atk: Num,
    def: Num,
    accessories: HashSet<Accessory>,
}

impl Player {
    fn max_exp(&self) -> Num {
        5 * self.stat.lv
    }

    fn gain_exp(&mut self, exp: Num) {
        let mut true_exp = exp;
        if self.modifiers.accessories.contains(&Accessory::EX) {
            true_exp = (true_exp as f64 * 1.2 + 1e-8) as Num;
        }

        self.stat.exp += true_exp;
        if self.stat.exp >= self.max_exp() {
            // level up
            self.stat.exp = 0;
            self.stat.lv += 1;

            self.stat.hp_max += 5;
            self.stat.atk += 2;
            self.stat.def += 2;
            self.stat.heal_full();
        }
    }
}

enum GameEndReason {
    Win,
    Killed { by: String },
    OutOfTurns,
}

impl std::fmt::Display for GameEndReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameEndReason::Win => write!(f, "YOU WIN!"),
            GameEndReason::Killed { by } => write!(f, "YOU HAVE BEEN KILLED BY {}..", by),
            GameEndReason::OutOfTurns => write!(f, "Press any key to continue."),
        }
    }
}

fn main() {
    use fast_io::InputStream;
    use std::io::Write;

    // let mut input = fast_io::stdin_buf();
    let mut input = fast_io::stdin_at_once();
    // let mut output = fast_io::stdout_buf();
    let mut output = std::io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    input.skip_line();

    // read inputs
    // n x m grid
    let grid_bytes: Vec<u8> = (0..n)
        .flat_map(|_| input.line()[..m].to_vec().into_iter())
        .collect();
    let moves = input.line().to_vec();

    let idx_sub1 = |idx: usize| idx - 1;
    let idx = |r: usize, c: usize| r * m + c;
    let coord = |idx: usize| (idx / m, idx % m);

    let boss_pos = grid_bytes.iter().position(|&c| c == b'M').unwrap();
    let player_pos = grid_bytes.iter().position(|&c| c == b'@').unwrap();
    let n_boss = 1;
    let n_monster = grid_bytes.iter().filter(|&&c| c == b'&').count() + n_boss;
    let n_chest = grid_bytes.iter().filter(|&&c| c == b'B').count();

    let mut grid: Vec<Cell> = grid_bytes
        .iter()
        .map(|&c| match c {
            b'@' | b'.' => Cell::Empty,
            b'&' | b'M' | b'B' => Cell::Empty, // initialize later
            b'#' => Cell::Wall,
            b'^' => Cell::Spike,
            _ => panic!(),
        })
        .collect();

    for _ in 0..n_monster {
        let r = idx_sub1(input.value());
        let c = idx_sub1(input.value());
        let name = input.value();
        let atk = input.value();
        let def = input.value();
        let hp = input.value();
        let exp = input.value();

        grid[idx(r, c)] = Cell::Monster {
            boss: idx(r, c) == boss_pos,
            name,
            stat: Stat {
                atk,
                def,
                hp,
                hp_max: hp,
                exp,
                lv: -1,
            },
        };
    }

    for _ in 0..n_chest {
        // r, c, t
        let r = idx_sub1(input.value());
        let c = idx_sub1(input.value());
        let t: String = input.value();
        grid[idx(r, c)] = Cell::Chest(match t.as_str() {
            "W" => ChestContent::Atk(input.value()),
            "A" => ChestContent::Def(input.value()),
            "O" => ChestContent::Accessory(input.value()),
            _ => panic!(),
        });
    }

    drop(input);
    drop(grid_bytes);

    let mut player = Player {
        stat: Stat {
            atk: 2,
            def: 2,
            hp: 20,
            hp_max: 20,
            exp: 0,
            lv: 1,
        },
        modifiers: Modifiers {
            atk: 0,
            def: 0,
            accessories: Default::default(),
        },
    };

    fn move_player(
        n: usize,
        m: usize,
        grid: &mut Vec<Cell>,
        player: &mut Player,
        pos_init: usize,
        pos: &mut usize,
        dir: u8,
    ) -> Result<(), GameEndReason> {
        let idx = |r: usize, c: usize| r * m + c;
        let coord = |idx: usize| (idx / m, idx % m);

        let pos_next = || -> Option<usize> {
            let (pr, pc) = coord(*pos);
            let pos_next = match dir {
                b'U' if pr > 0 => Some(idx(pr - 1, pc)),
                b'D' if pr < n - 1 => Some(idx(pr + 1, pc)),
                b'L' if pc > 0 => Some(idx(pr, pc - 1)),
                b'R' if pc < m - 1 => Some(idx(pr, pc + 1)),
                _ => None, // if out of bounds, do nothing
            }?;

            // if wall, do nothing
            if let Cell::Wall = grid[pos_next] {
                return None;
            }

            Some(pos_next)
        };

        pos_next().map(|pos_next| *pos = pos_next);

        let cell = &mut grid[*pos];
        match cell {
            Cell::Chest(content) => {
                match content {
                    ChestContent::Atk(atk) => {
                        player.modifiers.atk = *atk;
                    }
                    ChestContent::Def(def) => {
                        player.modifiers.def = *def;
                    }
                    ChestContent::Accessory(accessory) => {
                        if player.modifiers.accessories.len() < 4 {
                            player.modifiers.accessories.insert(*accessory);
                        }
                    }
                }
                *cell = Cell::Empty;
                Ok(())
            }
            Cell::Monster {
                boss,
                name: monster_name,
                stat: monster_stat,
            } => {
                let boss = *boss;
                // println!("monster encounter: {:?}", (&boss, &monster_name, &monster_stat));

                fn calc_dmg(
                    battle_turn: u32,
                    src: &Stat,
                    src_modifiers: &Modifiers,
                    dst: &Stat,
                    dst_modifiers: &Modifiers,
                ) -> Num {
                    let mut atk = src.atk + src_modifiers.atk;
                    let def = dst.def + dst_modifiers.def;
                    if battle_turn == 0 {
                        if src_modifiers.accessories.contains(&Accessory::CO) {
                            atk *= if src_modifiers.accessories.contains(&Accessory::DX) {
                                3
                            } else {
                                2
                            };
                        }
                    }
                    (atk - def).max(1)
                }

                let mut battle_turn = 0;
                loop {
                    let hu_active = battle_turn == 0
                        && boss
                        && player.modifiers.accessories.contains(&Accessory::HU);
                    if hu_active {
                        player.stat.heal_full();
                    }

                    let dmg = calc_dmg(
                        battle_turn,
                        &player.stat,
                        &player.modifiers,
                        monster_stat,
                        &Modifiers::default(),
                    );
                    if monster_stat.gain_true_dmg(dmg) {
                        player.gain_exp(monster_stat.exp);

                        if player.modifiers.accessories.contains(&Accessory::HR) {
                            player.stat.heal(3);
                        }

                        *cell = Cell::Empty;
                        if boss {
                            return Err(GameEndReason::Win);
                        } else {
                            return Ok(());
                        }
                    }

                    let mut dmg = calc_dmg(
                        battle_turn,
                        monster_stat,
                        &Modifiers::default(),
                        &player.stat,
                        &player.modifiers,
                    );
                    if hu_active {
                        dmg = 0;
                    }
                    if player.stat.gain_true_dmg(dmg) {
                        if player.modifiers.accessories.remove(&Accessory::RE) {
                            player.stat.heal_full();
                            monster_stat.heal_full();
                            *pos = pos_init;
                            return Ok(());
                        } else {
                            return Err(GameEndReason::Killed {
                                by: monster_name.clone(),
                            });
                        }
                    }

                    battle_turn += 1;
                }
            }
            Cell::Wall => unreachable!(),
            Cell::Spike => {
                let dmg = if player.modifiers.accessories.contains(&Accessory::DX) {
                    1
                } else {
                    5
                };
                if player.stat.gain_true_dmg(dmg) {
                    if player.modifiers.accessories.remove(&Accessory::RE) {
                        player.stat.heal_full();
                        *pos = pos_init;
                        return Ok(());
                    } else {
                        return Err(GameEndReason::Killed {
                            by: "SPIKE TRAP".to_string(),
                        });
                    }
                }

                Ok(())
            }
            Cell::Empty => Ok(()),
        }
    }

    // game loop
    let mut turns = 0;
    let player_pos_init = player_pos;
    let mut player_pos = player_pos_init;
    let game_loop = || -> GameEndReason {
        for dir in moves {
            turns += 1;
            match move_player(
                n,
                m,
                &mut grid,
                &mut player,
                player_pos_init,
                &mut player_pos,
                dir,
            ) {
                Ok(()) => (),
                Err(reason) => return reason,
            }
            // print_state(&mut output, &grid, &player, player_pos, turns, None, m);
            // writeln!(output).unwrap();
        }
        GameEndReason::OutOfTurns
    };
    let ended_by = game_loop();

    // print final states

    fn print_state<W: std::io::Write>(
        output: &mut W,
        grid: &[Cell],
        player: &Player,
        player_pos: usize,
        turns: usize,
        ended_by: Option<GameEndReason>,
        m: usize,
    ) {
        let mut grid_bytes = grid
            .iter()
            .map(|cell| match cell {
                Cell::Empty => b'.',
                Cell::Wall => b'#',
                Cell::Spike => b'^',
                Cell::Monster { boss: true, .. } => b'M',
                Cell::Monster { boss: false, .. } => b'&',
                Cell::Chest(_) => b'B',
            })
            .collect::<Vec<u8>>();

        if !matches!(ended_by, Some(GameEndReason::Killed { .. })) {
            grid_bytes[player_pos] = b'@';
        }

        for row in grid_bytes.chunks(m) {
            writeln!(output, "{}", std::str::from_utf8(row).unwrap()).unwrap();
        }
        writeln!(output, "Passed Turns : {}", turns).unwrap();

        // let Player { atk, def, hp, hp_max, exp, lv, modifiers } = &player;
        let Player {
            stat:
                Stat {
                    atk,
                    def,
                    hp,
                    hp_max,
                    exp,
                    lv,
                },
            modifiers,
        } = &player;
        writeln!(output, "LV : {}", lv).unwrap();
        writeln!(output, "HP : {}/{}", (*hp).max(0), hp_max).unwrap();
        writeln!(output, "ATT : {}+{}", atk, modifiers.atk).unwrap();
        writeln!(output, "DEF : {}+{}", def, modifiers.def).unwrap();
        writeln!(output, "EXP : {}/{}", exp, player.max_exp()).unwrap();

        if let Some(ended_by) = ended_by {
            write!(output, "{}", ended_by).unwrap();
        }

        // writeln!(output, "-- Accessories : {}", modifiers.accessories.iter().map(|acc| format!("{:?}", acc)).collect::<Vec<String>>().join(", ")).unwrap();
    }

    print_state(
        &mut output,
        &grid,
        &player,
        player_pos,
        turns,
        Some(ended_by),
        m,
    );

    // for row in grid.chunks(m) {
    //     for cell in row {
    //         print!("{:?} ", cell);
    //     }
    //     println!();
    // }

    output.flush().unwrap();
}
