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
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
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

    // 3-face of tesseract
    // basis representation:
    // (e1, e2, e3, e4, -e1, -e2, -e3, -e4)
    type Cell = i8;

    // base point with three orientation axes
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct OrientedCell([Cell; 4]);

    // three generators of tesseract's symmetry group,
    // and its inverses:
    // (u1, u2, u3, -u1, -u2, -u3)
    #[derive(Debug, Copy, Clone)]
    struct Direction(i8);
    impl Direction {
        // group action
        fn rotate(self, position: OrientedCell) -> OrientedCell {
            let OrientedCell([p, i, j, k]) = position;
            OrientedCell(match self {
                Direction(1) => [i, -p, j, k],
                Direction(-1) => [-i, p, j, k],
                Direction(2) => [j, i, -p, k],
                Direction(-2) => [-j, i, p, k],
                Direction(3) => [k, i, j, -p],
                Direction(-3) => [-k, i, j, p],
                _ => panic!(),
            })
        }

        fn translate_planar(self, position: [usize; 3]) -> [usize; 3] {
            let [x, y, z] = position;
            match self {
                Direction(1) => [x + 1, y, z],
                Direction(-1) => [x - 1, y, z],
                Direction(2) => [x, y + 1, z],
                Direction(-2) => [x, y - 1, z],
                Direction(3) => [x, y, z + 1],
                Direction(-3) => [x, y, z - 1],
                _ => panic!(),
            }
        }
    }

    fn directions() -> impl Iterator<Item = Direction> {
        (1..=3).chain((-3..=-1).rev()).map(|i| Direction(i))
    }

    let nx: usize = input.value();
    let ny: usize = input.value();
    let nz: usize = input.value();
    type Grid<T> = [[[T; 10]; 10]; 10];
    let mut grid = [[[false; 10]; 10]; 10];
    for z in 1..=nz {
        for y in 1..=ny {
            for (x, &c) in input.token()[..nx].into_iter().enumerate() {
                let x = x + 1;
                grid[x][y][z] = match c {
                    b'.' => false,
                    b'x' => true,
                    _ => panic!(),
                }
            }
        }
    }

    use std::collections::HashSet;
    let mut cell_visited = HashSet::new();
    let mut planar_visited = [[[false; 10]; 10]; 10];
    let cell_start = OrientedCell([1, 2, 3, 4]);
    cell_visited.insert(cell_start.0[0]);
    
    let [x0, y0, z0] = (1..=nx)
        .flat_map(|x| (1..=ny).flat_map(move |y| (1..=nz).map(move |z| [x, y, z])))
        .find(|&[x, y, z]| grid[x][y][z])
        .unwrap();
    planar_visited[x0][y0][z0] = true;

    fn dfs(
        grid: &Grid<bool>,
        cell_visited: &mut HashSet<Cell>,
        planar_visited: &mut Grid<bool>,
        cell_position: OrientedCell,
        planar_position: [usize; 3],
    ) -> bool {
        for dir in directions() {
            let cell_next = dir.rotate(cell_position);
            let planar_next = dir.translate_planar(planar_position);
            let [nx, ny, nz] = planar_next;
            if grid[nx][ny][nz] && !planar_visited[nx][ny][nz] 
            {
                if cell_visited.contains(&cell_next.0[0]) {
                    return false;
                }
                
                cell_visited.insert(cell_next.0[0]);
                planar_visited[nx][ny][nz] = true;

                // println!("{:?}", (cell_next, planar_next));
                dfs(grid, cell_visited, planar_visited, cell_next, planar_next);
                // println!("pop");
            }
        }

        true
    }
    let mut result = dfs(
        &grid,
        &mut cell_visited,
        &mut planar_visited,
        cell_start,
        [x0, y0, z0],
    );
    result = result && cell_visited.len() == 8;

    if result {
        println!("Yes");
    } else {
        println!("No");
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
