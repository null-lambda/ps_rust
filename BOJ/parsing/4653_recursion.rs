use std::io::Write;

use simple_io::stdin;

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

#[derive(Debug)]
enum Node {
    Branch(u8, usize, usize),
    Leaf(u8),
}

#[derive(Debug, Default, Clone)]
struct NodeData {
    width: u32,
    height: u32,
    x: u32,
    y: u32,
}

fn parse_rec(flat_ast: &mut Vec<Node>, bytes: &mut impl Iterator<Item = u8>) -> usize {
    let b = bytes.next().unwrap();
    let idx;
    match b {
        b'A'..=b'Z' => {
            idx = flat_ast.len();
            flat_ast.push(Node::Leaf(b));
        }
        b'|' | b'-' => {
            let left = parse_rec(flat_ast, bytes);
            let right = parse_rec(flat_ast, bytes);
            idx = flat_ast.len();
            flat_ast.push(Node::Branch(b, left, right));
        }
        _ => panic!(),
    }
    idx
}

fn init_size_rec(flat_ast: &[Node], dp: &mut [NodeData], u: usize) {
    match flat_ast[u] {
        Node::Leaf(_) => {
            dp[u].height = 2;
            dp[u].width = 2;
        }
        Node::Branch(b'-', left, right) => {
            init_size_rec(flat_ast, dp, left);
            init_size_rec(flat_ast, dp, right);
            dp[u].width = dp[left].width.max(dp[right].width);
            dp[u].height = dp[left].height + dp[right].height;
        }
        Node::Branch(b'|', left, right) => {
            init_size_rec(flat_ast, dp, left);
            init_size_rec(flat_ast, dp, right);
            dp[u].width = dp[left].width + dp[right].width;
            dp[u].height = dp[left].height.max(dp[right].height);
        }
        _ => panic!(),
    }
}

fn stretch_rec(flat_ast: &[Node], dp: &mut [NodeData], u: usize, x: u32, y: u32) {
    dp[u].x = x;
    dp[u].y = y;
    match flat_ast[u] {
        Node::Leaf(_) => {}
        Node::Branch(b'-', left, right) => {
            let inner_height = dp[left].height + dp[right].height;
            dp[left].height = dp[u].height * dp[left].height / inner_height;
            dp[right].height = dp[u].height * dp[right].height / inner_height;
            if dp[left].height + dp[right].height < dp[u].height {
                dp[left].height += 1;
            }
            assert!(dp[left].height + dp[right].height == dp[u].height);

            dp[left].width = dp[u].width;
            dp[right].width = dp[u].width;

            stretch_rec(flat_ast, dp, left, x, y);
            stretch_rec(flat_ast, dp, right, x, y + dp[left].height);
        }
        Node::Branch(b'|', left, right) => {
            let inner_width = dp[left].width + dp[right].width;
            dp[left].width = dp[u].width * dp[left].width / inner_width;
            dp[right].width = dp[u].width * dp[right].width / inner_width;
            if dp[left].width + dp[right].width < dp[u].width {
                dp[left].width += 1;
            }
            assert!(dp[left].width + dp[right].width == dp[u].width);

            dp[left].height = dp[u].height;
            dp[right].height = dp[u].height;

            stretch_rec(flat_ast, dp, left, x, y);
            stretch_rec(flat_ast, dp, right, x + dp[left].width, y);
        }
        _ => panic!(),
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for i_tc in 1..=input.value() {
        writeln!(output, "{i_tc}").unwrap();
        let preorder = input.token().bytes();

        let mut flat_ast = vec![];
        let root = parse_rec(&mut flat_ast, &mut preorder.into_iter());

        let n = flat_ast.len();
        let mut dp = vec![NodeData::default(); n];
        init_size_rec(&flat_ast, &mut dp, root);
        stretch_rec(&flat_ast, &mut dp, root, 0, 0);

        let ch = dp[root].height + 1;
        let cw = dp[root].width + 1;
        let mut canvas = vec![vec![' '; cw as usize]; ch as usize];

        // Render frame
        for x in 1..cw {
            canvas[ch as usize - 1][x as usize] = '-';
        }
        for y in 1..ch {
            canvas[y as usize][cw as usize - 1] = '|';
        }
        for u in (0..n).rev() {
            for x in dp[u].x + 1..dp[u].x + dp[u].width {
                canvas[dp[u].y as usize][x as usize] = '-';
            }
            for y in dp[u].y + 1..dp[u].y + dp[u].height {
                canvas[y as usize][dp[u].x as usize] = '|';
            }
            canvas[(dp[u].y + dp[u].height) as usize][(dp[u].x + dp[u].width) as usize] = '*';
        }
        canvas[dp[root].y as usize][(dp[root].x + dp[root].width) as usize] = '*';
        canvas[(dp[root].y + dp[root].height) as usize][dp[root].x as usize] = '*';

        // Render characters
        for u in 0..n {
            if let Node::Leaf(c) = flat_ast[u] {
                let (x, y) = (dp[u].x, dp[u].y);
                canvas[y as usize][x as usize] = c as char;
            }
        }

        for row in canvas {
            for c in row {
                write!(output, "{}", c).unwrap();
            }
            writeln!(output).unwrap();
        }
    }
}
