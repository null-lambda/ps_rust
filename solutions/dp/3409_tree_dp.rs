use std::io::{Read, Write};
use std::iter::Peekable;
use std::{collections::HashMap, io::BufRead};

#[derive(Debug, Default)]
struct Ctx<'a> {
    var_map: HashMap<&'a str, usize>,
    nodes_map: HashMap<usize, Node>,
    nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
enum Node {
    Add(usize, usize),
    Terminal(Vec<u8>),
}

fn parse_line<'a>(s: &'a str, ctx: &mut Ctx<'a>) {
    let Ctx {
        var_map, nodes_map, ..
    } = ctx;
    let mut get_var_id = |s: &'a str| {
        let idx = var_map.len();
        *var_map.entry(s).or_insert(idx)
    };

    let (lhs, rhs) = s.split_at(s.find('=').unwrap());
    let lhs = get_var_id(lhs.trim());
    let rhs = &rhs[1..];

    if let Some(i) = rhs.find('+') {
        let (a, b) = rhs.split_at(i);
        let b = &b[1..];

        let a = get_var_id(a.trim());
        let b = get_var_id(b.trim());
        nodes_map.insert(lhs, Node::Add(a, b));
    } else {
        nodes_map.insert(lhs, Node::Terminal(rhs.trim().as_bytes().to_vec()));
    }
}

impl<'a> Ctx<'a> {
    fn remap_nodes(&mut self) {
        let Ctx {
            var_map,
            nodes_map,
            nodes,
        } = self;

        *nodes = vec![Node::Terminal(vec![]); var_map.len()];
        for (lhs, node) in nodes_map.drain() {
            nodes[lhs] = node.clone();
        }
    }

    fn consume(&self, pattern: &mut Peekable<impl Iterator<Item = u8>>, lhs: usize) {
        let mut visited = vec![false; self.nodes.len()];
        self.consume_inner(&mut visited, pattern, lhs);
    }

    fn consume_inner(
        &self,
        visited: &mut [bool],
        pattern: &mut Peekable<impl Iterator<Item = u8>>,
        lhs: usize,
    ) {
        if visited[lhs] || pattern.peek().is_none() {
            return;
        }
        match &self.nodes[lhs] {
            Node::Add(a, b) => {
                visited[lhs] = true;
                self.consume_inner(visited, pattern, *a);
                self.consume_inner(visited, pattern, *b);
            }
            Node::Terminal(s) => {
                visited[lhs] = true;
                for i in 0..s.len() {
                    if pattern.peek().is_none() {
                        return;
                    }
                    if pattern.peek().unwrap() == &s[i] {
                        visited.fill(false);
                        pattern.next();
                    }
                }
            }
        }
    }
}

fn main() {
    let mut input_buf = vec![];
    std::io::stdin().read_to_end(&mut input_buf).unwrap();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let input = unsafe { std::str::from_utf8_unchecked(&input_buf) };
    let mut lines = input.lines();

    let t: usize = lines.next().unwrap().parse().unwrap();
    for _ in 0..t {
        let mut ctx = Ctx::default();

        let k: usize = lines.next().unwrap().parse().unwrap();

        for _ in 0..k {
            parse_line(lines.next().unwrap(), &mut ctx);
        }

        ctx.remap_nodes();
        let lhs = ctx.var_map[lines.next().unwrap().trim()];
        let mut pattern = lines.next().unwrap().bytes().peekable();
        ctx.consume(&mut pattern, lhs);

        if pattern.peek().is_none() {
            writeln!(output, "YES").unwrap();
        } else {
            writeln!(output, "NO").unwrap();
        }
    }
}
