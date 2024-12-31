use std::collections::HashMap;
use std::io::{BufRead, Write};

#[derive(Debug)]
enum Node {
    Rule(Vec<Vec<usize>>),
    Terminal(String),
}

#[derive(Debug)]
struct TauntDfa {
    nodes: Vec<Node>,
    state: Vec<usize>,
}

impl TauntDfa {
    fn from_bnf(bnf: &str) -> Self {
        let mut rules: Vec<(&str, Vec<Vec<&str>>)> = vec![];

        // parse bnf to rules
        for tokens in bnf.lines() {
            let (mut lhs, rhs) = tokens.split_once("::=").unwrap();
            lhs = lhs.trim();
            let rhs = rhs
                .split(|c| c == '|')
                .map(|s| {
                    let s = s.trim();
                    let mut group = vec![];
                    let mut start = 0;

                    for (i, c) in s.char_indices() {
                        if c == '<' {
                            if start != i {
                                group.push(&s[start..i]);
                            }
                            start = i;
                        } else if c == '>' {
                            group.push(&s[start..=i]);
                            start = i + 1;
                        }
                    }
                    if start < s.len() {
                        group.push(&s[start..]);
                    }
                    group
                })
                .collect();
            rules.push((lhs, rhs));
        }

        // convert rules to index-based table
        let mut str_to_idx: HashMap<String, usize> = Default::default();
        let mut nodes = vec![];
        let mut add_ident = |nodes: &mut Vec<Node>, s: &str, is_lhs: bool| -> usize {
            let idx = str_to_idx.len();
            *str_to_idx.entry(s.to_string()).or_insert_with(|| {
                nodes.push(match s.chars().next() {
                    Some('<') if is_lhs => Node::Rule(vec![]),
                    Some('<') => panic!(),
                    _ => Node::Terminal(s.to_string()),
                });
                idx
            })
        };

        for (lhs, _) in &rules {
            add_ident(&mut nodes, lhs, true);
        }

        for (i, (_, rhs)) in rules.iter().enumerate() {
            let rules: Vec<Vec<usize>> = rhs
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|s| add_ident(&mut nodes, s, false))
                        .collect()
                })
                .collect();
            match &mut nodes[i] {
                Node::Rule(xs) => *xs = rules,
                _ => panic!(),
            }
        }

        let state = vec![0; nodes.len()];

        Self { nodes, state }
    }

    fn step(&mut self, output: &mut impl std::io::Write, taunt_count: &mut usize) {
        write!(output, "Taunter: ").unwrap();
        Self::step_inner(
            &self.nodes,
            &mut self.state,
            output,
            taunt_count,
            0,
            &mut true,
        );
        write!(output, ".\n").unwrap();
    }

    fn step_inner(
        nodes: &[Node],
        state: &mut [usize],
        output: &mut impl std::io::Write,
        taunt_count: &mut usize,
        idx: usize,
        first_word: &mut bool,
    ) {
        if idx == 0 {
            if *taunt_count > 0 {
                *taunt_count -= 1;
            }
        }
        match &nodes[idx] {
            Node::Rule(choices) => {
                let pos = state[idx];
                state[idx] += 1;
                state[idx] %= choices.len();

                for &u in &choices[pos] {
                    Self::step_inner(nodes, state, output, taunt_count, u, first_word);
                }
                if idx == 0 {
                    *first_word = true;
                }
            }
            Node::Terminal(s) if *first_word => {
                let mut cs: Vec<_> = s.chars().collect();
                // capitalize first non-whitespace character
                for c in &mut cs {
                    if c.is_alphabetic() {
                        *c = c.to_uppercase().next().unwrap();
                        *first_word = false;
                        break;
                    }
                }
                write!(output, "{}", cs.iter().collect::<String>()).unwrap();

                // let s_cap: String = c.next().unwrap().to_uppercase().chain(c).collect();
                // write!(output, "{}", s_cap).unwrap();
            }
            Node::Terminal(s) => write!(output, "{}", s).unwrap(),
        }
    }
}

struct KeywordDfa {
    keyword: Vec<char>,
    action: String,
    state: usize,
}

impl KeywordDfa {
    fn new(keyword: &str, action: &str) -> Self {
        Self {
            keyword: keyword.chars().collect(),
            action: action.to_string(),
            state: 0,
        }
    }

    fn step(&mut self, output: &mut impl std::io::Write, feed: &str, taunt_count: &mut usize) {
        for c in feed.chars() {
            if c == self.keyword[self.state] {
                self.state += 1;
                if self.state == self.keyword.len() {
                    if *taunt_count > 0 {
                        *taunt_count -= 1;
                        writeln!(output, "{}", self.action).unwrap();
                    }
                    self.state = 0;
                }
            }
        }
    }
}

fn main() {
    let bnf: &str =  "<taunt> ::= <sentence> | <taunt> <sentence> | <noun>! | <sentence>
<sentence> ::= <past-rel> <noun-phrase> | <present-rel> <noun-phrase> | <past-rel> <article> <noun> 
<noun-phrase> ::= <article> <modified-noun>
<modified-noun> ::= <noun> | <modifier> <noun>
<modifier> ::= <adjective> | <adverb> <adjective>
<present-rel> ::= your <present-person> <present-verb>
<past-rel> ::= your <past-person> <past-verb>
<present-person> ::= steed | king | first-born
<past-person> ::= mother | father | grandmother | grandfather | godfather
<noun> ::= hamster | coconut | duck | herring | newt | peril | chicken | vole | parrot | mouse | twit 
<present-verb> ::= is | masquerades as
<past-verb> ::= was | personified
<article> ::= a
<adjective> ::= silly | wicked | sordid | naughty | repulsive | malodorous | ill-tempered
<adverb> ::= conspicuously | categorically | positively | cruelly | incontrovertibly"
    ;

    let input = std::io::BufReader::new(std::io::stdin().lock());
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    // build dfa
    let mut taunt_dfa = TauntDfa::from_bnf(bnf);
    let mut holy_grail_dfa = KeywordDfa::new("theholygrail", "Taunter: (A childish hand gesture).");

    for line in input.lines() {
        let line = line.unwrap();
        let tokens = line.split_ascii_whitespace().collect::<Vec<_>>();
        let mut taunt_count = tokens
            .iter()
            .filter(|s| s.chars().any(|c| c.is_alphabetic()))
            .count()
            .div_ceil(3);

        write!(output, "Knight:").unwrap();
        for token in tokens {
            write!(output, " {}", token).unwrap();
        }
        writeln!(output).unwrap();

        holy_grail_dfa.step(&mut output, &line, &mut taunt_count);
        while taunt_count > 0 {
            taunt_dfa.step(&mut output, &mut taunt_count);
        }
        writeln!(output).unwrap();
    }
}
