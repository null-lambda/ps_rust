use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
    io::Write,
};

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> &'static str {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buffer = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buffer, stat[6])) }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::new(stdout)
    }
}

mod datetime {
    use std::{
        fmt::{Display, Formatter},
        ops::Sub,
    };

    #[repr(C)]
    #[derive(Debug, Default)]
    struct Tm {
        sec: i32,
        min: i32,
        hour: i32,
        mday: i32,
        mon: i32,
        year: i32,
        wday: i32,
        yday: i32,
        isdst: i32,
    }

    mod ffi {
        use super::Tm;
        extern "C" {
            pub fn strptime(s: *const u8, format: *const u8, tm: *mut Tm) -> *mut u8;
            pub fn mktime(tm: *const Tm) -> i64;
        }
    }

    #[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Instant(i64);

    impl Sub for Instant {
        type Output = Duration;
        fn sub(self, other: Instant) -> Duration {
            Duration(self.0 - other.0)
        }
    }

    pub fn parse(s_date: &[u8], s_time: &[u8]) -> Instant {
        unsafe {
            let s = format!(
                "{} {}",
                std::str::from_utf8_unchecked(s_date),
                std::str::from_utf8_unchecked(s_time)
            );
            let format = b"%Y-%m-%d %H:%M:%S";
            let mut tm = Tm::default();
            ffi::strptime(s.as_ptr(), format.as_ptr(), &mut tm);
            let epoch = ffi::mktime(&tm);
            Instant(epoch)
        }
    }

    #[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
    pub struct Duration(i64);

    impl Duration {
        pub fn from_mins(mins: i64) -> Self {
            Duration(mins * 60)
        }

        pub fn as_mins(&self) -> i64 {
            self.0 / 60
        }
    }

    impl Display for Duration {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            let mut mins = self.0 / 60;
            let hours = mins / 60;
            mins %= 60;
            write!(f, "{}:{:02}", hours, mins)
        }
    }
}

#[derive(Debug)]
struct Problem {
    order: usize,
    pscore: u64,
}

#[derive(Debug, Default)]
struct Submission {
    score: u64,
    penalty1: u32,
    penalty2: u32,
    penalty: u32,
    n_tries: u32,
    best_tries: u32,
    best_sid: u32,
    recent_sid: u32,
}

#[derive(Debug, Default)]
struct User {
    rank: u32,
    submissions: BTreeMap<u16, Submission>,
    last_ac_sid: u32,
    last_sid: u32,
    total_penalty: u64,
    total_score: u64,
}

fn main() {
    let buf = fast_io::stdin();
    let mut lines = buf.lines();
    let mut output = fast_io::stdout();

    let mut tokens = lines.next().unwrap().split_whitespace();

    let penalty: u32 = tokens.next().unwrap().parse().unwrap();
    let start_time = datetime::parse(
        tokens.next().unwrap().as_bytes(),
        tokens.next().unwrap().as_bytes(),
    );
    let last: bool = tokens.next().unwrap() == "1";
    let ce: bool = tokens.next().unwrap() == "1";
    let cscore: bool = tokens.next().unwrap() == "1";
    let format: bool = tokens.next().unwrap() == "1";

    let n_problems: usize = lines.next().unwrap().parse().unwrap();
    let problems: HashMap<u16, Problem> = (0..n_problems)
        .map(|_| {
            let mut tokens = lines.next().unwrap().split_whitespace();
            let id = tokens.next().unwrap().parse().unwrap();
            let order = tokens.next().unwrap().parse().unwrap();
            let pscore = tokens.next().unwrap().parse().unwrap();
            (id, Problem { order, pscore })
        })
        .collect();

    let n_users: usize = lines.next().unwrap().parse().unwrap();
    let mut users: HashMap<&str, User> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .take(n_users)
        .map(|id| (id, User::default()))
        .collect();
    for (_, user) in &mut users {
        for pid in problems.keys() {
            user.submissions.insert(*pid, Submission::default());
        }
    }

    let n_submissions: usize = lines.next().unwrap().parse().unwrap();
    for _ in 0..n_submissions {
        let mut tokens = lines.next().unwrap().split_whitespace();
        let sid: u32 = tokens.next().unwrap().parse().unwrap();
        let pid: u16 = tokens.next().unwrap().parse().unwrap();
        let uid = tokens.next().unwrap();
        let result: u8 = tokens.next().unwrap().parse().unwrap();

        let presult: u8 = tokens.next().unwrap().parse().unwrap();
        let score: u64 = tokens.next().unwrap().parse().unwrap();
        let date = datetime::parse(
            tokens.next().unwrap().as_bytes(),
            tokens.next().unwrap().as_bytes(),
        );

        let user = users.get_mut(uid);
        if user.is_none() {
            continue;
        }
        let user = user.unwrap();

        if result == 13 || ce && result == 11 {
            continue;
        }

        user.last_sid = sid;
        let submission = user.submissions.get_mut(&pid).unwrap();
        let problem = problems.get(&pid).unwrap();

        submission.n_tries += 1;
        let ac = result == 4;
        if ac {
            user.last_ac_sid = sid;

            let score = if cscore {
                if presult == 0 && score == 0 {
                    problem.pscore
                } else {
                    score
                }
            } else {
                1
            };
            let better_submission = score > submission.score;
            if better_submission {
                // println!("{} {} {} {} {}", sid, pid, uid, score, submission.score);
                submission.best_sid = sid;
                submission.score = if cscore { score } else { 1 };
                submission.penalty1 = (submission.n_tries - 1) * penalty;
                submission.penalty2 = (date - start_time).as_mins() as u32;
                submission.penalty = if last {
                    submission.penalty2
                } else {
                    submission.penalty1 + submission.penalty2
                };
                submission.best_tries = submission.n_tries;
            }
        }
    }

    let mut users: Vec<Box<(&str, User)>> = users.into_iter().map(Box::new).collect();
    for u in &mut users {
        let (_, user) = u.as_mut();
        let mut total_penalty1 = 0;
        let mut total_penalty2 = 0;
        for (_, submission) in &user.submissions {
            user.total_score += submission.score;
            if last {
                total_penalty1 += submission.penalty1 as u64;
                total_penalty2 = total_penalty2.max(submission.penalty2 as u64);
            } else {
                total_penalty1 += submission.penalty1 as u64;
                total_penalty2 += submission.penalty2 as u64;
            }
        }
        user.total_penalty = total_penalty1 + total_penalty2;
    }

    let key_rank = |info: &User| (Reverse(info.total_score), info.total_penalty);
    let key_sub = |info: &User| (info.last_ac_sid, info.last_sid);
    users.sort_by_key(|u| {
        let (id, info) = (*u).as_ref();
        (key_rank(info), key_sub(info), *id)
    });

    for i in 0..users.len() {
        if i > 0 && key_rank(&users[i - 1].1) == key_rank(&users[i].1) {
            users[i].1.rank = users[i - 1].1.rank;
        } else {
            users[i].1.rank = i as u32 + 1;
        }
    }

    let format_penalty = |penalty: u64| {
        if format {
            format!("{}", datetime::Duration::from_mins(penalty as i64))
        } else {
            penalty.to_string()
        }
    };

    let mut problems: Vec<(u16, Problem)> = problems.into_iter().collect();
    problems.sort_by_key(|p| p.1.order);
    for u in &users {
        let (uid, user) = u.as_ref();
        write!(output, "{},{},", user.rank, uid).unwrap();

        if cscore {
            for (pid, problem) in &problems {
                let submission = user.submissions.get(pid).unwrap();
                if submission.n_tries == 0 {
                    write!(output, "0/--,").unwrap();
                } else if submission.best_sid == 0 {
                    write!(output, "w/{}/--,", submission.n_tries).unwrap();
                } else {
                    let success = if submission.score == problem.pscore {
                        'a'
                    } else {
                        'p'
                    };
                    write!(
                        output,
                        "{}/{}/{}/{},",
                        success,
                        submission.score,
                        submission.best_tries,
                        format_penalty(submission.penalty as u64)
                    )
                    .unwrap();
                }
            }
        } else {
            for (pid, _) in &problems {
                let submission = user.submissions.get(pid).unwrap();
                if submission.n_tries == 0 {
                    write!(output, "0/--,").unwrap();
                } else if submission.best_sid == 0 {
                    write!(output, "w/{}/--,", submission.n_tries).unwrap();
                } else {
                    write!(
                        output,
                        "a/{}/{},",
                        submission.best_tries,
                        format_penalty(submission.penalty as u64)
                    )
                    .unwrap();
                }
            }
        }
        write!(
            output,
            "{}/{}",
            user.total_score,
            format_penalty(user.total_penalty)
        )
        .unwrap();
        writeln!(output).unwrap();
    }
}
