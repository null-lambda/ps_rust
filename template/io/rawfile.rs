use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    os::unix::io::{FromRawFd, IntoRawFd},
};

fn main() {
    let stdin = unsafe { File::from_raw_fd(0) };
    let stdout = unsafe { File::from_raw_fd(1) };
    let (mut reader, mut writer) = (BufReader::new(stdin), BufWriter::new(stdout));

    // do reading and writing
    writer.write_all(b"hello world\n").unwrap();
    writer.write_fmt(format_args!("hello {}\n", "world")).unwrap();
}