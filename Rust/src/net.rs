use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};


// Authors: Morteza Farrokhnejad, Ali Farrokhnejad


pub type Socket = TcpStream;

pub fn closesocket_safe(_s: Socket) {
    // Drop handles closure in Rust
}

pub fn send_all(s: &mut Socket, data: &[u8]) -> Result<()> {
    s.write_all(data).map_err(Into::into)
}

pub fn recv_all(s: &mut Socket, data: &mut [u8]) -> Result<()> {
    let mut off = 0;
    while off < data.len() {
        let n = s.read(&mut data[off..])?;
        if n == 0 {
            return Err(anyhow!("peer closed connection"));
        }
        off += n;
    }
    Ok(())
}

pub fn connect_tcp(host: &str, port: u16) -> Result<Socket> {
    let addr = format!("{}:{}", host, port);
    TcpStream::connect(addr).map_err(Into::into)
}

pub fn listen_and_accept(bind_ip: &str, port: u16) -> Result<Socket> {
    let addr = format!("{}:{}", bind_ip, port);
    let listener = TcpListener::bind(addr)?;
    let (stream, _) = listener.accept()?;
    Ok(stream)
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Msg {
    Init = 1,
    PivotScan = 2,
    GetRow = 3,
    PutRow = 4,
    LocalSwap = 5,
    GetPivotTail = 6,
    BcastPivotTail = 7,
    Eliminate = 8,
    GetBlock = 9,
    GetStats = 10,
    Shutdown = 11,
    Ok = 200,
    Err = 500,
}

pub struct ByteWriter {
    pub buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn pod_i32(&mut self, v: i32) {
        self.buf.write_i32::<LittleEndian>(v).unwrap();
    }

    pub fn pod_u64(&mut self, v: u64) {
        self.buf.write_u64::<LittleEndian>(v).unwrap();
    }

    pub fn pod_f64(&mut self, v: f64) {
        self.buf.write_f64::<LittleEndian>(v).unwrap();
    }

    pub fn pod_u16(&mut self, v: u16) {
        self.buf.write_u16::<LittleEndian>(v).unwrap();
    }

    pub fn pod_u32(&mut self, v: u32) {
        self.buf.write_u32::<LittleEndian>(v).unwrap();
    }

    pub fn bytes(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }
}

pub struct ByteReader<'a> {
    p: &'a [u8],
}

impl<'a> ByteReader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { p: b }
    }

    pub fn pod_i32(&mut self) -> i32 {
        let v = LittleEndian::read_i32(self.p);
        self.p = &self.p[4..];
        v
    }

    pub fn pod_u64(&mut self) -> u64 {
        let v = LittleEndian::read_u64(self.p);
        self.p = &self.p[8..];
        v
    }

    pub fn pod_f64(&mut self) -> f64 {
        let v = LittleEndian::read_f64(self.p);
        self.p = &self.p[8..];
        v
    }

    pub fn pod_u32(&mut self) -> u32 {
        let v = LittleEndian::read_u32(self.p);
        self.p = &self.p[4..];
        v
    }

    pub fn bytes(&mut self, out: &mut [u8]) {
        out.copy_from_slice(self.p);
        self.p = &[];
    }
}

pub fn send_msg(s: &mut Socket, typ: Msg, payload: &[u8]) -> Result<()> {
    let mut buf = Vec::new();
    let payload_bytes = (payload.len() + 4) as u32;
    buf.write_u32::<LittleEndian>(payload_bytes).unwrap();
    buf.write_u16::<LittleEndian>(typ as u16).unwrap();
    buf.write_u16::<LittleEndian>(0).unwrap();
    buf.extend_from_slice(payload);
    send_all(s, &buf)
}

pub fn recv_msg(s: &mut Socket, type_out: &mut Msg, payload_out: &mut Vec<u8>) -> Result<()> {
    let mut len_buf = [0u8; 4];
    recv_all(s, &mut len_buf)?;
    let payload_bytes = LittleEndian::read_u32(&len_buf);
    if payload_bytes < 4 {
        return Err(anyhow!("Bad frame: payload_bytes < 4"));
    }
    let mut frame = vec![0u8; (payload_bytes - 4) as usize + 4];
    recv_all(s, &mut frame)?;
    let t = LittleEndian::read_u16(&frame[0..2]);
    let _rsv = LittleEndian::read_u16(&frame[2..4]);
    *type_out = unsafe { std::mem::transmute(t) }; // Assume enum values match
    payload_out.clear();
    payload_out.extend_from_slice(&frame[4..]);
    Ok(())
}