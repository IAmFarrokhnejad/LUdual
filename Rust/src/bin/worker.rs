use anyhow::Result;
use clap::Parser;
use lu::{ByteReader, ByteWriter, Dist, MatrixSpec, Msg, WorkerOptions, generate_row, val_ij};
use lu::net::{listen_and_accept, recv_msg, send_msg, Socket};
use rayon::prelude::*;
use std::time::Instant;

// Authors: Morteza Farrokhnejad, Ali Farrokhnejad
struct WorkerState {
    n: i32,
    rank: i32,
    p: i32,
    row0: i32,
    rows: i32,
    seed: u64,
    ms: MatrixSpec,
    threads: i32,
    a: Vec<f64>,
    b: Vec<f64>,
    cur_k: i32,
    cur_pivot: f64,
    pivot_tail: Vec<f64>,
    t_scan: f64,
    t_elim: f64,
    t_recv_pivot: f64,
    bytes_in: u64,
    bytes_out: u64,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            n: 0,
            rank: 0,
            p: 1,
            row0: 0,
            rows: 0,
            seed: 0,
            ms: MatrixSpec::default(),
            threads: 0,
            a: Vec::new(),
            b: Vec::new(),
            cur_k: -1,
            cur_pivot: 0.0,
            pivot_tail: Vec::new(),
            t_scan: 0.0,
            t_elim: 0.0,
            t_recv_pivot: 0.0,
            bytes_in: 0,
            bytes_out: 0,
        }
    }

    fn init_storage(&mut self) {
        self.a.resize((self.rows as usize) * (self.n as usize), 0.0);
        self.b.resize(self.rows as usize, 0.0);
        self.pivot_tail.resize(self.n as usize, 0.0);

        let mut row = vec![0.0; self.n as usize];
        for li in 0..self.rows {
            let gi = self.row0 + li;
            let mut bi = 0.0;
            let mut sumabs = 0.0;
            generate_row(self.n, self.seed, &self.ms, gi, &mut row, &mut bi, &mut sumabs);
            self.a[(li as usize * self.n as usize)..(li as usize * self.n as usize + self.n as usize)].copy_from_slice(&row);
            self.b[li as usize] = bi;
        }
    }
}

fn send_ok(s: &mut Socket) -> Result<()> {
    send_msg(s, Msg::Ok, &[])
}

fn send_err(s: &mut Socket, msg: &str) -> Result<()> {
    let mut w = ByteWriter::new();
    w.pod_u32(msg.len() as u32);
    w.bytes(msg.as_bytes());
    send_msg(s, Msg::Err, &w.buf)
}

fn handle_message(s: &mut Socket, st: &mut WorkerState, typ: Msg, payload: &[u8]) -> Result<bool> {
    match typ {
        Msg::Init => {
            let mut r = ByteReader::new(payload);
            st.n = r.pod_i32();
            st.rank = r.pod_i32();
            st.p = r.pod_i32();
            st.row0 = r.pod_i32();
            st.rows = r.pod_i32();
            st.seed = r.pod_u64();
            st.ms.mode = unsafe { std::mem::transmute(r.pod_i32()) };
            st.ms.alpha = r.pod_f64();
            st.ms.beta = r.pod_f64();
            st.ms.eps = r.pod_f64();
            st.threads = r.pod_i32();
            if st.threads > 0 {
                rayon::ThreadPoolBuilder::new().num_threads(st.threads as usize).build_global()?;
            }
            st.init_storage();
            send_ok(s)?;
            Ok(true)
        }
        _ if st.n <= 0 => {
            send_err(s, "Worker not initialized (missing INIT).")?;
            Ok(true)
        }
        Msg::PivotScan => {
            let mut r = ByteReader::new(payload);
            let k = r.pod_i32();
            let t0 = Instant::now();
            let mut best = 0.0;
            let mut best_gi = -1;
            let start_g = cmp::max(k, st.row0);
            let end_g = st.row0 + st.rows;
            if start_g < end_g {
                let li0 = start_g - st.row0;
                for li in li0..st.rows {
                    let gi = st.row0 + li;
                    let v = st.a[(li as usize * st.n as usize + k as usize)].abs();
                    if v > best {
                        best = v;
                        best_gi = gi;
                    }
                }
            }
            st.t_scan += t0.elapsed().as_secs_f64();
            let mut w = ByteWriter::new();
            w.pod_f64(best);
            w.pod_i32(best_gi);
            st.bytes_out += (4 + 4 + w.buf.len()) as u64;
            send_msg(s, Msg::PivotScan, &w.buf)?;
            Ok(true)
        }
        Msg::LocalSwap => {
            let mut r = ByteReader::new(payload);
            let gi1 = r.pod_i32();
            let gi2 = r.pod_i32();
            let li1 = gi1 - st.row0;
            let li2 = gi2 - st.row0;
            if li1 < 0 || li1 >= st.rows || li2 < 0 || li2 >= st.rows {
                send_err(s, "LOCAL_SWAP rows not owned by this worker.")?;
                return Ok(true);
            }
            for j in 0..st.n {
                let idx1 = li1 as usize * st.n as usize + j as usize;
                let idx2 = li2 as usize * st.n as usize + j as usize;
                let temp = st.a[idx1];
                st.a[idx1] = st.a[idx2];
                st.a[idx2] = temp;
            }
            let temp = st.b[li1 as usize];
            st.b[li1 as usize] = st.b[li2 as usize];
            st.b[li2 as usize] = temp;
            send_ok(s)?;
            Ok(true)
        }
        Msg::GetRow => {
            let mut r = ByteReader::new(payload);
            let gi = r.pod_i32();
            let li = gi - st.row0;
            if li < 0 || li >= st.rows {
                send_err(s, "GET_ROW row not owned by this worker.")?;
                return Ok(true);
            }
            let mut w = ByteWriter::new();
            w.pod_i32(gi);
            w.pod_f64(st.b[li as usize]);
            for j in 0..st.n {
                w.pod_f64(st.a[li as usize * st.n as usize + j as usize]);
            }
            st.bytes_out += (4 + 4 + w.buf.len()) as u64;
            send_msg(s, Msg::GetRow, &w.buf)?;
            Ok(true)
        }
        Msg::PutRow => {
            let mut r = ByteReader::new(payload);
            let gi = r.pod_i32();
            let bi = r.pod_f64();
            let li = gi - st.row0;
            if li < 0 || li >= st.rows {
                send_err(s, "PUT_ROW row not owned by this worker.")?;
                return Ok(true);
            }
            st.b[li as usize] = bi;
            for j in 0..st.n {
                st.a[li as usize * st.n as usize + j as usize] = r.pod_f64();
            }
            send_ok(s)?;
            Ok(true)
        }
        Msg::GetPivotTail => {
            let mut r = ByteReader::new(payload);
            let k = r.pod_i32();
            let li = k - st.row0;
            if li < 0 || li >= st.rows {
                send_err(s, "GET_PIVOT_TAIL: pivot row not owned by this worker.")?;
                return Ok(true);
            }
            let pivot = st.a[li as usize * st.n as usize + k as usize];
            let tail_len = st.n - (k + 1);
            let mut w = ByteWriter::new();
            w.pod_i32(k);
            w.pod_f64(pivot);
            w.pod_i32(tail_len);
            for j in (k + 1)..st.n {
                w.pod_f64(st.a[li as usize * st.n as usize + j as usize]);
            }
            st.bytes_out += (4 + 4 + w.buf.len()) as u64;
            send_msg(s, Msg::GetPivotTail, &w.buf)?;
            Ok(true)
        }
        Msg::BcastPivotTail => {
            let t0 = Instant::now();
            let mut r = ByteReader::new(payload);
            st.cur_k = r.pod_i32();
            st.cur_pivot = r.pod_f64();
            let tail_len = r.pod_i32();
            if st.cur_k < 0 || st.cur_k >= st.n {
                send_err(s, "BCAST_PIVOT_TAIL: bad k")?;
                return Ok(true);
            }
            if tail_len != st.n - (st.cur_k + 1) {
                send_err(s, "BCAST_PIVOT_TAIL: tail_len mismatch")?;
                return Ok(true);
            }
            for i in 0..tail_len {
                st.pivot_tail[(st.cur_k + 1 + i) as usize] = r.pod_f64();
            }
            st.t_recv_pivot += t0.elapsed().as_secs_f64();
            send_ok(s)?;
            Ok(true)
        }
        Msg::Eliminate => {
            let mut r = ByteReader::new(payload);
            let k = r.pod_i32();
            if k != st.cur_k {
                send_err(s, "ELIMINATE: k does not match last BCAST_PIVOT_TAIL")?;
                return Ok(true);
            }
            let pivot = st.cur_pivot;
            if pivot == 0.0 {
                send_err(s, "ELIMINATE: pivot is zero")?;
                return Ok(true);
            }
            let t0 = Instant::now();
            let start_g = cmp::max(k + 1, st.row0);
            let end_g = st.row0 + st.rows;
            if start_g < end_g {
                let li0 = start_g - st.row0;
                (li0..st.rows).par_iter().for_each(|&li| {
                    let li_usize = li as usize;
                    let mut aik = st.a[li_usize * st.n as usize + k as usize] / pivot;
                    st.a[li_usize * st.n as usize + k as usize] = aik;
                    for j in (k + 1..st.n) {
                        st.a[li_usize * st.n as usize + j as usize] -= aik * st.pivot_tail[j as usize];
                    }
                });
            }
            st.t_elim += t0.elapsed().as_secs_f64();
            send_ok(s)?;
            Ok(true)
        }
        Msg::GetBlock => {
            let mut w = ByteWriter::new();
            w.pod_i32(st.row0);
            w.pod_i32(st.rows);
            for v in &st.a {
                w.pod_f64(*v);
            }
            for v in &st.b {
                w.pod_f64(*v);
            }
            st.bytes_out += (4 + 4 + w.buf.len()) as u64;
            send_msg(s, Msg::GetBlock, &w.buf)?;
            Ok(true)
        }
        Msg::GetStats => {
            let mut w = ByteWriter::new();
            w.pod_f64(st.t_scan);
            w.pod_f64(st.t_recv_pivot);
            w.pod_f64(st.t_elim);
            w.pod_u64(st.bytes_in);
            w.pod_u64(st.bytes_out);
            send_msg(s, Msg::GetStats, &w.buf)?;
            Ok(true)
        }
        Msg::Shutdown => {
            send_ok(s)?;
            Ok(false)
        }
        _ => {
            send_err(s, "Unknown message type.")?;
            Ok(true)
        }
    }
}

fn main() -> Result<()> {
    let opt = WorkerOptions::parse();
    let (bind_ip, port) = parse_hostport(&opt.listen);
    if port == 0 {
        return Err(anyhow!("Invalid listen: {}", opt.listen));
    }
    println!("lu_worker: listening on {}:{}", bind_ip, port);
    println!("lu_worker: ready (will accept multiple driver sessions).");

    loop {
        let mut s = listen_and_accept(&bind_ip, port)?;
        println!("lu_worker: driver connected.");

        let mut st = WorkerState::new();
        if let Some(t) = opt.threads {
            st.threads = t;
        }

        let mut shutdown_requested = false;

        loop {
            let mut typ = Msg::Ok;
            let mut payload = Vec::new();
            if let Err(e) = recv_msg(&mut s, &mut typ, &mut payload) {
                let msg = e.to_string();
                if msg.contains("peer closed connection") {
                    println!("lu_worker: driver disconnected (session ended).");
                    break;
                }
                return Err(e);
            }

            st.bytes_in += (4 + 4 + payload.len()) as u64;

            let cont = handle_message(&mut s, &mut st, typ, &payload)?;
            if !cont {
                shutdown_requested = true;
                break;
            }
        }

        if shutdown_requested {
            println!("lu_worker: shutdown.");
            return Ok(());
        }
    }
}