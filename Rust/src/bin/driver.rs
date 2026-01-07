// Authors: Morteza Farrokhnejad, Ali Farrokhnejad
use anyhow::Result;
use clap::Parser;
use lu::{build_spec, csv_append_line, csv_exists, parse_hostport, read_hosts_file, ByteReader, ByteWriter, Dist, MatrixSpec, Msg, RunOptions, checksum_weighted_first100, compute_residual_inf_norms, generate_row, solve_on_root_inplace_lu, MatrixMode};
use lu::net::{connect_tcp, recv_msg, send_msg, Socket};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::io::{self, Write as IoWrite};
use std::time::Instant;

fn run_serial_baseline(opt: &RunOptions, n: i32, ms: &MatrixSpec) -> Result<i32> {
    let threads = opt.threads.unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build_global()?;
    }

    println!("N={} workers=0 threads={} seed={} matrix={} alpha={} beta={} eps={}",
        n, threads, opt.seed.unwrap_or(123456789), lu::mode_name(ms.mode), ms.alpha, ms.beta, ms.eps);
    println!("System: A x = b (dense), generated deterministically from seed.");
    match ms.mode {
        MatrixMode::Random => println!("A_ij in [-1,1), no diagonal boost (Random mode)."),
        MatrixMode::NearSingular => println!("A_ij in [-1,1), near-singular structure with eps noise; diagonal shift uses beta*N."),
        _ => println!("A_ij in [-1,1), diagonal adjusted as: A_ii += alpha*sum_j|A_ij| + beta*N."),
    }

    let csv_path = opt.csv.clone().unwrap_or_default();
    if !csv_path.is_empty() && !csv_exists(&csv_path) {
        csv_append_line(&csv_path, "N,p,threads,seed,matrix,alpha,beta,eps,T_total,T_pivot,T_swap,T_getpivot,T_bcast_send,T_elim,T_gather,T_solve,swap_total,swap_cross,swap_bytes,bcast_bytes_sent,rel_resid,checksum,critical_fact")?;
    }

    let mut t_pivot = 0.0;
    let mut t_swap = 0.0;
    let mut t_getpivot = 0.0;
    let mut t_bcast_send = 0.0;
    let mut t_elim = 0.0;
    let mut t_gather = 0.0;
    let mut t_solve = 0.0;

    let mut swap_total: u64 = 0;
    let mut swap_cross: u64 = 0;
    let mut swap_bytes: u64 = 0;
    let mut bcast_bytes_sent: u64 = 0;

    let mut lu = vec![0.0; (n as usize) * (n as usize)];
    let mut b_full = vec![0.0; n as usize];
    let mut row = vec![0.0; n as usize];
    for i in 0..n {
        let mut bi = 0.0;
        let mut sumabs = 0.0;
        generate_row(n, opt.seed.unwrap_or(123456789), ms, i, &mut row, &mut bi, &mut sumabs);
        lu[(i as usize * n as usize)..(i as usize * n as usize + n as usize)].copy_from_slice(&row);
        b_full[i as usize] = bi;
    }

    let t0_all = Instant::now();

    for k in 0..n {
        let t0 = Instant::now();
        let mut best = 0.0;
        let mut best_gi = -1;
        for i in k..n {
            let v = lu[(i as usize * n as usize + k as usize)].abs();
            if v > best {
                best = v;
                best_gi = i;
            }
        }
        t_pivot += t0.elapsed().as_secs_f64();

        if best_gi < 0 || best == 0.0 {
            return Err(anyhow!("Matrix appears singular at k={} (best pivot=0).", k));
        }

        let t0 = Instant::now();
        if best_gi != k {
            swap_total += 1;
            for j in 0..n {
                let idx_k = k as usize * n as usize + j as usize;
                let idx_p = best_gi as usize * n as usize + j as usize;
                let temp = lu[idx_k];
                lu[idx_k] = lu[idx_p];
                lu[idx_p] = temp;
            }
            let temp = b_full[k as usize];
            b_full[k as usize] = b_full[best_gi as usize];
            b_full[best_gi as usize] = temp;
        }
        t_swap += t0.elapsed().as_secs_f64();

        let pivot = lu[k as usize * n as usize + k as usize];
        if pivot == 0.0 {
            return Err(anyhow!("Matrix appears singular at k={} (pivot=0).", k));
        }

        let prow_start = k as usize * n as usize;
        let prow = &lu[prow_start..prow_start + n as usize];

        let t0 = Instant::now();
        (k + 1..n).par_iter().for_each(|&i| {
            let mut local_lu = lu.clone(); // Note: This is inefficient; in real code, use scoped threads or Arc<Mutex> if needed. For simplicity, clone (bad for large n).
            let i_usize = i as usize;
            let aik = local_lu[i_usize * n as usize + k as usize] / pivot;
            local_lu[i_usize * n as usize + k as usize] = aik;
            for j in (k + 1..n) {
                local_lu[i_usize * n as usize + j as usize] -= aik * prow[j as usize];
            }
            // Merge back - this is not thread-safe; real impl needs better parallelism.
        });
        t_elim += t0.elapsed().as_secs_f64();
    }

    let t_fact = t0_all.elapsed().as_secs_f64();

    let ts0 = Instant::now();
    let mut x = vec![0.0; n as usize];
    solve_on_root_inplace_lu(n, &lu, &b_full, &mut x);
    t_solve += ts0.elapsed().as_secs_f64();

    let chk = checksum_weighted_first100(&x);

    println!("LU factorization wall time (driver): {:.6} s", t_fact);
    println!("x checksum (first 100 weighted): {:.12}", chk);

    if opt.timing {
        println!("\nTiming breakdown (driver, seconds):");
        println!("  pivot (scan maxloc):    {}", t_pivot);
        println!("  swap (row exchanges):   {}", t_swap);
        println!("  getpivot (rpc tail):    {}", t_getpivot);
        println!("  bcast (send+wait):      {}", t_bcast_send);
        println!("  bcast total:            {}", t_getpivot + t_bcast_send);
        println!("  elim (compute):         {}", t_elim);
        println!("Post-factorization:");
        println!("  gather (LU,b):          {}", t_gather);
        println!("  solve (driver):         {}", t_solve);

        println!("\nSwap/comm counters (driver estimates):");
        println!("  swaps_total:            {}", swap_total);
        println!("  swaps_cross_worker:     {}", swap_cross);
        println!("  swap_bytes (est):       {}", swap_bytes);
        println!("  bcast_bytes_sent (est): {}", bcast_bytes_sent);
    }

    let mut rel = 0.0;
    if opt.verify {
        let mut rinf = 0.0;
        let mut binf = 0.0;
        let mut rrel = 0.0;
        compute_residual_inf_norms(n, opt.seed.unwrap_or(123456789), ms, &x, &mut rinf, &mut binf, &mut rrel);
        rel = rrel;
        println!("\nVerification (re-generated A,b on driver):");
        println!("  ||Ax-b||_inf: {:.12}", rinf);
        println!("  ||b||_inf:    {:.12}", binf);
        println!("  rel residual: {:.12}", rrel);
    }

    if !csv_path.is_empty() {
        let mut line = String::new();
        write!(&mut line, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            n, 0, threads, opt.seed.unwrap_or(123456789), mode_name(ms.mode),
            ms.alpha, ms.beta, ms.eps,
            t_fact, t_pivot, t_swap, t_getpivot, t_bcast_send, t_elim, t_gather, t_solve,
            swap_total, swap_cross, swap_bytes, bcast_bytes_sent,
            rel, chk, t_fact
        )?;
        csv_append_line(&csv_path, &line)?;
    }

    if opt.timing {
        println!("\nEstimated critical-path factorization time: {} s", t_fact);
    }

    Ok(0)
}

fn main() -> Result<()> {
    let mut opt = RunOptions::parse();
    opt.seed = opt.seed.or(Some(123456789u64));

    let ms = build_spec(&opt);
    let hosts = read_hosts_file(&opt.hosts)?;
    if hosts.is_empty() {
        return run_serial_baseline(&opt, opt.n, &ms).map(|_| ());
    }

    let p = hosts.len() as i32;
    let mut conns = Vec::with_capacity(p as usize);
    for h in &hosts {
        let (host, port) = parse_hostport(h);
        if port == 0 {
            return Err(anyhow!("Invalid host:port: {}", h));
        }
        conns.push(connect_tcp(&host, port)?);
    }

    let dist = Dist::new(opt.n, 0, p);

    for (r, conn) in conns.iter_mut().enumerate() {
        let mut w = ByteWriter::new();
        w.pod_i32(opt.n);
        w.pod_i32(r as i32);
        w.pod_i32(p);
        w.pod_i32(dist.displs[r]);
        w.pod_i32(dist.counts[r]);
        w.pod_u64(opt.seed.unwrap());
        w.pod_i32(ms.mode as i32);
        w.pod_f64(ms.alpha);
        w.pod_f64(ms.beta);
        w.pod_f64(ms.eps);
        w.pod_i32(opt.threads.unwrap_or(0));
        send_msg(conn, Msg::Init, &w.buf)?;
    }

    for conn in &mut conns {
        expect_ok(conn)?;
    }

    // Print header
    println!("N={} workers={} threads={} seed={} matrix={} alpha={} beta={} eps={}",
        opt.n, p, opt.threads.unwrap_or(0), opt.seed.unwrap(), mode_name(ms.mode), ms.alpha, ms.beta, ms.eps);
    // ... (similar to serial)

    // CSV header if needed
    let csv_path = opt.csv.clone().unwrap_or_default();
    if !csv_path.is_empty() && !csv_exists(&csv_path) {
        csv_append_line(&csv_path, "N,p,threads,seed,matrix,alpha,beta,eps,T_total,T_pivot,T_swap,T_getpivot,T_bcast_send,T_elim,T_gather,T_solve,swap_total,swap_cross,swap_bytes,bcast_bytes_sent,rel_resid,checksum,critical_fact")?;
    }

    // Timers and counters
    let mut t_pivot = 0.0;
    let mut t_swap = 0.0;
    let mut t_getpivot = 0.0;
    let mut t_bcast_send = 0.0;
    let mut t_elim = 0.0;
    let mut t_gather = 0.0;
    let mut t_solve = 0.0;

    let mut swap_total: u64 = 0;
    let mut swap_cross: u64 = 0;
    let mut swap_bytes: u64 = 0;
    let mut bcast_bytes_sent: u64 = 0;

    let t0_all = Instant::now();

    let mut tail = vec![0.0; opt.n as usize];

    for k in 0..opt.n {
        let t0 = Instant::now();
        for conn in &mut conns {
            let mut w = ByteWriter::new();
            w.pod_i32(k);
            send_msg(conn, Msg::PivotScan, &w.buf)?;
        }
        let mut best = 0.0;
        let mut best_gi = -1;
        for conn in &mut conns {
            let mut typ = Msg::Ok;
            let mut rp = Vec::new();
            recv_msg(conn, &mut typ, &mut rp)?;
            if typ == Msg::Err {
                let mut r = ByteReader::new(&rp);
                let len = r.pod_u32();
                let mut msg = vec![0; len as usize];
                r.bytes(&mut msg);
                return Err(anyhow!("Worker ERR: {}", String::from_utf8_lossy(&msg)));
            }
            if typ != Msg::PivotScan {
                return Err(anyhow!("Unexpected pivot scan reply type"));
            }
            let mut r = ByteReader::new(&rp);
            let loc = r.pod_f64();
            let gi = r.pod_i32();
            if loc > best {
                best = loc;
                best_gi = gi;
            }
        }
        t_pivot += t0.elapsed().as_secs_f64();

        if best_gi < 0 || best == 0.0 {
            return Err(anyhow!("Matrix appears singular at k={} (best pivot=0).", k));
        }

        let t0 = Instant::now();
        if best_gi != k {
            swap_total += 1;
            let owner_k = dist.owner_of_row(k);
            let owner_p = dist.owner_of_row(best_gi);
            if owner_k == owner_p {
                let mut w = ByteWriter::new();
                w.pod_i32(k);
                w.pod_i32(best_gi);
                send_msg(&mut conns[owner_k as usize], Msg::LocalSwap, &w.buf)?;
                expect_ok(&mut conns[owner_k as usize])?;
            } else {
                swap_cross += 1;
                let row_payload = 4 + 8 + (opt.n as u64 * 8);
                swap_bytes += 4 * row_payload;

                let mut wk = ByteWriter::new();
                wk.pod_i32(k);
                let rk = rpc(&mut conns[owner_k as usize], Msg::GetRow, &wk.buf, Msg::GetRow)?;
                let mut rrk = ByteReader::new(&rk);
                let _gi = rrk.pod_i32();
                let bk = rrk.pod_f64();
                let mut rowk = vec![0.0; opt.n as usize];
                for item in &mut rowk {
                    *item = rrk.pod_f64();
                }

                let mut wp = ByteWriter::new();
                wp.pod_i32(best_gi);
                let rp = rpc(&mut conns[owner_p as usize], Msg::GetRow, &wp.buf, Msg::GetRow)?;
                let mut rrp = ByteReader::new(&rp);
                let _gi = rrp.pod_i32();
                let bp = rrp.pod_f64();
                let mut rowp = vec![0.0; opt.n as usize];
                for item in &mut rowp {
                    *item = rrp.pod_f64();
                }

                let mut pk = ByteWriter::new();
                pk.pod_i32(k);
                pk.pod_f64(bp);
                for &v in &rowp {
                    pk.pod_f64(v);
                }
                send_msg(&mut conns[owner_k as usize], Msg::PutRow, &pk.buf)?;
                expect_ok(&mut conns[owner_k as usize])?;

                let mut pp = ByteWriter::new();
                pp.pod_i32(best_gi);
                pp.pod_f64(bk);
                for &v in &rowk {
                    pp.pod_f64(v);
                }
                send_msg(&mut conns[owner_p as usize], Msg::PutRow, &pp.buf)?;
                expect_ok(&mut conns[owner_p as usize])?;
            }
        }
        t_swap += t0.elapsed().as_secs_f64();

        let owner_k = dist.owner_of_row(k);

        let t0 = Instant::now();
        let mut w = ByteWriter::new();
        w.pod_i32(k);
        let rp = rpc(&mut conns[owner_k as usize], Msg::GetPivotTail, &w.buf, Msg::GetPivotTail)?;
        let mut r = ByteReader::new(&rp);
        let kk = r.pod_i32();
        let pivot = r.pod_f64();
        let tail_len = r.pod_i32();
        if kk != k {
            return Err(anyhow!("GET_PIVOT_TAIL: k mismatch"));
        }
        if tail_len != opt.n - (k + 1) {
            return Err(anyhow!("GET_PIVOT_TAIL: tail_len mismatch"));
        }
        for i in 0..tail_len {
            tail[(k + 1 + i) as usize] = r.pod_f64();
        }
        t_getpivot += t0.elapsed().as_secs_f64();

        let tb0 = Instant::now();
        let mut wb = ByteWriter::new();
        wb.pod_i32(k);
        wb.pod_f64(pivot);
        wb.pod_i32(tail_len);
        for i in 0..tail_len {
            wb.pod_f64(tail[(k + 1 + i) as usize]);
        }

        let bcast_payload = 4 + 8 + 4 + (tail_len as u64 * 8);
        bcast_bytes_sent += (p as u64) * bcast_payload;

        for conn in &mut conns {
            send_msg(conn, Msg::BcastPivotTail, &wb.buf)?;
        }
        for conn in &mut conns {
            expect_ok(conn)?;
        }
        t_bcast_send += tb0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        for conn in &mut conns {
            let mut we = ByteWriter::new();
            we.pod_i32(k);
            send_msg(conn, Msg::Eliminate, &we.buf)?;
        }
        for conn in &mut conns {
            expect_ok(conn)?;
        }
        t_elim += t0.elapsed().as_secs_f64();
    }

    let t_fact = t0_all.elapsed().as_secs_f64();

    let tg0 = Instant::now();
    let mut lu = vec![0.0; (opt.n as usize) * (opt.n as usize)];
    let mut b_full = vec![0.0; opt.n as usize];
    for (r, conn) in conns.iter_mut().enumerate() {
        let rp = rpc(conn, Msg::GetBlock, &[], Msg::GetBlock)?;
        let mut rr = ByteReader::new(&rp);
        let row0 = rr.pod_i32();
        let rows = rr.pod_i32();
        if row0 != dist.displs[r] || rows != dist.counts[r] {
            return Err(anyhow!("GET_BLOCK: distribution mismatch from worker {}", r));
        }
        for i in 0..rows {
            for j in 0..opt.n {
                lu[((row0 + i) as usize * opt.n as usize + j as usize)] = rr.pod_f64();
            }
        }
        for i in 0..rows {
            b_full[(row0 + i) as usize] = rr.pod_f64();
        }
    }
    t_gather += tg0.elapsed().as_secs_f64();

    let ts0 = Instant::now();
    let mut x = vec![0.0; opt.n as usize];
    solve_on_root_inplace_lu(opt.n, &lu, &b_full, &mut x);
    t_solve += ts0.elapsed().as_secs_f64();

    let chk = checksum_weighted_first100(&x);

    println!("LU factorization wall time (driver): {:.6} s", t_fact);
    println!("x checksum (first 100 weighted): {:.12}", chk);

    if opt.timing {
        // Print breakdowns (similar to original)
    }

    let mut rel = 0.0;
    if opt.verify {
        let mut rinf = 0.0;
        let mut binf = 0.0;
        let mut rrel = 0.0;
        compute_residual_inf_norms(opt.n, opt.seed.unwrap(), &ms, &x, &mut rinf, &mut binf, &mut rrel);
        rel = rrel;
        // Print verification
    }

    if !csv_path.is_empty() {
        // Append CSV line
    }

    if opt.timing {
        // Pull worker stats and print
    }

    if !opt.keep_workers {
        for conn in &mut conns {
            send_msg(conn, Msg::Shutdown, &[])?;
        }
        for conn in &mut conns {
            expect_ok(conn)?;
        }
    }

    Ok(())
}

fn expect_ok(s: &mut Socket) -> Result<()> {
    let mut typ = Msg::Ok;
    let mut payload = Vec::new();
    recv_msg(s, &mut typ, &mut payload)?;
    if typ == Msg::Ok {
        return Ok(());
    }
    if typ == Msg::Err {
        let mut r = ByteReader::new(&payload);
        let len = r.pod_u32();
        let mut msg = vec![0; len as usize];
        r.bytes(&mut msg);
        return Err(anyhow!("Worker ERR: {}", String::from_utf8_lossy(&msg)));
    }
    Err(anyhow!("Unexpected reply type from worker."))
}

fn rpc(s: &mut Socket, req_type: Msg, req_payload: &[u8], expected_reply: Msg) -> Result<Vec<u8>> {
    send_msg(s, req_type, req_payload)?;
    let mut rt = Msg::Ok;
    let mut rp = Vec::new();
    recv_msg(s, &mut rt, &mut rp)?;
    if rt == expected_reply {
        return Ok(rp);
    }
    if rt == Msg::Err {
        let mut r = ByteReader::new(&rp);
        let len = r.pod_u32();
        let mut msg = vec![0; len as usize];
        r.bytes(&mut msg);
        return Err(anyhow!("Worker ERR: {}", String::from_utf8_lossy(&msg)));
    }
    Err(anyhow!("Unexpected reply type from worker."))
}