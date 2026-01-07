use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use rayon::prelude::*;
use std::cmp;
use std::f64;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixMode {
    DiagDominant,
    WeakDiagDominant,
    Random,
    NearSingular,
}

#[derive(Copy, Clone, Debug)]
pub struct MatrixSpec {
    pub mode: MatrixMode,
    pub alpha: f64,
    pub beta: f64,
    pub eps: f64,
}

impl Default for MatrixSpec {
    fn default() -> Self {
        Self {
            mode: MatrixMode::DiagDominant,
            alpha: 0.0,
            beta: 0.0,
            eps: 1e-3,
        }
    }
}

pub fn mode_name(m: MatrixMode) -> &'static str {
    match m {
        MatrixMode::DiagDominant => "dd",
        MatrixMode::WeakDiagDominant => "weakdd",
        MatrixMode::Random => "rand",
        MatrixMode::NearSingular => "near_singular",
    }
}

pub fn parse_mode(s: &str) -> Option<MatrixMode> {
    match s {
        "dd" | "diag" | "diagd" => Some(MatrixMode::DiagDominant),
        "weakdd" | "weak" => Some(MatrixMode::WeakDiagDominant),
        "rand" | "random" => Some(MatrixMode::Random),
        "near_singular" | "ns" | "nearsing" => Some(MatrixMode::NearSingular),
        _ => None,
    }
}

pub fn splitmix64(x: &mut u64) -> u64 {
    *x += 0x9E3779B97F4A7C15u64;
    let mut z = *x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9u64;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBu64;
    z ^ (z >> 31)
}

pub fn u01_from_u64(u: u64) -> f64 {
    let mant = (u >> 11) | 0x3FF0000000000000u64;
    f64::from_bits(mant) - 1.0
}

pub fn val_ij(mut seed: u64, i: i32, j: i32) -> f64 {
    seed ^= ((i + 1) as u64) * 0xD2B74407B1CE6E93u64;
    seed ^= ((j + 1) as u64) * 0xCA5A826395121157u64;
    let u = u01_from_u64(splitmix64(&mut seed));
    2.0 * u - 1.0
}

pub fn base_aij(seed: u64, ms: &MatrixSpec, i: i32, j: i32) -> f64 {
    if ms.mode == MatrixMode::NearSingular {
        let su = seed ^ 0x9A73B52D1F6EB7C1u64;
        let sv = seed ^ 0xC3A5C85C97CB3127u64;
        let u = val_ij(su, i, 0);
        let v = val_ij(sv, 0, j);
        let noise = val_ij(seed, i, j);
        u * v + ms.eps * noise
    } else {
        val_ij(seed, i, j)
    }
}

pub fn diag_adjust(n: i32, ms: &MatrixSpec, row_sum_abs: f64) -> f64 {
    match ms.mode {
        MatrixMode::DiagDominant | MatrixMode::WeakDiagDominant => {
            ms.alpha * row_sum_abs + ms.beta * n as f64
        }
        MatrixMode::NearSingular => ms.beta * n as f64,
        _ => 0.0,
    }
}

pub fn generate_row(
    n: i32,
    seed: u64,
    ms: &MatrixSpec,
    i: i32,
    row_out: &mut [f64],
    b_out: &mut f64,
    row_sum_abs_out: &mut f64,
) {
    *row_sum_abs_out = 0.0;
    for j in 0..n {
        let v = base_aij(seed, ms, i, j);
        row_out[j as usize] = v;
        *row_sum_abs_out += v.abs();
    }
    let diag_add = diag_adjust(n, ms, *row_sum_abs_out);
    row_out[i as usize] += diag_add;
    let sb = seed ^ 0x4F1BBCDCB7A2B3D5u64;
    *b_out = val_ij(sb, i, 0) * 10.0;
}

pub fn compute_residual_inf_norms(
    n: i32,
    seed: u64,
    ms: &MatrixSpec,
    x: &[f64],
    resid_inf_out: &mut f64,
    b_inf_out: &mut f64,
    rel_resid_inf_out: &mut f64,
) {
    *resid_inf_out = 0.0;
    *b_inf_out = 0.0;
    let mut row = vec![0.0; n as usize];
    for i in 0..n {
        let mut bi = 0.0;
        let mut row_sum_abs = 0.0;
        generate_row(n, seed, ms, i, &mut row, &mut bi, &mut row_sum_abs);
        let mut dot = 0.0;
        for j in 0..n {
            dot += row[j as usize] * x[j as usize];
        }
        let ri = dot - bi;
        *resid_inf_out = resid_inf_out.max(ri.abs());
        *b_inf_out = b_inf_out.max(bi.abs());
    }
    *rel_resid_inf_out = if *b_inf_out > 0.0 { *resid_inf_out / *b_inf_out } else { *resid_inf_out };
}

#[derive(Clone, Debug)]
pub struct Dist {
    pub n: i32,
    pub rank: i32,
    pub size: i32,
    pub counts: Vec<i32>,
    pub displs: Vec<i32>,
    pub rows: i32,
    pub row0: i32,
}

impl Dist {
    pub fn new(n: i32, rank: i32, size: i32) -> Self {
        let mut counts = vec![0; size as usize];
        let mut displs = vec![0; size as usize];
        let base = n / size;
        let rem = n % size;
        let mut off = 0;
        for r in 0..size {
            counts[r as usize] = base + if r < rem { 1 } else { 0 };
            displs[r as usize] = off;
            off += counts[r as usize];
        }
        let rows = counts[rank as usize];
        let row0 = displs[rank as usize];
        Self {
            n,
            rank,
            size,
            counts,
            displs,
            rows,
            row0,
        }
    }

    pub fn owner_of_row(&self, global_i: i32) -> i32 {
        for r in 0..self.size {
            let start = self.displs[r as usize];
            let end = start + self.counts[r as usize];
            if global_i >= start && global_i < end {
                return r;
            }
        }
        -1
    }
}

pub fn solve_on_root_inplace_lu(n: i32, lu: &[f64], b: &[f64], x_out: &mut Vec<f64>) {
    let n = n as usize;
    x_out.resize(n, 0.0);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= lu[i * n + j] * y[j];
        }
        y[i] = sum;
    }
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1..n) {
            sum -= lu[i * n + j] * x_out[j];
        }
        let diag = lu[i * n + i];
        x_out[i] = if diag != 0.0 { sum / diag } else { 0.0 };
    }
}

pub fn checksum_weighted_first100(x: &[f64]) -> f64 {
    let m = cmp::min(100, x.len());
    let mut s = 0.0;
    for i in 0..m {
        s += (i as f64 + 1.0) * x[i];
    }
    s
}

#[derive(Parser, Debug, Clone)]
#[command(name = "lu_driver")]
pub struct RunOptions {
    pub n: i32,
    #[arg(long)]
    pub hosts: String,
    pub seed: Option<u64>,
    #[arg(long)]
    pub matrix: Option<String>,
    #[arg(long)]
    pub alpha: Option<f64>,
    #[arg(long)]
    pub beta: Option<f64>,
    #[arg(long)]
    pub eps: Option<f64>,
    #[arg(long)]
    pub threads: Option<i32>,
    #[arg(long)]
    pub timing: bool,
    #[arg(long)]
    pub verify: bool,
    #[arg(long)]
    pub csv: Option<String>,
    #[arg(long)]
    pub keep_workers: bool,
}

pub fn build_spec(opt: &RunOptions) -> MatrixSpec {
    let mut ms = MatrixSpec::default();
    ms.mode = opt.matrix.as_ref().and_then(|s| parse_mode(s)).unwrap_or(MatrixMode::DiagDominant);
    ms.alpha = opt.alpha.unwrap_or(match ms.mode {
        MatrixMode::DiagDominant => 2.0,
        MatrixMode::WeakDiagDominant => 0.2,
        _ => 0.0,
    });
    ms.beta = opt.beta.unwrap_or(if ms.mode == MatrixMode::DiagDominant { 1.0 } else { 0.0 });
    ms.eps = opt.eps.unwrap_or(1e-3);
    ms
}

#[derive(Parser, Debug, Clone)]
#[command(name = "lu_worker")]
pub struct WorkerOptions {
    #[arg(long, default_value = "0.0.0.0:5000")]
    pub listen: String,
    #[arg(long)]
    pub threads: Option<i32>,
}

pub fn csv_exists(path: &str) -> bool {
    Path::new(path).exists()
}

pub fn csv_append_line(path: &str, line: &str) -> Result<()> {
    if path.is_empty() {
        return Ok(());
    }
    let mut f = std::fs::OpenOptions::new().append(true).create(true).open(path)?;
    writeln!(f, "{}", line)?;
    Ok(())
}

pub fn read_hosts_file(path: &str) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let mut line = line?;
        if let Some(pos) = line.find('#') {
            line = line[..pos].to_string();
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(line.to_string());
    }
    Ok(out)
}

pub fn parse_hostport(s: &str) -> (String, u16) {
    if let Some(pos) = s.find(':') {
        let host = s[..pos].to_string();
        let port = s[pos + 1..].parse().unwrap_or(0);
        (host, port)
    } else {
        (s.to_string(), 0)
    }
}

// Authors: Morteza Farrokhnejad, Ali Farrokhnejad