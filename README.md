# LUdual: Dual-Language Distributed LU Factorization Solver


## Overview

LUdual is a bilingual implementation of a distributed LU factorization solver for dense linear systems (Ax = b), featuring both C++ and Rust versions. Built on an MPI-lite protocol over TCP, it orchestrates factorization across worker nodes using deterministic matrix generation for reproducibility and verification. The project supports various matrix modes (diagonally dominant, random, near-singular) and includes serial baselines for performance comparisons.

## Features

- **Distributed Factorization**: Row-block partitioning across workers with RPC-based coordination for pivoting, swaps, broadcasts, and elimination.
- **Deterministic Generation**: Matrices and RHS vectors generated from seeds for identical results in serial vs. distributed runs.
- **Matrix Modes**: Diagonally dominant (`dd`), weakly dominant (`weakdd`), random (`rand`), near-singular (`near_singular`).
- **Verification**: Residual norms and checksums to ensure correctness.
- **Timing & Metrics**: Detailed breakdowns for pivot, swap, broadcast, elimination phases; CSV logging for experiments.
- **Dual Implementations**:
  - **C++**: Uses Winsock for networking, OpenMP for parallelism; battle-tested for Windows labs.
  - **Rust**: Leverages std::net, Rayon for threads; emphasizes safety but may require tuning for equivalent perf.
- **Cross-Worker Swaps**: Efficient local vs. remote row exchanges.
- **Serial Baseline**: Run without workers for T1 measurements.


## Prerequisites

- **C++ Version**:
  - Compiler: MSVC (Windows) or equivalent with C++11+.
  - Libraries: OpenMP (optional for parallelism).
- **Rust Version**:
  - Rust: 1.75+ (via rustup).
  - Crates: rayon, clap, anyhow, byteorder (via Cargo.toml).
- **General**: TCP network access; hosts.txt for worker IPs/ports.

## Installation

Clone the repo:
```
git clone https://github.com/IAmFarrokhnejad/LUdual.git
cd LUdual
```

### Building C++
- Use Visual Studio or CMake for build.
- Example (MSVC):
  ```
  cl /EHsc /openmp lu_driver.cpp lu_worker.cpp lu_common.cpp
  ```
- Outputs: `lu_driver.exe`, `lu_worker.exe`.

### Building Rust
- Install dependencies:
  ```
  cargo build --release
  ```
- Binaries: `target/release/driver`, `target/release/worker`.


## Usage

### Hosts File
Create `hosts.txt` with worker addresses (one per line, e.g., `192.168.1.10:5000`).

### Running Workers
- **C++**: `lu_worker.exe --listen 0.0.0.0:5000 --threads 4`
- **Rust**: `cargo run --bin worker -- --listen 0.0.0.0:5000 --threads 4`

Start on each node.

### Running Driver
- **C++**: `lu_driver.exe 1000 --hosts hosts.txt --seed 123 --matrix dd --threads 4 --timing --verify --csv results.csv`
- **Rust**: `cargo run --bin driver -- 1000 --hosts hosts.txt --seed 123 --matrix dd --threads 4 --timing --verify --csv results.csv`

For serial: Empty `hosts.txt`.

**Options**:
- `N`: Matrix size.
- `--matrix`: Mode (dd, weakdd, rand, near_singular).
- `--alpha`, `--beta`, `--eps`: Diagonal adjustments.
- `--keep-workers`: Don't shutdown workers post-run.


## Examples

- Diagonal Dominant (100x100, 2 workers): Run workers on ports 5000/5001, then driver with hosts.txt.
- Verification: Check residuals <1e-10 for well-conditioned matrices.
- CSV Output: For scaling studies (N, p, times, swaps).

## Performance Notes

- C++: Optimized with OpenMP; expect better vectorization.
- Rust: Rayon parallelism; safer but profile for bottlenecks.
- Benchmarks (as of Jan 2026): Test on your hardware—Rust might lag in tight loops due to bounds checks.


## Contributing

Pull requests welcome! Focus on:
- Porting to Linux (epoll for Rust, POSIX sockets for C++).

Follow standard Git flow; include benchmarks in PRs.

## License

MIT License. See [LICENSE](LICENSE) for details.