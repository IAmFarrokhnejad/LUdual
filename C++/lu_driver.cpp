// lu_driver.cpp - MPI-lite driver/orchestrator
//
// High-level overview:
//   This program orchestrates a distributed LU factorization over a set of
//   worker processes (lu_worker.exe). It implements a simple RPC protocol
//   over TCP (see lu_net.h) to coordinate pivot search, swaps, pivot-tail
//   broadcasts, and elimination steps. The driver can also run a serial
//   baseline (no workers) by providing an empty hosts file; this uses the
//   identical deterministic matrix generator so the work is algorithmically
//   the same as the distributed case (useful for T1 baseline measurements).
//
//   Important data layout decisions:
//     - The dense matrix A is stored row-major as a flat vector<double> of size n*n.
//     - LU factorization is performed in-place in the same full-matrix layout:
//         * Below-diagonal entries store L multipliers (A[i,k] = L[i,k])
//         * On/above-diagonal entries store U (including diagonal).
//     - Rows are partitioned in contiguous blocks across p workers. The driver's
//       "counts" and "displs" arrays track that partitioning.
//     - All network messages are length-prefixed frames (see net::send_msg / recv_msg).
//
//   Algorithm summary (per k = 0..n-1):
//     1) Pivot scan: ask every worker for the largest |A[i,k]| in its block.
//     2) Select global pivot (max magnitude). If pivot row != k, perform swap:
//           - If both rows owned by same worker -> send LOCAL_SWAP RPC.
//           - Else -> GET_ROW both rows, then PUT_ROW swapped rows back to owners.
//     3) GET_PIVOT_TAIL from owner of pivot row (the entries A[k,k+1..n-1]) and pivot value.
//     4) BCAST_PIVOT_TAIL to all workers so they can perform elimination using that pivot-tail.
//     5) Send ELIMINATE to all workers to instruct them to apply the elimination for column k.
//     6) Repeat next k.
//
//   At the end: GET_BLOCK from each worker to gather final LU and b, then solve on driver
//   using solve_on_root_inplace_LU for the final x. Timing and correctness verification
//   (residuals, checksum) are supported via RunOptions flags.
//
//   Notes on determinism & verification:
//     - A, b are generated deterministically from the seed and MatrixSpec both on
//       driver (serial baseline) and workers (distributed). This allows verification
//       via recomputing A*b and comparing residuals after solve.
//
//   Concurrency & performance:
//     - The driver uses blocking RPCs and waits for workers at each phase.
//     - The workers perform parallel elimination using OpenMP; the driver measures
//       per-phase times and collects worker-local timings to produce simple critical-path estimates.
//
// Usage:
//   lu_driver.exe N --hosts hosts.txt [seed] [--matrix ...] [--alpha a] [--beta b] [--eps e]
//                 [--threads T] [--timing] [--verify] [--csv out.csv]
//
// hosts.txt: one "IP:port" (or hostname:port) per line, empty lines and #comments allowed.
//
// Run model in lab:
//   - Start lu_worker.exe on each machine (same port).
//   - Run lu_driver.exe on the main machine.
// Authors: Morteza Farrokhnejad, Ali Farrokhnejad

#include "lu_common.h"
#include "lu_net.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <omp.h>

// Represents a network connection to a worker (host:port and socket handle).
struct WorkerConn {
    std::string host;
    uint16_t port = 0;
    SOCKET s = INVALID_SOCKET;
}

// Read hosts file that lists one host:port per line. Lines may contain comments
// after a '#' and leading/trailing whitespace is trimmed.
//
// Returns vector<string> of non-empty host:port entries.
// Throws runtime_error if file cannot be opened.
static std::vector<std::string> read_hosts_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open hosts file: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        // strip comments (everything after '#')
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        // trim whitespace on both ends
        auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
        while (!line.empty() && is_ws((unsigned char)line.front())) line.erase(line.begin());
        while (!line.empty() && is_ws((unsigned char)line.back())) line.pop_back();
        if (line.empty()) continue;
        out.push_back(line);
    }
    return out;
}

// Expect an OK response (net::Msg::OK) on socket s. If the reply is ERR,
// decode the error string and throw runtime_error. If the reply is some other
// unexpected message type, also throw.
//
// This helper is used after sending an RPC that doesn't return a payload but
// should confirm success via an OK frame.
static void expect_ok(SOCKET s) {
    net::Msg t; std::vector<uint8_t> payload;
    net::recv_msg(s, t, payload);
    if (t == net::Msg::OK) return;
    if (t == net::Msg::ERR) {
        // ByteReader expects the error payload to contain a uint32 length + bytes
        net::ByteReader r(payload);
        uint32_t len = r.pod<uint32_t>();
        std::string msg(len, '\0');
        if (len) r.bytes(msg.data(), len);
        throw std::runtime_error("Worker ERR: " + msg);
    }
    throw std::runtime_error("Unexpected reply type from worker.");
}

// Perform a simple request/response RPC and return the payload of the expected reply.
// - s: socket to worker
// - req_type: message enum to send
// - req_payload: raw bytes to send as payload
// - expected_reply: message enum expected in reply (if not returned, throw)
// If worker sends ERR we decode and throw runtime_error.
static std::vector<uint8_t> rpc(SOCKET s, net::Msg req_type, const std::vector<uint8_t>& req_payload, net::Msg expected_reply) {
    net::send_msg(s, req_type, req_payload);
    net::Msg rt; std::vector<uint8_t> rp;
    net::recv_msg(s, rt, rp);
    if (rt == expected_reply) return rp;
    if (rt == net::Msg::ERR) {
        net::ByteReader r(rp);
        uint32_t len = r.pod<uint32_t>();
        std::string msg(len, '\0');
        if (len) r.bytes(msg.data(), len);
        throw std::runtime_error("Worker ERR: " + msg);
    }
    throw std::runtime_error("Unexpected reply type from worker.");
}

// Convert RunOptions into a MatrixSpec (small adapter).
static MatrixSpec build_spec(const RunOptions& opt) {
    MatrixSpec ms;
    ms.mode = opt.mode;
    ms.alpha = opt.alpha;
    ms.beta = opt.beta;
    ms.eps = opt.eps;
    return ms;
}

// Append a line to CSV; create file if necessary. Throws on failure.
// If path is empty, do nothing (CSV disabled).
static void csv_append_line(const std::string& path, const std::string& line) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("Failed to open CSV for append: " + path);
    f << line << "\n";
}

// Test if CSV exists (used to write header only once).
static bool csv_exists(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    return (bool)f;
}

// Helper accessors for the flat row-major LU vector (full matrix).
// Using inline helpers makes the code below clearer than repeated index arithmetic.
static inline double& Aat_full(std::vector<double>& A, int n, int i, int j) {
    return A[(size_t)i * (size_t)n + (size_t)j];
}
static inline const double& Aat_full(const std::vector<double>& A, int n, int i, int j) {
    return A[(size_t)i * (size_t)n + (size_t)j];
}

// ---------------------- Serial baseline (0 workers) ----------------------
//
// If hosts.txt is empty (or contains only comments/blank lines), the driver runs a
// single-process LU factorization locally, using the same deterministic matrix generator
// and the same in-place LU format (L multipliers stored below diagonal, U on/above diagonal).
//
// This enables a true "T1" baseline for speedup/efficiency calculations without changing
// the algorithmic work, while removing all network/RPC overhead.
static int run_serial_baseline(const RunOptions& opt, int n, const MatrixSpec& ms) {
    // If the user requested a thread count, set OpenMP threads on the driver.
    if (opt.threads > 0) omp_set_num_threads(opt.threads);

    // Print run configuration (user-friendly header)
    std::cout << "N=" << n << " workers=0 threads=" << (opt.threads > 0 ? opt.threads : 0)
        << " seed=" << opt.seed
        << " matrix=" << mode_name(opt.mode)
        << " alpha=" << ms.alpha
        << " beta=" << ms.beta
        << " eps=" << ms.eps
        << "\n";
    std::cout << "System: A x = b (dense), generated deterministically from seed.\n";
    if (opt.mode == MatrixMode::Random) {
        std::cout << "A_ij in [-1,1), no diagonal boost (Random mode).\n";
    }
    else if (opt.mode == MatrixMode::NearSingular) {
        std::cout << "A_ij in [-1,1), near-singular structure with eps noise; diagonal shift uses beta*N.\n";
    }
    else {
        std::cout << "A_ij in [-1,1), diagonal adjusted as: A_ii += alpha*sum_j|A_ij| + beta*N.\n";
    }

    // If CSV requested and file doesn't exist, emit header.
    if (!opt.csv_path.empty() && !csv_exists(opt.csv_path)) {
        csv_append_line(opt.csv_path,
            "N,p,threads,seed,matrix,alpha,beta,eps,"
            "T_total,T_pivot,T_swap,T_getpivot,T_bcast_send,T_elim,T_gather,T_solve,"
            "swap_total,swap_cross,swap_bytes,bcast_bytes_sent,"
            "rel_resid,checksum,critical_fact");
    }

    // Driver-side timing counters. For serial baseline, comm-related fields remain zero.
    double t_pivot = 0.0;
    double t_swap = 0.0;
    double t_getpivot = 0.0;
    double t_bcast_send = 0.0;
    double t_elim = 0.0;
    double t_gather = 0.0;
    double t_solve = 0.0;

    // Swap / communication counters - serial baseline only estimates some stats.
    uint64_t swap_total = 0;
    uint64_t swap_cross = 0;
    uint64_t swap_bytes = 0;
    uint64_t bcast_bytes_sent = 0;

    // Allocate full LU (n*n) and b vector and fill using deterministic generator.
    // This matches what worker::init_storage() does for the distributed case.
    std::vector<double> LU((size_t)n * (size_t)n, 0.0);
    std::vector<double> b_full((size_t)n, 0.0);
    {
        std::vector<double> row((size_t)n);
        for (int i = 0; i < n; ++i) {
            double bi = 0.0, sumabs = 0.0;
            // populate row and corresponding b_i using the same generator as workers
            generate_row(n, opt.seed, ms, i, row.data(), bi, sumabs);
            std::copy(row.begin(), row.end(), &LU[(size_t)i * (size_t)n]);
            b_full[(size_t)i] = bi;
        }
    }

    // Start overall timer for factorization.
    double t0_all = omp_get_wtime();

    // In-place LU factorization with partial pivoting.
    // For each pivot column k:
    //   - find pivot row (max |A[i,k]| for i in k..n-1)
    //   - swap rows if needed (and swap b entries)
    //   - compute L multipliers for rows i>k and update row entries j>k
    for (int k = 0; k < n; ++k) {
        // Pivot scan: linear search for max abs pivot in column k
        double t0 = omp_get_wtime();
        double best = 0.0;
        int best_gi = -1;
        for (int i = k; i < n; ++i) {
            double v = std::abs(Aat_full(LU, n, i, k));
            if (v > best) { best = v; best_gi = i; }
        }
        double t1 = omp_get_wtime();
        t_pivot += (t1 - t0);

        // If no pivot found or pivot is exactly zero then matrix is singular.
        if (best_gi < 0 || best == 0.0) {
            throw std::runtime_error("Matrix appears singular at k=" + std::to_string(k) + " (best pivot=0).");
        }

        // Swap rows k and best_gi (both A and b).
        t0 = omp_get_wtime();
        if (best_gi != k) {
            ++swap_total;
            for (int j = 0; j < n; ++j) {
                std::swap(Aat_full(LU, n, k, j), Aat_full(LU, n, best_gi, j));
            }
            std::swap(b_full[(size_t)k], b_full[(size_t)best_gi]);
        }
        t1 = omp_get_wtime();
        t_swap += (t1 - t0);

        // Elimination step: compute multipliers and update trailing submatrix.
        // Using OpenMP parallel for across rows i=k+1..n-1. Inner j-loop is vector-friendly.
        double pivot = Aat_full(LU, n, k, k);
        if (pivot == 0.0) {
            throw std::runtime_error("Matrix appears singular at k=" + std::to_string(k) + " (pivot=0).");
        }

        const double* prow = &LU[(size_t)k * (size_t)n];

        t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; ++i) {
            double* rowi = &LU[(size_t)i * (size_t)n];
            double aik = rowi[(size_t)k] / pivot;
            rowi[(size_t)k] = aik; // store L multiplier in lower part
#pragma omp simd
            for (int j = k + 1; j < n; ++j) {
                rowi[(size_t)j] -= aik * prow[(size_t)j];
            }
        }
        t1 = omp_get_wtime();
        t_elim += (t1 - t0);
    }

    double t_fact = omp_get_wtime() - t0_all;

    // After factorization, perform forward/back substitution to compute x on driver.
    double ts0 = omp_get_wtime();
    std::vector<double> x;
    solve_on_root_inplace_LU(n, LU, b_full, x);
    double ts1 = omp_get_wtime();
    t_solve += (ts1 - ts0);

    // Weighted checksum over first 100 elements to give a compact correctness digest.
    double chk = checksum_weighted_first100(x);

    std::cout << "LU factorization wall time (driver): " << std::setprecision(6) << t_fact << " s\n";
    std::cout << "x checksum (first 100 weighted): " << std::setprecision(12) << chk << "\n";

    // If timing requested, print breakdowns and counters (pivot/elim/swap etc).
    if (opt.timing) {
        std::cout << "\nTiming breakdown (driver, seconds):\n";
        std::cout << "  pivot (scan maxloc):    " << t_pivot << "\n";
        std::cout << "  swap (row exchanges):   " << t_swap << "\n";
        std::cout << "  getpivot (rpc tail):    " << t_getpivot << "\n";
        std::cout << "  bcast (send+wait):      " << t_bcast_send << "\n";
        std::cout << "  bcast total:            " << (t_getpivot + t_bcast_send) << "\n";
        std::cout << "  elim (compute):         " << t_elim << "\n";
        std::cout << "Post-factorization:\n";
        std::cout << "  gather (LU,b):          " << t_gather << "\n";
        std::cout << "  solve (driver):         " << t_solve << "\n";

        std::cout << "\nSwap/comm counters (driver estimates):\n";
        std::cout << "  swaps_total:            " << swap_total << "\n";
        std::cout << "  swaps_cross_worker:     " << swap_cross << "\n";
        std::cout << "  swap_bytes (est):       " << swap_bytes << "\n";
        std::cout << "  bcast_bytes_sent (est): " << bcast_bytes_sent << "\n";
    }

    // If verify requested, compute residual norms by regenerating A,b and computing ||Ax-b||_inf.
    double rel = 0.0;
    if (opt.verify) {
        double rinf = 0.0, binf = 0.0, rrel = 0.0;
        compute_residual_inf_norms(n, opt.seed, ms, x, rinf, binf, rrel);
        rel = rrel;
        std::cout << "\nVerification (re-generated A,b on driver):\n";
        std::cout << "  ||Ax-b||_inf: " << std::setprecision(12) << rinf << "\n";
        std::cout << "  ||b||_inf:    " << std::setprecision(12) << binf << "\n";
        std::cout << "  rel residual: " << std::setprecision(12) << rrel << "\n";
    }

    // If CSV requested, append a line summarizing results for this run (useful for experiments).
    if (!opt.csv_path.empty()) {
        std::ostringstream line;
        line << n << "," << 0 << "," << opt.threads << "," << opt.seed << "," << mode_name(opt.mode) << ","
            << ms.alpha << "," << ms.beta << "," << ms.eps << ","
            << t_fact << "," << t_pivot << "," << t_swap << "," << t_getpivot << "," << t_bcast_send << "," << t_elim << ","
            << t_gather << "," << t_solve << ","
            << swap_total << "," << swap_cross << "," << swap_bytes << "," << bcast_bytes_sent << ","
            << rel << "," << chk << "," << t_fact; // critical_fact == t_fact (no workers)
        csv_append_line(opt.csv_path, line.str());
    }

    if (opt.timing) {
        std::cout << "\nEstimated critical-path factorization time: " << t_fact << " s\n";
    }

    return 0;
}

// ---------------------- main (driver orchestrator) ----------------------
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    RunOptions opt;
    std::string err;
    // Parse driver CL args (populates opt). parse_driver_args returns false on error and sets err.
    if (!parse_driver_args(argc, argv, opt, err)) {
        std::cerr << err << "\n";
        return 2;
    }

    const int n = opt.n;
    const MatrixSpec ms = build_spec(opt);

    try {
        // Initialize Winsock (RAII wrapper) - required on Windows.
        net::WSAInit wsa;

        // Load hosts.txt entries. If empty -> run serial baseline.
        auto hosts = read_hosts_file(opt.hosts_path);
        if (hosts.empty()) {
            return run_serial_baseline(opt, n, ms);
        }
        const int p = (int)hosts.size();

        // Connect to all workers; create WorkerConn array.
        std::vector<WorkerConn> conns((size_t)p);
        for (int r = 0; r < p; ++r) {
            auto hp = net::parse_hostport(hosts[(size_t)r]);
            if (hp.second == 0) throw std::runtime_error("Host entry must be host:port, got: " + hosts[(size_t)r]);
            conns[(size_t)r].host = hp.first;
            conns[(size_t)r].port = hp.second;
            // Establish TCP connection; throws on failure.
            conns[(size_t)r].s = net::connect_tcp(hp.first, hp.second);
        }

        // Compute simple block distribution counts[] and displs[] for row partitioning:
        // rows are assigned in contiguous blocks as evenly as possible.
        std::vector<int> counts((size_t)p), displs((size_t)p);
        {
            int base = n / p;
            int rem = n % p;
            int off = 0;
            for (int r = 0; r < p; ++r) {
                counts[(size_t)r] = base + (r < rem ? 1 : 0);
                displs[(size_t)r] = off;
                off += counts[(size_t)r];
            }
        }
        // owner_of_row lambda returns the worker index owning a given global row gi.
        auto owner_of_row = [&](int gi) -> int {
            for (int r = 0; r < p; ++r) {
                int start = displs[(size_t)r];
                int end = start + counts[(size_t)r];
                if (gi >= start && gi < end) return r;
            }
            return -1;
        };

        // INIT all workers: send them their block metadata + global parameters so they can
        // initialize their local A and b deterministically and allocate storage.
        for (int r = 0; r < p; ++r) {
            net::ByteWriter w;
            w.pod((int32_t)n);                       // global matrix size
            w.pod((int32_t)r);                       // this worker's rank
            w.pod((int32_t)p);                       // total workers
            w.pod((int32_t)displs[(size_t)r]);       // row0 (first global row owned by this worker)
            w.pod((int32_t)counts[(size_t)r]);       // rows (number of rows owned)
            w.pod((uint64_t)opt.seed);               // RNG seed for deterministic A,b
            w.pod((int32_t)opt.mode);                // matrix mode enum as int32
            w.pod((double)ms.alpha);                 // diag adjust alpha
            w.pod((double)ms.beta);                  // diag adjust beta
            w.pod((double)ms.eps);                   // noise epsilon for near_singular mode
            w.pod((int32_t)opt.threads);             // thread hint for worker OpenMP
            net::send_msg(conns[(size_t)r].s, net::Msg::INIT, w.buf);
        }
        // Expect OK from each worker for INIT; expect_ok decodes ERR to exception.
        for (int r = 0; r < p; ++r) expect_ok(conns[(size_t)r].s);

        // Print driver run header (mirrors serial baseline header).
        std::cout << "N=" << n << " workers=" << p << " threads=" << (opt.threads > 0 ? opt.threads : 0)
            << " seed=" << opt.seed
            << " matrix=" << mode_name(opt.mode)
            << " alpha=" << ms.alpha
            << " beta=" << ms.beta
            << " eps=" << ms.eps
            << "\n";
        std::cout << "System: A x = b (dense), generated deterministically from seed.\n";
        if (opt.mode == MatrixMode::Random) {
            std::cout << "A_ij in [-1,1), no diagonal boost (Random mode).\n";
        }
        else if (opt.mode == MatrixMode::NearSingular) {
            std::cout << "A_ij in [-1,1), near-singular structure with eps noise; diagonal shift uses beta*N.\n";
        }
        else {
            std::cout << "A_ij in [-1,1), diagonal adjusted as: A_ii += alpha*sum_j|A_ij| + beta*N.\n";
        }

        // If CSV requested and file doesn't exist, write header.
        if (!opt.csv_path.empty() && !csv_exists(opt.csv_path)) {
            csv_append_line(opt.csv_path,
                "N,p,threads,seed,matrix,alpha,beta,eps,"
                "T_total,T_pivot,T_swap,T_getpivot,T_bcast_send,T_elim,T_gather,T_solve,"
                "swap_total,swap_cross,swap_bytes,bcast_bytes_sent,"
                "rel_resid,checksum,critical_fact");
        }

        // Driver-side aggregated timers and counters (these include driver-local time spent,
        // not worker-side compute time which is collected separately).
        double t_pivot = 0.0;
        double t_swap = 0.0;
        double t_getpivot = 0.0;   // GET_PIVOT_TAIL RPC time on driver
        double t_bcast_send = 0.0; // driver send+wait time for pivot-tail broadcast
        double t_elim = 0.0;
        double t_gather = 0.0;
        double t_solve = 0.0;

        uint64_t swap_total = 0;
        uint64_t swap_cross = 0;
        uint64_t swap_bytes = 0;            // estimate of bytes moved due to cross-worker swaps
        uint64_t bcast_bytes_sent = 0;      // estimate of driver->workers bytes for pivot broadcast

        double t0_all = omp_get_wtime();

        // Temp buffer to receive pivot tail from the owner and reuse for broadcast.
        std::vector<double> tail((size_t)n);

        // Main factorization loop: iterate over pivot column k.
        for (int k = 0; k < n; ++k) {
            // Pivot scan: instruct every worker to scan its local block and reply with
            // the local maximum absolute pivot magnitude and the global row index.
            double t0 = omp_get_wtime();
            for (int r = 0; r < p; ++r) {
                net::ByteWriter w; w.pod((int32_t)k);
                net::send_msg(conns[(size_t)r].s, net::Msg::PIVOT_SCAN, w.buf);
            }
            double best = 0.0;
            int best_gi = -1;

            // Collect replies serially (driver expects exactly one reply per worker).
            for (int r = 0; r < p; ++r) {
                net::Msg rt; std::vector<uint8_t> rp;
                net::recv_msg(conns[(size_t)r].s, rt, rp);
                if (rt == net::Msg::ERR) {
                    net::ByteReader rr(rp);
                    uint32_t len = rr.pod<uint32_t>();
                    std::string msg(len, '\0');
                    if (len) rr.bytes(msg.data(), len);
                    throw std::runtime_error("Worker ERR during pivot scan: " + msg);
                }
                if (rt != net::Msg::PIVOT_SCAN) throw std::runtime_error("Unexpected pivot scan reply type");
                net::ByteReader rr(rp);
                double loc = rr.pod<double>();        // local max absolute pivot
                int gi = rr.pod<int32_t>();          // global row index for that local max
                if (loc > best) { best = loc; best_gi = gi; }
            }
            double t1 = omp_get_wtime();
            t_pivot += (t1 - t0);

            // Sanity check for singular matrix (no viable pivot).
            if (best_gi < 0 || best == 0.0) {
                throw std::runtime_error("Matrix appears singular at k=" + std::to_string(k) + " (best pivot=0).");
            }

            // Swap rows if pivot row differs from k. The b vector is swapped by workers too.
            t0 = omp_get_wtime();
            if (best_gi != k) {
                ++swap_total;
                int owner_k = owner_of_row(k);
                int owner_p = owner_of_row(best_gi);
                if (owner_k < 0 || owner_p < 0) throw std::runtime_error("owner_of_row failed");

                if (owner_k == owner_p) {
                    // Local swap inside the same worker: send LOCAL_SWAP which instructs the worker
                    // to exchange two local rows and their b entries (fast, no bulk transfer).
                    net::ByteWriter w; w.pod((int32_t)k); w.pod((int32_t)best_gi);
                    net::send_msg(conns[(size_t)owner_k].s, net::Msg::LOCAL_SWAP, w.buf);
                    expect_ok(conns[(size_t)owner_k].s);
                }
                else {
                    // Cross-worker swap: driver must fetch both rows (GET_ROW), then send them back
                    // to the opposite owners (PUT_ROW). This moves O(n) doubles per row over TCP.
                    ++swap_cross;
                    // Estimate traffic: 2x GET_ROW (worker->driver) + 2x PUT_ROW (driver->worker)
                    // Payload per row message: int32 gi + double b + n doubles
                    const uint64_t row_payload = (uint64_t)sizeof(int32_t) + (uint64_t)sizeof(double) + (uint64_t)n * (uint64_t)sizeof(double);
                    swap_bytes += 4ULL * row_payload;

                    // GET_ROW for row k from owner_k
                    net::ByteWriter wk; wk.pod((int32_t)k);
                    auto rk = rpc(conns[(size_t)owner_k].s, net::Msg::GET_ROW, wk.buf, net::Msg::GET_ROW);
                    net::ByteReader rrk(rk);
                    (void)rrk.pod<int32_t>(); // gi (ignored)
                    double bk = rrk.pod<double>();
                    std::vector<double> rowk((size_t)n);
                    rrk.bytes(rowk.data(), (size_t)n * sizeof(double));

                    // GET_ROW for best_gi from owner_p
                    net::ByteWriter wp; wp.pod((int32_t)best_gi);
                    auto rp = rpc(conns[(size_t)owner_p].s, net::Msg::GET_ROW, wp.buf, net::Msg::GET_ROW);
                    net::ByteReader rrp(rp);
                    (void)rrp.pod<int32_t>();
                    double bp = rrp.pod<double>();
                    std::vector<double> rowp((size_t)n);
                    rrp.bytes(rowp.data(), (size_t)n * sizeof(double));

                    // PUT swapped rows: put row_p into owner_k at index k, and row_k into owner_p at best_gi.
                    net::ByteWriter pk; pk.pod((int32_t)k); pk.pod(bp); pk.bytes(rowp.data(), (size_t)n * sizeof(double));
                    net::send_msg(conns[(size_t)owner_k].s, net::Msg::PUT_ROW, pk.buf);
                    expect_ok(conns[(size_t)owner_k].s);

                    net::ByteWriter pp; pp.pod((int32_t)best_gi); pp.pod(bk); pp.bytes(rowk.data(), (size_t)n * sizeof(double));
                    net::send_msg(conns[(size_t)owner_p].s, net::Msg::PUT_ROW, pp.buf);
                    expect_ok(conns[(size_t)owner_p].s);
                }
            }
            t1 = omp_get_wtime();
            t_swap += (t1 - t0);

            // After swap, the pivot row k is owned by owner_of_row(k) (could be same as earlier).
            int owner_k = owner_of_row(k);

            // GET_PIVOT_TAIL: ask the owner to return:
            //   (int32 k), (double pivot), (int32 tail_len), (tail_len doubles = A[k,k+1..n-1])
            // The driver uses this to broadcast to all workers.
            t0 = omp_get_wtime();
            {
                net::ByteWriter w; w.pod((int32_t)k);
                auto rp = rpc(conns[(size_t)owner_k].s, net::Msg::GET_PIVOT_TAIL, w.buf, net::Msg::GET_PIVOT_TAIL);
                net::ByteReader r(rp);
                int kk = r.pod<int32_t>();
                double pivot = r.pod<double>();
                int tail_len = r.pod<int32_t>();
                if (kk != k) throw std::runtime_error("GET_PIVOT_TAIL: k mismatch");
                if (tail_len != n - (k + 1)) throw std::runtime_error("GET_PIVOT_TAIL: tail_len mismatch");
                if (tail_len > 0) r.bytes(&tail[(size_t)k + 1], (size_t)tail_len * sizeof(double));

                t1 = omp_get_wtime();
                t_getpivot += (t1 - t0);

                // Broadcast the pivot tail to every worker (BCAST_PIVOT_TAIL).
                // We measure the driver's portion of this broadcast (send+wait for OKs).
                double tb0 = omp_get_wtime();

                net::ByteWriter wb;
                wb.pod((int32_t)k);
                wb.pod((double)pivot);
                wb.pod((int32_t)tail_len);
                if (tail_len > 0) wb.bytes(&tail[(size_t)k + 1], (size_t)tail_len * sizeof(double));

                // Estimate bytes sent by driver to workers for this broadcast (for reporting).
                // Payload: k(int32) + pivot(double) + tail_len(int32) + tail_len*doubles
                const uint64_t bcast_payload =
                    (uint64_t)sizeof(int32_t) +
                    (uint64_t)sizeof(double) +
                    (uint64_t)sizeof(int32_t) +
                    (uint64_t)tail_len * (uint64_t)sizeof(double);
                bcast_bytes_sent += (uint64_t)p * bcast_payload;

                // Send BCAST_PIVOT_TAIL to all workers and wait for OKs.
                for (int rnk = 0; rnk < p; ++rnk) {
                    net::send_msg(conns[(size_t)rnk].s, net::Msg::BCAST_PIVOT_TAIL, wb.buf);
                }
                for (int rnk = 0; rnk < p; ++rnk) expect_ok(conns[(size_t)rnk].s);

                double tb1 = omp_get_wtime();
                t_bcast_send += (tb1 - tb0);
            }

            // Instruct all workers to perform elimination for column k. Each worker will
            // use its local portion of column k and the received pivot tail to update its rows.
            t0 = omp_get_wtime();
            for (int rnk = 0; rnk < p; ++rnk) {
                net::ByteWriter we; we.pod((int32_t)k);
                net::send_msg(conns[(size_t)rnk].s, net::Msg::ELIMINATE, we.buf);
            }
            for (int rnk = 0; rnk < p; ++rnk) expect_ok(conns[(size_t)rnk].s);
            t1 = omp_get_wtime();
            t_elim += (t1 - t0);
        }

        double t_fact = omp_get_wtime() - t0_all;

        // Post-factorization: Gather LU and b blocks from each worker to the driver.
        double tg0 = omp_get_wtime();
        std::vector<double> LU((size_t)n * (size_t)n);
        std::vector<double> b_full((size_t)n);

        for (int r = 0; r < p; ++r) {
            auto rp = rpc(conns[(size_t)r].s, net::Msg::GET_BLOCK, {}, net::Msg::GET_BLOCK);
            net::ByteReader rr(rp);
            int row0 = rr.pod<int32_t>();
            int rows = rr.pod<int32_t>();
            if (row0 != displs[(size_t)r] || rows != counts[(size_t)r]) {
                throw std::runtime_error("GET_BLOCK: distribution mismatch from worker " + std::to_string(r));
            }
            // Read LU block (rows * n doubles) then b block (rows doubles).
            rr.bytes(&LU[(size_t)row0 * (size_t)n], (size_t)rows * (size_t)n * sizeof(double));
            rr.bytes(&b_full[(size_t)row0], (size_t)rows * sizeof(double));
        }
        double tg1 = omp_get_wtime();
        t_gather += (tg1 - tg0);

        // Solve on driver using assembled LU and b (forward/back substitution).
        double ts0 = omp_get_wtime();
        std::vector<double> x;
        solve_on_root_inplace_LU(n, LU, b_full, x);
        double ts1 = omp_get_wtime();
        t_solve += (ts1 - ts0);

        double chk = checksum_weighted_first100(x);

        std::cout << "LU factorization wall time (driver): " << std::setprecision(6) << t_fact << " s\n";
        std::cout << "x checksum (first 100 weighted): " << std::setprecision(12) << chk << "\n";

        // Print driver timing breakdown similar to serial baseline but including comm costs.
        if (opt.timing) {
            std::cout << "\nTiming breakdown (driver, seconds):\n";
            std::cout << "  pivot (gather maxloc):  " << t_pivot << "\n";
            std::cout << "  swap (row exchanges):   " << t_swap << "\n";
            std::cout << "  getpivot (rpc tail):    " << t_getpivot << "\n";
            std::cout << "  bcast (send+wait):      " << t_bcast_send << "\n";
            std::cout << "  bcast total:            " << (t_getpivot + t_bcast_send) << "\n";
            std::cout << "  elim (coord+compute):   " << t_elim << "\n";
            std::cout << "Post-factorization:\n";
            std::cout << "  gather (LU,b):          " << t_gather << "\n";
            std::cout << "  solve (driver):         " << t_solve << "\n";

            std::cout << "\nSwap/comm counters (driver estimates):\n";
            std::cout << "  swaps_total:            " << swap_total << "\n";
            std::cout << "  swaps_cross_worker:     " << swap_cross << "\n";
            std::cout << "  swap_bytes (est):       " << swap_bytes << "\n";
            std::cout << "  bcast_bytes_sent (est): " << bcast_bytes_sent << "\n";
        }

        // If verify requested, compute residuals as in serial baseline (regenerated A,b).
        double rel = 0.0;
        if (opt.verify) {
            double rinf = 0.0, binf = 0.0, rrel = 0.0;
            compute_residual_inf_norms(n, opt.seed, ms, x, rinf, binf, rrel);
            rel = rrel;
            std::cout << "\nVerification (re-generated A,b on driver):\n";
            std::cout << "  ||Ax-b||_inf: " << std::setprecision(12) << rinf << "\n";
            std::cout << "  ||b||_inf:    " << std::setprecision(12) << binf << "\n";
            std::cout << "  rel residual: " << std::setprecision(12) << rrel << "\n";
        }

        // Pull worker stats (scan/recv/elim times and byte counters) for better diagnostics.
        double max_worker = 0.0;
        double scan_max = 0.0, recv_max = 0.0, elim_max = 0.0;
        uint64_t bytes_in_sum = 0, bytes_out_sum = 0;
        if (opt.timing) {
            std::cout << "\nWorker timing (seconds):\n";
            for (int r = 0; r < p; ++r) {
                auto rp = rpc(conns[(size_t)r].s, net::Msg::GET_STATS, {}, net::Msg::GET_STATS);
                net::ByteReader rr(rp);
                double sscan = rr.pod<double>();
                double srecv = rr.pod<double>();
                double selim = rr.pod<double>();
                uint64_t bin = rr.pod<uint64_t>();
                uint64_t bout = rr.pod<uint64_t>();
                double tot = sscan + srecv + selim;
                max_worker = std::max(max_worker, tot);
                scan_max = std::max(scan_max, sscan);
                recv_max = std::max(recv_max, srecv);
                elim_max = std::max(elim_max, selim);
                bytes_in_sum += bin;
                bytes_out_sum += bout;
                std::cout << "  worker[" << r << "] scan=" << sscan << " recv=" << srecv << " elim=" << selim
                    << "  bytes_in=" << bin << " bytes_out=" << bout << "\n";
            }

            // Phase-level critical path estimates (best-effort) combine driver timings
            // and the maximum worker time for that phase. These are not strict proofs of
            // critical path but are useful heuristics for performance analysis.
            const double crit_pivot = std::max(t_pivot, scan_max);
            const double crit_swap = t_swap; // driver swap time used as estimate
            const double crit_bcast = std::max(t_bcast_send, recv_max) + t_getpivot; // GET_PIVOT_TAIL is driver-only
            const double crit_elim = std::max(t_elim, elim_max);
            std::cout << "\nCritical-path estimates (seconds):\n";
            std::cout << "  pivot  crit=" << crit_pivot << " (driver=" << t_pivot << ", workers_scan_max=" << scan_max << ")\n";
            std::cout << "  swap   crit=" << crit_swap << " (driver=" << t_swap << ")\n";
            std::cout << "  bcast  crit=" << crit_bcast << " (getpivot=" << t_getpivot << ", driver_send=" << t_bcast_send
                << ", workers_recv_max=" << recv_max << ")\n";
            std::cout << "  elim   crit=" << crit_elim << " (driver=" << t_elim << ", workers_elim_max=" << elim_max << ")\n";
            std::cout << "\nWorker traffic summary (counters from workers):\n";
            std::cout << "  bytes_in_sum=" << bytes_in_sum << " bytes_out_sum=" << bytes_out_sum << "\n";
        }

        // Append final CSV line if requested: includes times, counters and a critical-path estimate
        if (!opt.csv_path.empty()) {
            std::ostringstream line;
            line << n << "," << p << "," << opt.threads << "," << opt.seed << "," << mode_name(opt.mode) << ","
                << ms.alpha << "," << ms.beta << "," << ms.eps << ","
                << t_fact << "," << t_pivot << "," << t_swap << "," << t_getpivot << "," << t_bcast_send << "," << t_elim << ","
                << t_gather << "," << t_solve << ","
                << swap_total << "," << swap_cross << "," << swap_bytes << "," << bcast_bytes_sent << ","
                << rel << "," << chk << ",";

            // critical-path factorization estimate (max of driver and slowest worker).
            double critical_fact = std::max(t_fact, max_worker);
            line << critical_fact;
            csv_append_line(opt.csv_path, line.str());
        }

        // Shutdown workers unless user passed --keep-workers. When driver sends SHUTDOWN,
        // each worker replies with OK and then exits its process (see worker main).
        if (!opt.keep_workers) {
            for (int r = 0; r < p; ++r) net::send_msg(conns[(size_t)r].s, net::Msg::SHUTDOWN, {});
            for (int r = 0; r < p; ++r) expect_ok(conns[(size_t)r].s);
        }
        else {
            std::cout << "\nNote: workers kept alive (--keep-workers); remember to shut them down manually.\n";
        }
        // Close all sockets cleanly.
        for (auto& c : conns) net::closesocket_safe(c.s);

        // Final reported critical-path estimate for the run.
        if (opt.timing) {
            double critical = std::max(t_fact, max_worker);
            std::cout << "\nEstimated critical-path factorization time: " << critical << " s\n";
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "lu_driver fatal: " << e.what() << "\n";
        return 1;
    }
}
