// lu_worker.cpp - MPI-lite worker process
//
// Overview:
//   A worker process holds a contiguous block of rows [row0, row0+rows) of the global
//   dense matrix A and vector b. It initializes its local rows deterministically
//   using the same seed and MatrixSpec as the driver so that distributed and serial
//   runs are comparable and verifiable. The worker responds to driver RPCs to:
//     - report local pivot maxima (PIVOT_SCAN)
//     - exchange rows (GET_ROW, PUT_ROW, LOCAL_SWAP)
//     - provide pivot-tail (GET_PIVOT_TAIL)
//     - receive pivot-tail broadcasts (BCAST_PIVOT_TAIL)
//     - execute elimination (ELIMINATE)
//     - return its full block (GET_BLOCK) and lightweight statistics (GET_STATS)
//     - shutdown (SHUTDOWN)
//
//   The worker keeps simple timing counters and byte counters for diagnostic output.
//   Parallel compute within a worker (the elimination loops) uses OpenMP threads.
//
// Usage (run on every compute node):
//   lu_worker.exe --listen 0.0.0.0:5000
//
// Important session semantics:
//   - The worker accepts a driver connection and serves a full factorization session.
//   - If the driver uses --keep-workers, it may close the connection at the end of a run
//     (peer closed connection). The worker treats that as end-of-session and returns to accept().
//   - SHUTDOWN causes the worker to reply OK and then exit the process.
// Authors: Morteza Farrokhnejad, Ali Farrokhnejad

#include "lu_common.h"
#include "lu_net.h"

#include <iostream>
#include <fstream>
#include <omp.h>

// Local accessor helpers for the worker's local matrix A which is stored row-major
// in a flat vector of length rows * n. 'li' is the local row index 0..rows-1, 'j' is column index.
static inline double& Aat(std::vector<double>& A, int n, int li, int j) {
    return A[(size_t)li * (size_t)n + (size_t)j];
}
static inline const double& Aat(const std::vector<double>& A, int n, int li, int j) {
    return A[(size_t)li * (size_t)n + (size_t)j];
}

// Per-connection worker state maintained across RPCs for a session.
struct WorkerState {
    int n = 0;       // global matrix dimension
    int rank = 0;    // this worker's rank (0..p-1)
    int p = 1;       // total workers
    int row0 = 0;    // first global row index owned by this worker
    int rows = 0;    // number of rows owned by this worker

    uint64_t seed = 0;   // RNG seed used to generate rows
    MatrixSpec ms{};     // matrix generation parameters
    int threads = 0;     // OpenMP thread hint

    std::vector<double> A; // local block: rows*n doubles (row-major)
    std::vector<double> b; // local RHS block: rows doubles

    // Current pivot info for the most recent broadcast step:
    int cur_k = -1;                // pivot column/row index corresponding to last BCAST
    double cur_pivot = 0.0;        // pivot value A[k,k] received via GET_PIVOT_TAIL
    std::vector<double> pivot_tail; // full-size vector where only [k+1..n-1] are valid entries

    // Lightweight timing & traffic counters (used for driver diagnostics).
    double t_scan = 0.0;        // sum of local pivot scan times
    double t_elim = 0.0;        // sum of elimination times
    double t_recv_pivot = 0.0;  // sum of times taken to receive pivot tail (BCAST)
    uint64_t bytes_in = 0;      // bytes received (framed bytes)
    uint64_t bytes_out = 0;     // bytes sent (framed bytes)

    // Allocate storage and deterministically generate the local A and b using generate_row().
    // Because generate_row uses the same seed+MatrixSpec as driver, the matrix is reproducible.
    void init_storage() {
        A.assign((size_t)rows * (size_t)n, 0.0);
        b.assign((size_t)rows, 0.0);
        pivot_tail.assign((size_t)n, 0.0);

        std::vector<double> row((size_t)n);
        for (int li = 0; li < rows; ++li) {
            int gi = row0 + li; // convert local row to global row index
            double bi = 0.0, sumabs = 0.0;
            // generate the global row 'gi' locally (same algorithm as driver)
            generate_row(n, seed, ms, gi, row.data(), bi, sumabs);
            std::copy(row.begin(), row.end(), &A[(size_t)li * (size_t)n]);
            b[(size_t)li] = bi;
        }
    }
};

// Helpers to send standard OK/ERR frames back to driver.
static void send_ok(SOCKET s) {
    net::send_msg(s, net::Msg::OK, {});
}
static void send_err(SOCKET s, const std::string& msg) {
    // Encode an error string as: uint32 len, bytes...
    net::ByteWriter w;
    uint32_t len = (uint32_t)msg.size();
    w.pod(len);
    w.bytes(msg.data(), msg.size());
    net::send_msg(s, net::Msg::ERR, w.buf);
}

// Core RPC dispatch for a session. Returns true to continue serving session, false to
// indicate session termination (SHUTDOWN returns false so main will exit process).
// All exceptions caught inside are turned into ERR replies (so driver receives consistent errors).
static bool handle_message(SOCKET s, WorkerState& st, net::Msg type, const std::vector<uint8_t>& payload) {
    using namespace std;

    try {
        // ----------------- INIT -----------------
        // Payload layout expected from INIT:
        //   int32 n, int32 rank, int32 p, int32 row0, int32 rows, uint64 seed,
        //   int32 mode, double alpha, double beta, double eps, int32 threads
        // This initializes WorkerState and calls init_storage().
        if (type == net::Msg::INIT) {
            net::ByteReader r(payload);
            st.n = r.pod<int32_t>();
            st.rank = r.pod<int32_t>();
            st.p = r.pod<int32_t>();
            st.row0 = r.pod<int32_t>();
            st.rows = r.pod<int32_t>();
            st.seed = r.pod<uint64_t>();
            int32_t mode_i = r.pod<int32_t>();
            st.ms.mode = (MatrixMode)mode_i;
            st.ms.alpha = r.pod<double>();
            st.ms.beta = r.pod<double>();
            st.ms.eps = r.pod<double>();
            st.threads = r.pod<int32_t>();

            // Apply thread hint if provided.
            if (st.threads > 0) omp_set_num_threads(st.threads);

            // Allocate A/b and generate rows deterministically.
            st.init_storage();
            send_ok(s);
            return true;
        }

        // If we haven't received INIT yet, all other messages are invalid.
        if (st.n <= 0) {
            send_err(s, "Worker not initialized (missing INIT).");
            return true;
        }

        // ----------------- PIVOT_SCAN -----------------
        // Request: int32 k
        // Reply:  double best_abs_value, int32 global_row_index_of_best (or -1)
        // Worker computes max |A[i,k]| over its owned rows i >= k and returns both magnitude and
        // the global row index. This is used by driver to select global pivot.
        if (type == net::Msg::PIVOT_SCAN) {
            net::ByteReader r(payload);
            int k = r.pod<int32_t>();

            double t0 = omp_get_wtime();
            double best = 0.0;
            int best_gi = -1;

            // Local scan range: rows whose global index gi satisfy gi >= k.
            int start_g = std::max(k, st.row0);
            int end_g = st.row0 + st.rows;
            if (start_g < end_g) {
                int li0 = start_g - st.row0;
                for (int li = li0; li < st.rows; ++li) {
                    int gi = st.row0 + li;
                    double v = std::abs(Aat(st.A, st.n, li, k));
                    if (v > best) { best = v; best_gi = gi; }
                }
            }

            double t1 = omp_get_wtime();
            st.t_scan += (t1 - t0);

            // Pack reply: best (double) then best_gi (int32)
            net::ByteWriter w;
            w.pod(best);
            w.pod((int32_t)best_gi);
            // Update bytes_out accounting: frame header + payload size
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::PIVOT_SCAN, w.buf);
            return true;
        }

        // ----------------- LOCAL_SWAP -----------------
        // Request: int32 gi1, int32 gi2
        // Swap two rows that must both be owned by this worker. Swap both A rows and b entries.
        if (type == net::Msg::LOCAL_SWAP) {
            net::ByteReader r(payload);
            int gi1 = r.pod<int32_t>();
            int gi2 = r.pod<int32_t>();
            int li1 = gi1 - st.row0;
            int li2 = gi2 - st.row0;
            if (li1 < 0 || li1 >= st.rows || li2 < 0 || li2 >= st.rows) {
                send_err(s, "LOCAL_SWAP rows not owned by this worker.");
                return true;
            }
            for (int j = 0; j < st.n; ++j) std::swap(Aat(st.A, st.n, li1, j), Aat(st.A, st.n, li2, j));
            std::swap(st.b[(size_t)li1], st.b[(size_t)li2]);
            send_ok(s);
            return true;
        }

        // ----------------- GET_ROW -----------------
        // Request: int32 gi
        // Reply:   int32 gi, double b_gi, double A[0..n-1] (n doubles)
        if (type == net::Msg::GET_ROW) {
            net::ByteReader r(payload);
            int gi = r.pod<int32_t>();
            int li = gi - st.row0;
            if (li < 0 || li >= st.rows) {
                send_err(s, "GET_ROW row not owned by this worker.");
                return true;
            }
            net::ByteWriter w;
            w.pod((int32_t)gi);
            w.pod(st.b[(size_t)li]);
            w.bytes(&st.A[(size_t)li * (size_t)st.n], (size_t)st.n * sizeof(double));
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::GET_ROW, w.buf);
            return true;
        }

        // ----------------- PUT_ROW -----------------
        // Request payload: int32 gi, double bi, n doubles (row).
        // Overwrite the local row gi's data accordingly.
        if (type == net::Msg::PUT_ROW) {
            net::ByteReader r(payload);
            int gi = r.pod<int32_t>();
            double bi = r.pod<double>();
            int li = gi - st.row0;
            if (li < 0 || li >= st.rows) {
                send_err(s, "PUT_ROW row not owned by this worker.");
                return true;
            }
            st.b[(size_t)li] = bi;
            r.bytes(&st.A[(size_t)li * (size_t)st.n], (size_t)st.n * sizeof(double));
            send_ok(s);
            return true;
        }

        // ----------------- GET_PIVOT_TAIL -----------------
        // Request: int32 k
        // Reply:   int32 k, double pivot (A[k,k]), int32 tail_len, tail_len doubles = A[k,k+1..n-1]
        // The pivot row must be owned by this worker.
        if (type == net::Msg::GET_PIVOT_TAIL) {
            net::ByteReader r(payload);
            int k = r.pod<int32_t>();
            int li = k - st.row0; // pivot row index relative to this worker
            if (li < 0 || li >= st.rows) {
                send_err(s, "GET_PIVOT_TAIL: pivot row not owned by this worker.");
                return true;
            }
            double pivot = Aat(st.A, st.n, li, k);
            int tail_len = st.n - (k + 1);

            net::ByteWriter w;
            w.pod((int32_t)k);
            w.pod(pivot);
            w.pod((int32_t)tail_len);
            if (tail_len > 0) {
                // send the tail entries (A[k,k+1..n-1])
                w.bytes(&st.A[(size_t)li * (size_t)st.n + (size_t)(k + 1)], (size_t)tail_len * sizeof(double));
            }
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::GET_PIVOT_TAIL, w.buf);
            return true;
        }

        // ----------------- BCAST_PIVOT_TAIL -----------------
        // Driver->worker broadcast with payload: int32 k, double pivot, int32 tail_len, tail_len doubles
        // On receipt the worker stores pivot and pivot_tail and updates timing for recv.
        if (type == net::Msg::BCAST_PIVOT_TAIL) {
            double t0 = omp_get_wtime();
            net::ByteReader r(payload);
            st.cur_k = r.pod<int32_t>();
            st.cur_pivot = r.pod<double>();
            int tail_len = r.pod<int32_t>();
            if (st.cur_k < 0 || st.cur_k >= st.n) {
                send_err(s, "BCAST_PIVOT_TAIL: bad k");
                return true;
            }
            if (tail_len != st.n - (st.cur_k + 1)) {
                send_err(s, "BCAST_PIVOT_TAIL: tail_len mismatch");
                return true;
            }
            if (tail_len > 0) {
                // copy received tail into pivot_tail vector at indices [k+1..n-1]
                r.bytes(&st.pivot_tail[(size_t)st.cur_k + 1], (size_t)tail_len * sizeof(double));
            }
            double t1 = omp_get_wtime();
            st.t_recv_pivot += (t1 - t0);
            send_ok(s);
            return true;
        }

        // ----------------- ELIMINATE -----------------
        // Request: int32 k
        // Worker expects that it previously received a BCAST_PIVOT_TAIL for the same k.
        // It computes for each local row i>k:
        //   aik = A[i,k] / pivot
        //   store aik in A[i,k] (L)
        //   for j=k+1..n-1: A[i,j] -= aik * pivot_tail[j]
        // The worker performs the per-row update in parallel using OpenMP.
        if (type == net::Msg::ELIMINATE) {
            net::ByteReader r(payload);
            int k = r.pod<int32_t>();
            if (k != st.cur_k) {
                send_err(s, "ELIMINATE: k does not match last BCAST_PIVOT_TAIL");
                return true;
            }
            double pivot = st.cur_pivot;
            if (pivot == 0.0) {
                send_err(s, "ELIMINATE: pivot is zero");
                return true;
            }

            double t0 = omp_get_wtime();
            // Only rows with global index > k participate in this elimination step.
            int start_g = std::max(k + 1, st.row0);
            int end_g = st.row0 + st.rows;
            if (start_g < end_g) {
                int li0 = start_g - st.row0;
#pragma omp parallel for schedule(static)
                for (int li = li0; li < st.rows; ++li) {
                    double aik = Aat(st.A, st.n, li, k) / pivot;
                    Aat(st.A, st.n, li, k) = aik; // store L multiplier in-place

                    double* rowp = &st.A[(size_t)li * (size_t)st.n];
#pragma omp simd
                    for (int j = k + 1; j < st.n; ++j) {
                        rowp[(size_t)j] -= aik * st.pivot_tail[(size_t)j];
                    }
                }
            }
            double t1 = omp_get_wtime();
            st.t_elim += (t1 - t0);
            send_ok(s);
            return true;
        }

        // ----------------- GET_BLOCK -----------------
        // Reply with: int32 row0, int32 rows, rows*n doubles (LU block), rows doubles (b block)
        if (type == net::Msg::GET_BLOCK) {
            net::ByteWriter w;
            w.pod((int32_t)st.row0);
            w.pod((int32_t)st.rows);
            w.bytes(st.A.data(), st.A.size() * sizeof(double));
            w.bytes(st.b.data(), st.b.size() * sizeof(double));
            st.bytes_out += sizeof(uint32_t) + 4 + w.buf.size();
            net::send_msg(s, net::Msg::GET_BLOCK, w.buf);
            return true;
        }

        // ----------------- GET_STATS -----------------
        // Reply with timing counters and bytes in/out for diagnostics.
        if (type == net::Msg::GET_STATS) {
            net::ByteWriter w;
            w.pod((double)st.t_scan);
            w.pod((double)st.t_recv_pivot);
            w.pod((double)st.t_elim);
            w.pod((uint64_t)st.bytes_in);
            w.pod((uint64_t)st.bytes_out);
            net::send_msg(s, net::Msg::GET_STATS, w.buf);
            return true;
        }

        // ----------------- SHUTDOWN -----------------
        // Reply OK and indicate to caller that the session should end and worker should exit.
        if (type == net::Msg::SHUTDOWN) {
            send_ok(s);
            return false; // signal to main loop to terminate process
        }

        // Unknown message type: reply with ERR (keeps session alive).
        send_err(s, "Unknown message type.");
        return true;
    }
    catch (const std::exception& e) {
        // Convert internal exception into ERR reply so the driver gets the error description.
        send_err(s, std::string("Exception: ") + e.what());
        return true;
    }
}

// Main worker entry: parse CLI, initialize Winsock, listen for driver connections and serve
// multiple driver sessions (unless a SHUTDOWN causes exit).
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    WorkerOptions wopt;
    std::string err;
    if (!parse_worker_args(argc, argv, wopt, err)) {
        std::cerr << err << "\n";
        return 2;
    }

    try {
        net::WSAInit wsa;

        std::cout << "lu_worker: listening on " << wopt.bind_ip << ":" << wopt.port << "\n";
        std::cout << "lu_worker: ready (will accept multiple driver sessions).\n";

        // Outer loop: accept new driver connections repeatedly. If driver disconnects
        // without SHUTDOWN (normal when --keep-workers is set), simply go back to accept().
        for (;;) {
            SOCKET s = INVALID_SOCKET;
            try {
                s = net::listen_and_accept(wopt.bind_ip, wopt.port);
                std::cout << "lu_worker: driver connected.\n";

                WorkerState st;
                if (wopt.threads > 0) st.threads = wopt.threads;

                bool shutdown_requested = false;

                // Session loop: process incoming framed messages until either SHUTDOWN or driver disconnect.
                for (;;) {
                    net::Msg type;
                    std::vector<uint8_t> payload;

                    try {
                        net::recv_msg(s, type, payload);
                    }
                    catch (const std::exception& e) {
                        // Normal end-of-session when driver closes connection in --keep-workers mode.
                        std::string msg = e.what();
                        if (msg.find("peer closed connection") != std::string::npos) {
                            std::cout << "lu_worker: driver disconnected (session ended).\n";
                            break; // back to accept()
                        }
                        throw; // real recv error -> propagate to outer catch
                    }

                    // Account for frame header + payload in bytes_in for diagnostic reporting.
                    st.bytes_in += sizeof(uint32_t) + 4 + payload.size();

                    bool cont = handle_message(s, st, type, payload);
                    if (!cont) {
                        // handle_message returns false only on SHUTDOWN (worker should exit).
                        shutdown_requested = true;
                        break;
                    }
                }

                net::closesocket_safe(s);
                s = INVALID_SOCKET;

                if (shutdown_requested) {
                    std::cout << "lu_worker: shutdown.\n";
                    return 0;
                }

                // Otherwise, loop back to accept a new connection.
            }
            catch (const std::exception& e) {
                if (s != INVALID_SOCKET) net::closesocket_safe(s);
                std::cerr << "lu_worker session error: " << e.what() << "\n";
                // Keep worker alive; go back to accept() to wait for next driver session.
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "lu_worker fatal: " << e.what() << "\n";
        return 1;
    }
}