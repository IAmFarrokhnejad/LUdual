#pragma once
// Authors: Morteza Farrokhnejad, Ali Farrokhnejad
// lu_common.h
// Shared utilities for MPI-lite LU project (matrix generation, distribution, solve, verification).
// This file is derived from your lu_final.cpp logic to keep identical A,b generation and verification.

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

enum class MatrixMode {
    DiagDominant,
    WeakDiagDominant,
    Random,
    NearSingular
};

struct MatrixSpec {
    MatrixMode mode = MatrixMode::DiagDominant;
    double alpha = 0.0; // diag boost via alpha*sumabs
    double beta  = 0.0; // diag shift via beta*N
    double eps   = 1e-3; // near_singular noise
};

const char* mode_name(MatrixMode m);
bool parse_mode(const std::string& s, MatrixMode& out);

// ---------------------- Deterministic RNG helpers ----------------------

uint64_t splitmix64(uint64_t& x);
double u01_from_u64(uint64_t u);
double val_ij(uint64_t seed, int i, int j);

// ---------------------- Matrix generator ----------------------

double base_Aij(uint64_t seed, const MatrixSpec& ms, int i, int j);
double diag_adjust(int n, const MatrixSpec& ms, double row_sum_abs);

void generate_row(
    int n,
    uint64_t seed,
    const MatrixSpec& ms,
    int i,
    double* row_out,   // length n
    double& b_out,
    double& row_sum_abs_out
);

void compute_residual_inf_norms(
    int n,
    uint64_t seed,
    const MatrixSpec& ms,
    const std::vector<double>& x,
    double& resid_inf_out,
    double& b_inf_out,
    double& rel_resid_inf_out
);

// ---------------------- Distribution helpers ----------------------

struct Dist {
    int n = 0;
    int rank = 0;
    int size = 1;
    int row0 = 0;
    int rows = 0;
    std::vector<int> counts;
    std::vector<int> displs;

    Dist() = default;
    Dist(int n_, int rank_, int size_);

    int owner_of_row(int global_i) const;
    int local_index(int global_i) const { return global_i - row0; }
};

// ---------------------- Solve on driver (root) ----------------------

void solve_on_root_inplace_LU(
    int n,
    const std::vector<double>& LU,
    const std::vector<double>& b,
    std::vector<double>& x_out
);

// ---------------------- Output helpers ----------------------

double checksum_weighted_first100(const std::vector<double>& x);

struct RunOptions {
    // Driver options
    int n = 0;
    uint64_t seed = 123456789ULL;

    MatrixMode mode = MatrixMode::DiagDominant;

    // Diagonal adjustment parameters (same semantics as lu_final.cpp):
    //   A_ii += alpha * sum_j |A_ij| + beta * N
    bool alpha_set = false;
    bool beta_set = false;
    double alpha = 0.0;
    double beta = 0.0;

    // Near-singular noise:
    bool eps_set = false;
    double eps = 1e-3;

    bool timing = false;
    bool verify = false;

    int threads = 0; // if 0, don't set OMP from driver (workers can use INIT threads)
    std::string csv_path;

    std::string hosts_path; // required for driver

	int keep_workers = 0; // for driver: if non-zero, don't shutdown workers on completions
};

bool parse_driver_args(int argc, char** argv, RunOptions& opt, std::string& err);

// worker CLI
struct WorkerOptions {
    std::string bind_ip = "0.0.0.0";
    uint16_t port = 5000;
    int threads = 0; // optional default
};

bool parse_worker_args(int argc, char** argv, WorkerOptions& opt, std::string& err);
