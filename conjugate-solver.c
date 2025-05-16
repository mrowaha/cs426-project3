/**
 * this is an MPI based conjugate solver implementation that makes use of the 
 * parallel matrix vector multiplication algorithm:
 * See https://github.com/mrowaha/parallel-matrix-vector-multiply
 * @author: Muhammad Rowaha
 * @email: ashfaqrowaha@gmail.com
 */

#include<mpi.h>
#include<stdlib.h> 
#include<math.h>
#include<stdio.h>
#include<string.h>

#include"config.h"
#include"spd-loader.h"
#include"matrix-ops.h"

#define MASTER 0

#include <stdlib.h>  // for malloc

#define malloc_n_by_p_vector(n, p) ((double *)malloc(((n) / (p)) * sizeof(double)))
#define malloc_n_by_sqrtp_vector(n, p) ((double *)malloc(((n) / ((int)sqrt(p))) * sizeof(double)))

double* expand_vector_within_columns(
    MPI_Comm comm,
    const int alpha,
    const int beta,
    double* p_vec,
    const int n,
    const int p,
    int* _pv_size  
);

void fold_z_within_rows(
    MPI_Comm comm,
    const int* coords,
    const int n,
    const int p,
    double** z,
    int* z_size
);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // perform validations on startup
    const int n = N;
    const int p = size - 1; // the -np param will be p + 1 because master does not participate in the computations
    const int sqrt_p = (int)sqrt(p); // p is perfect sqrt in this restriction
    const int log_sqrt_p = (int)(log2(sqrt_p));  // Number of folding steps
    if (sqrt_p * sqrt_p != p) {
        if (rank == MASTER)
            fprintf(stderr, "Number of workers must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (n % p != 0) {
        if (rank == MASTER)
            fprintf(stderr, "N mod workers should be 0");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm CONJUGATE_SOLVER_COMM = MPI_COMM_NULL;
    int color = (rank == MASTER) ? MPI_UNDEFINED : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &CONJUGATE_SOLVER_COMM);
    if (CONJUGATE_SOLVER_COMM != MPI_COMM_NULL) {
        // we are in the workers processing
        int worker_comm_rank, worker_comm_size;
        MPI_Comm_rank(CONJUGATE_SOLVER_COMM, &worker_comm_rank);
        MPI_Comm_size(CONJUGATE_SOLVER_COMM, &worker_comm_size);

        int coords[2]; // each process is identified by the (alpha, beta) coordinate in the 2D partitioning
        coords[0] = worker_comm_rank / sqrt_p;
        coords[1] = worker_comm_rank % sqrt_p;
        const int alpha = coords[0];
        const int beta = coords[1];

        const int n_by_sqrtp = n / sqrt_p;
        const int n_by_p = n / p;
        double *A_block = malloc(n_by_sqrtp * n_by_sqrtp * sizeof(double));
        MPI_Recv(A_block, n_by_sqrtp * n_by_sqrtp, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *b = malloc_n_by_p_vector(n, p);
        MPI_Recv(b, n_by_p, MPI_DOUBLE, MASTER, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *x = malloc_n_by_p_vector(n, p);
        memset(x, 0, n_by_p * sizeof(double));

        double *r_vec = malloc_n_by_p_vector(n, p);
        double *p_vec = malloc_n_by_p_vector(n, p);
        memcpy(r_vec, b, n_by_p * sizeof(double));
        memcpy(p_vec, b, n_by_p * sizeof(double));

        double _rho = dot(r_vec, r_vec, n_by_p);
        double rho = 0;
        MPI_Allreduce(&_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, CONJUGATE_SOLVER_COMM); // now we have ρ

        // expand p_vec within columns to form pv
        int pv_size;
        double *pv;

        for (int iter = 0; iter < MAX_ITER; iter++) {
            if (rho < TOL) {
                printf("%d: rho TOL reached\n", worker_comm_rank);
                break; // if the global rho has reached TOL, all processes break
            }

            pv = expand_vector_within_columns(CONJUGATE_SOLVER_COMM, alpha, beta, p_vec, n, p, &pv_size);
            if (pv_size != n_by_sqrtp) {
                fprintf(stderr, "conjugate solver rank %d: final pv size was expected to be n / sqrt(p) = %d but got %d\n", worker_comm_rank, n_by_sqrtp, pv_size);
                MPI_Abort(CONJUGATE_SOLVER_COMM, 1);
            }

            double *z = malloc_n_by_sqrtp_vector(n, p);
            int z_size = n_by_sqrtp;
            matrix_vector_multiply(A_block, pv, z, n_by_sqrtp, n_by_sqrtp);
            fold_z_within_rows(CONJUGATE_SOLVER_COMM, coords, n, p, &z, &z_size);
            if (z_size != n_by_p) {
                fprintf(stderr, "conjugate solver rank %d: final fold z size was expected to be n / p = %d but got %d\n", worker_comm_rank, n_by_p, z_size);
                MPI_Abort(CONJUGATE_SOLVER_COMM, 1);
            }
            // now z is basically y^{alpha, beta}

            // perform a transpose
            int transpose_partner = beta * sqrt_p + alpha;  // (β, α)
            double* y = malloc(z_size * sizeof(double)); // this is now y^{βα}
            if (worker_comm_rank != transpose_partner) {
                MPI_Sendrecv(
                    z, z_size, MPI_DOUBLE, transpose_partner, 0,
                    y, z_size, MPI_DOUBLE, transpose_partner, 0,
                    CONJUGATE_SOLVER_COMM, MPI_STATUS_IGNORE
                );
            } else {
                memcpy(y, z, z_size * sizeof(double));
            }
            free(z);
            
            double _gamma = dot(p_vec, y, n_by_p);
            double gamma = 0;
            MPI_Allreduce(&_gamma, &gamma, 1, MPI_DOUBLE, MPI_SUM, CONJUGATE_SOLVER_COMM);

            double _psi = dot(y, y, n_by_p);
            double psi = 0;
            MPI_Allreduce(&_psi, &psi, 1, MPI_DOUBLE, MPI_SUM, CONJUGATE_SOLVER_COMM);

            double a = rho / gamma;
            double B = a * ( psi / gamma ) - 1;
            rho = B * rho;
            
            add_scaled_vector_to(x, a, p_vec, n_by_p);
            subtract_scaled_vector_from(r_vec, a, y, n_by_p);
            for (int i = 0; i < n_by_p; i++) {
                p_vec[i] = r_vec[i] + B * p_vec[i];
            }
            free(y);
        }


        char file[100];
        snprintf(file, 100, "x_(%d,%d).txt", alpha, beta);
        write_vector_to_file(x, n_by_p, file);

        MPI_Barrier(CONJUGATE_SOLVER_COMM);
        free(A_block);
        free(b);
        free(r_vec);
        free(p_vec);
        free(pv);
        free(x);
    }

    if (rank == MASTER && CONJUGATE_SOLVER_COMM == MPI_COMM_NULL) {
        // if it is the master process
        double *A, *b;
        A = load_matrix(n);
        b = load_vector(n);

        const int n_by_sqrtp = n / sqrt_p;
        const int n_by_p = n / p;

        for (int alpha = 0; alpha < sqrt_p; ++alpha) {
            for (int beta = 0; beta < sqrt_p; ++beta) {
                int dest = 1 + alpha * sqrt_p + beta;
                double *A_block = malloc(n_by_sqrtp * n_by_sqrtp * sizeof(double));
                for (int i = 0; i < n_by_sqrtp; ++i) {
                    for (int j = 0; j < n_by_sqrtp; ++j) {
                        A_block[i * n_by_sqrtp + j] =
                            A[(alpha * n_by_sqrtp + i) * n + (beta * n_by_sqrtp + j)];
                    }
                }
                MPI_Send(A_block, n_by_sqrtp * n_by_sqrtp, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
                free(A_block);
            }
        }

        for (int beta = 0; beta < sqrt_p; ++beta) {
            for (int alpha = 0; alpha < sqrt_p; ++alpha) {
                double *offset = b + (beta * n_by_sqrtp) + (alpha * n_by_p);
                double *_b = malloc(n_by_p * sizeof(double));
                memcpy(_b, offset, n_by_p * sizeof(double));
                int dest = 1 + alpha * sqrt_p + beta;
                MPI_Send(_b, n_by_p, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
                free(_b);
            }
        }


        free(b);
        free(A);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}


double* expand_vector_within_columns(
    MPI_Comm comm,
    const int alpha,
    const int beta,
    double* p_vec,
    const int n,
    const int p,
    int* _pv_size  
) {

    // primitives
    const int n_by_p = n / p;
    const int sqrt_p = (int)sqrt(p); // p is perfect sqrt in this restriction
    const int log_sqrt_p = (int)(log2(sqrt_p));  // Number of folding steps

    int pv_size = n_by_p; // initially have this much size per process this will increase gradually because of concatenation from n/p to n/sqrt(p)
    double *pv = malloc_n_by_p_vector(n, p); // initally pv will start from size n / p and contain the p_vec
    memcpy(pv, p_vec, n_by_p * sizeof(double));
    for(int i = log_sqrt_p-1; i >= 0; i--) {
        int bit_mask = 1 << i;
        int partner_alpha = alpha ^ bit_mask;    // Flip i-th bit of β
        int partner_rank = partner_alpha * sqrt_p + beta;
        double* recv_buffer = malloc(pv_size * sizeof(double));
        MPI_Sendrecv(
            pv, pv_size, MPI_DOUBLE, partner_rank, 0,
            recv_buffer, pv_size, MPI_DOUBLE, partner_rank, 0,
            comm, MPI_STATUS_IGNORE
        );

        double* concatenated = NULL;
        if ((alpha & bit_mask) != 0) {
            concatenated = concatenate_vectors(recv_buffer, pv_size, pv, pv_size);
        } else {
            concatenated = concatenate_vectors(pv, pv_size, recv_buffer, pv_size);
        }
        free(pv);
        free(recv_buffer);
        pv = concatenated;
        pv_size *= 2;
    }
    *_pv_size = pv_size;
    return pv;
}

void fold_z_within_rows(
    MPI_Comm comm,
    const int* coords,
    const int n,
    const int p,
    double** z,
    int* z_size
) {

    // primitives
    const int n_by_p = n / p;
    const int sqrt_p = (int)sqrt(p); // p is perfect sqrt in this restriction
    const int log_sqrt_p = (int)(log2(sqrt_p));  // Number of folding steps
    const int n_by_sqrtp = n / sqrt_p;

    if (*(z_size) != n_by_sqrtp) {
        fprintf(stderr, "P(%d, %d): in the fold start phase, size of z must be equal to n / sqrt(p)\n", coords[0], coords[1]);
        MPI_Abort(comm, EXIT_FAILURE);
    }


    int fold_size = *z_size; // fold starts from n / sqrt(p) and goes all the way down to n / p
    int alpha = coords[0];
    int beta = coords[1];
    for (int i = 0; i < log_sqrt_p; i++) {
        int half_size = fold_size / 2;
        double* z1 = *z;                  // First half
        double* z2 = *z + half_size;      // Second half

        int bit_mask = 1 << i;
        int partner_beta = beta ^ bit_mask;    // Flip i-th bit of β
        int partner_rank = alpha * sqrt_p + partner_beta;

        MPI_Status status;
        double* recv_buf = malloc(half_size * sizeof(double));
        double* next_zblock = malloc(half_size * sizeof(double));
        if ((beta & bit_mask) != 0) {
            // i-th bit is 1: send z1, receive w2, update z2 := z2 + w2
            MPI_Send(z1, half_size, MPI_DOUBLE, partner_rank, 100 + i, comm);
            MPI_Recv(recv_buf, half_size, MPI_DOUBLE, partner_rank, 200 + i, comm, &status);
            for (int j = 0; j < half_size; ++j) {
                next_zblock[j] = z2[j] + recv_buf[j];
            }
        } else {
            // i-th bit is 0: receive w1, send z2, update z1 := z1 + w1
            MPI_Recv(recv_buf, half_size, MPI_DOUBLE, partner_rank, 100 + i, comm, &status);
            MPI_Send(z2, half_size, MPI_DOUBLE, partner_rank, 200 + i, comm);
            for (int j = 0; j < half_size; ++j) {
                next_zblock[j] = z1[j] + recv_buf[j];
            }
        }
        free(*z);
        free(recv_buf);
        *z = next_zblock;
        fold_size = half_size;
    }
    *z_size = fold_size;
}