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

typedef enum {
    TAG_send_x_result = 5,
    TAG_send_expanded_result = 6
} TAGS;

int block_index(int alpha, int beta, int sqrt_p) {
    return beta * sqrt_p + alpha;
}


double* expand_vector_within_columns(
    MPI_Comm comm,
    const int alpha,
    const int beta,
    double* p_vec,
    const int n,
    const int p,
    int* _pv_size  
);

static void fold_z_within_rows(MPI_Comm row_comm,
                                double *z_local,
                                int     n_by_sqrtp,
                                int     n_by_p,
                                int    *z_size
                            )
{
    int rank, size;
    MPI_Comm_rank(row_comm, &rank);
    MPI_Comm_size(row_comm, &size);

    double *z_sum = (double *)calloc(n_by_sqrtp, sizeof(double));
    memcpy(z_sum, z_local, n_by_sqrtp * sizeof(double));

    double *recv_buf = (double *)malloc(n_by_sqrtp * sizeof(double));

    for (int i = 1; i < size; ++i) {
        int src = (rank + i) % size;
        int dst = (rank - i + size) % size;

        MPI_Sendrecv(z_local, n_by_sqrtp, MPI_DOUBLE, dst, 0,
                     recv_buf, n_by_sqrtp, MPI_DOUBLE, src, 0,
                     row_comm, MPI_STATUS_IGNORE);

        for (int j = 0; j < n_by_sqrtp; ++j) {
            z_sum[j] += recv_buf[j];
        }
    }

    int offset = rank * n_by_p;
    memcpy(z_local, z_sum + offset, n_by_p * sizeof(double));

    free(z_sum);
    free(recv_buf);
    *z_size = n_by_p;
}


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
        const int transpose_partner = beta * sqrt_p + alpha;  // (β, α)

        MPI_Comm row_comm, col_comm;

        MPI_Comm_split(CONJUGATE_SOLVER_COMM, /*color =*/coords[0], coords[1], &row_comm);
        MPI_Comm_split(CONJUGATE_SOLVER_COMM, /*color =*/coords[1], coords[0], &col_comm);

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
        double rho = 0, pi = 0, kappa = 0;
        MPI_Allreduce(&_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, CONJUGATE_SOLVER_COMM); // now we have ρ

        int pv_size;
        double *pv;
        for (int iter = 0; iter < MAX_ITER; iter++) {
            if (rho < TOL) {
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
            // fold_z_within_rows(row_comm, coords, n, p, &z, &z_size);
            fold_z_within_rows(row_comm, z, n_by_sqrtp, n_by_p, &z_size);
            // if (z_size != n_by_p) {
            //     fprintf(stderr, "conjugate solver rank %d: final fold z size was expected to be n / p = %d but got %d\n", worker_comm_rank, n_by_p, z_size);
            //     MPI_Abort(CONJUGATE_SOLVER_COMM, 1);
            // }
            // now z is basically y^{alpha, beta}


            // perform a transpose
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

            double pi_k = dot(p_vec, y, n_by_p);
            MPI_Allreduce(&pi_k, &pi, 1, MPI_DOUBLE, MPI_SUM, CONJUGATE_SOLVER_COMM);
            double a = rho / pi;
            add_scaled_vector_to(x, a, p_vec, n_by_p);
            subtract_scaled_vector_from(r_vec, a, y, n_by_p);
            double _rho_new = dot(r_vec, r_vec, n_by_p);
            double rho_new;
            MPI_Allreduce(&_rho_new, &rho_new, 1, MPI_DOUBLE, MPI_SUM,
                        CONJUGATE_SOLVER_COMM);                  /* ρₖ₊₁ = rᵀ r        */

            double b = rho_new / rho;                           /* βₖ   = ρₖ₊₁ / ρₖ   */
            rho   = rho_new;                                       /* prepare for next k */

            for (int i = 0; i < n_by_p; ++i)
                p_vec[i] = r_vec[i] + b * p_vec[i];             /* pₖ₊₁ = rₖ₊₁ + β pₖ */

            free(y);
        }


        int x_size = n_by_p;
        double *r = expand_vector_within_columns(CONJUGATE_SOLVER_COMM, alpha, beta, x, n, p, &x_size);
        if (x_size != n_by_sqrtp) {
            fprintf(stderr, "failed to properly expand x across columns\n");
            MPI_Abort(CONJUGATE_SOLVER_COMM, EXIT_FAILURE);
        }
        if (alpha == 0) {
            MPI_Send(r, x_size, MPI_DOUBLE, MASTER, TAG_send_expanded_result, MPI_COMM_WORLD);
        }
        free(r);

        free(A_block);
        free(b);
        free(r_vec);
        free(p_vec);
        free(pv);
        free(x);
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
        MPI_Comm_free(&CONJUGATE_SOLVER_COMM);
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
                for (int i = 0; i < n_by_sqrtp; ++i)
                    for (int j = 0; j < n_by_sqrtp; ++j)
                        A_block[i * n_by_sqrtp + j] = A[(alpha * n_by_sqrtp + i) * n + (beta * n_by_sqrtp + j)];

                MPI_Send(A_block, n_by_sqrtp * n_by_sqrtp, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
                free(A_block);
            }
        }

        for (int beta = 0; beta < sqrt_p; ++beta) {
            for (int alpha = 0; alpha < sqrt_p; ++alpha) {
                int dest = 1 + alpha * sqrt_p + beta;
                int block_idx = block_index(alpha, beta, sqrt_p);
                double *offset = b + block_idx * n_by_p;

                double *_b = malloc(n_by_p * sizeof(double));
                memcpy(_b, offset, n_by_p * sizeof(double));
                MPI_Send(_b, n_by_p, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
                free(_b);
            }
        }

        // result accumulation
        double* x = malloc(n * sizeof(double));
        memset(x, 0, n * sizeof(double));
        for (int beta = 0; beta < sqrt_p; ++beta) {
            int src = 1 + beta; // since alpha = 0
            double *offset = x + beta * n_by_sqrtp;
            MPI_Recv(offset, n_by_sqrtp, MPI_DOUBLE, src, TAG_send_expanded_result, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received x_{beta} from (0,%d) rank=%d, into offset %d\n", beta, src, beta * n_by_sqrtp);
        }

        char out[100];
        snprintf(out, 100, "result_%d.txt", size);
        write_vector_to_file(x, n, out);
        free(b);
        free(A);
        free(x);
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