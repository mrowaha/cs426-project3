#include<mpi.h>
#include<stdlib.h>
#include<stdio.h>

#include"spd-loader.h"
#include"matrix-ops.h"
#include"config.h"

#define MASTER 0


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A, *b;
    if (rank == MASTER) {
        A = load_matrix(N);
        b = load_vector(N);
    }

    /**
     * the rows of A, p and q (output) vector will be equally divided among
     * size - 1 processes (workers)
     * it has been assumed that N % (size - 1) = 0
     */
    int worker_count = size - 1;
    if (N % worker_count != 0) {
        if (rank == MASTER)
            fprintf(stderr, "N mod workers should be 0");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n_k = N / worker_count; // the portion of the matrices and in/out vectors that each vector gets
    double *A_k, *b_k;

    if (rank != MASTER) {
        A_k = (double*)malloc(sizeof(double) * (N * n_k));
        b_k = (double*)malloc(sizeof(double) * n_k);
    }

    if (rank == MASTER) {
        for (int dest = 1; dest < size; ++dest) {
            int offset = (dest - 1) * n_k;
            // Send the rows of A (each row is of size N)
            MPI_Send(&A[offset * N], n_k * N, MPI_DOUBLE, dest, MASTER, MPI_COMM_WORLD);
            // Send the corresponding chunk of b
            MPI_Send(&b[offset], n_k, MPI_DOUBLE, dest, MASTER, MPI_COMM_WORLD);
        }
    } else {
         // Receive chunk of A
        MPI_Recv(A_k, n_k * N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Receive chunk of b
        MPI_Recv(b_k, n_k, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Comm WORKER_COMM;
    int color = (rank == MASTER) ? MPI_UNDEFINED : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &WORKER_COMM);

    double x[N] = {0.0}; // initial vector x
    if (WORKER_COMM != MPI_COMM_NULL) {
        int worker_comm_rank, worker_comm_size;
        MPI_Comm_rank(WORKER_COMM, &worker_comm_rank);
        MPI_Comm_size(WORKER_COMM, &worker_comm_size);

        double rho = 0.0, pi = 0.0, kappa = 0.0;
        double p[N] = {0.0}; // augmented p_k vector

        double *r_k, *p_k, *q_k, *x_k;
        r_k = (double*)malloc(sizeof(double) * n_k);
        p_k = (double*)malloc(sizeof(double) * n_k);
        q_k = (double*)malloc(sizeof(double) * n_k);
        x_k = (double*)malloc(sizeof(double) * n_k);


        matrix_vector_multiply(A_k, x, r_k, n_k, N); // r_k = A_k*x
        for (int i = 0; i < n_k; i++) {
            r_k[i] = b_k[i] - r_k[i]; // r_k = b_k - A_k*x
            p_k[i] = r_k[i]; // p_k = r_k
            x_k[i] = 0.0;
        }
        /**
         * so at this point we now have r_k = p_k = b_k - A_k*x
         */

        double rho_k = dot(r_k, r_k, n_k); // ρ_k = ⟨r_k, r_k⟩
        MPI_Allreduce(&rho_k, &rho, 1, MPI_DOUBLE, MPI_SUM, WORKER_COMM); // now we have ρ
        int *send_set = (int*)malloc(sizeof(int) * worker_comm_size-1);
        int *recv_set = send_set;
        // since we are dealing with SPD matrices, send set and recv set are equal
        int i = 0;
        for (int sibling_rank = 0; sibling_rank < worker_comm_size; sibling_rank++) {
            if (sibling_rank == worker_comm_rank) continue;
            send_set[i] = sibling_rank;
            i++;
        }

        MPI_Request *send_reqs = malloc(sizeof(MPI_Request) * (worker_comm_size - 1));
        MPI_Request *recv_reqs = malloc(sizeof(MPI_Request) * (worker_comm_size - 1));
        for (int iter = 0; iter < MAX_ITER; iter++) {
            if (rho < TOL) {
                // printf("worker %d converged at iteration %d with rho %lf\n", worker_comm_rank, iter, rho);
                break; // if the global rho has reached TOL, all processes break
            }

            // update its own augmented p vector entries
            int i = 0;
            for (int start = n_k * worker_comm_rank; start < n_k * (worker_comm_rank+1); start++) {
                p[start] = p_k[i];
                i++; 
            }

            /**
             * Post all non-blocking receives
             */
            for (int i = 0; i < worker_comm_size-1; i++) {
                int src = recv_set[i];
                MPI_Irecv(&p[n_k * src], n_k, MPI_DOUBLE, src, 0, WORKER_COMM, &recv_reqs[i]);
            }
            /**
             * Post all non-blocking sends
             */
            for (int i = 0; i < worker_comm_size-1; i++) {
                int dest = send_set[i];
                MPI_Isend(p_k, n_k, MPI_DOUBLE, dest, 0, WORKER_COMM, &send_reqs[i]);
            }
            /**
             * Wait for all sends and receives to complete
             */
            MPI_Waitall(worker_comm_size - 1, send_reqs, MPI_STATUSES_IGNORE);
            MPI_Waitall(worker_comm_size - 1, recv_reqs, MPI_STATUSES_IGNORE);
            MPI_Barrier(WORKER_COMM);

            matrix_vector_multiply(A_k, p, q_k, n_k, N); // q_k = A_k * p (the augmented p_k vector) 
            double pi_k = dot(p_k, q_k, n_k);  // pi_k = p_k dot q_k
            double kappa_k = dot(q_k, q_k, n_k); // kappa_j = q_k dot q_k
            MPI_Allreduce(&pi_k, &pi, 1, MPI_DOUBLE, MPI_SUM, WORKER_COMM); // now we have pi
            MPI_Allreduce(&kappa_k, &kappa, 1, MPI_DOUBLE, MPI_SUM, WORKER_COMM); // now we have kappa

            double alpha = rho / pi; // α = ρ / π
            double beta = alpha * (kappa / pi) - 1; // β =  α * (k / π) - 1
            rho = beta *  rho; // ρ = β * ρ
            
            add_scaled_vector_to(x_k, alpha, p_k, n_k); // x_k = x_k + a * p_k
            subtract_scaled_vector_from(r_k, alpha, q_k, n_k); // r_k = r_k - a * q_k
            
            for (int i = 0; i < n_k; i++) {
                p_k[i] = r_k[i] + beta * p_k[i];
            }
            // and we get p_k = r_k + (p_k + B * p_k)
        }
        MPI_Send(x_k, n_k, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);

        free(send_reqs);
        free(recv_reqs);
        free(r_k);
        free(p_k);
        free(send_set);
        MPI_Comm_free(&WORKER_COMM);
    }


    if (rank == MASTER) {
        for (int i = 1; i < size; i++) {
            MPI_Recv(&x[(i-1)*n_k], n_k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        FILE *fp = fopen("parallel_result.txt", "w");
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6lf\n", x[i]);
        }
        fclose(fp);
        free(A);
        free(b);
    } else {
        free (A_k);
        free (b_k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (rank == MASTER){
        printf("%d workers = elapsed time %f s\n", worker_count, end_time - start_time);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}