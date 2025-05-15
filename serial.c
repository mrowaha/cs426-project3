#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include"matrix-ops.h"
#include"spd-loader.h"
#include"config.h"

int main() {
    clock_t start_time = clock();

    double *A = load_matrix(N);
    double *b = load_vector(N);
    double x[N] = {0.0};

    double r[N], p[N], q[N];
    double rho, rho_new, pi, alpha, beta;

    matrix_vector_multiply(A, x, r, N, N);  // here we get r = A*x
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - r[i]; // here we get r = b - A*x
        p[i] = r[i]; // and p = r
    }
    // r = b - A*x, p = r is what we get eventually

    rho = dot(r, r, N); //  ρ = ⟨r, r⟩

    for (int iter = 0; iter < MAX_ITER; iter++) {
        if (rho < TOL) {
            // printf("Converged at iteration %d with rho %lf\n", iter, rho);
            break;
        }

        matrix_vector_multiply(A, p, q, N, N);                  // q = A * p
        pi = dot(p, q, N);                   // π = ⟨p, q⟩
        alpha = rho / pi;                    // α = ρ / π
        add_scaled_vector_to(x, alpha, p, N);            // x = x + αp
        subtract_scaled_vector_from(r, alpha, q, N);        // r = r - αq

        rho_new = dot(r, r, N);              // ρ_new = ⟨r, r⟩
        beta = rho_new / rho;                // β = ρ_new / ρ
        rho = rho_new;

        for (int i = 0; i < N; i++) {        // p = r + βp
            p[i] = r[i] + beta * p[i];
        }
    }

    FILE *fp = fopen("serial_result.txt", "w");
    for (int i = 0; i < N; i++) {
        fprintf(fp, "%.6lf\n", x[i]);
    }
    fclose(fp);

    clock_t end_time = clock();    // End timing

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("serial elapsed time  %f s\n", elapsed_time);

    free(A);
    free(b);

    return 0;
}
