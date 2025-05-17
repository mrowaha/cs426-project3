/**
 * amalgated header file the contains function definitions
 * shared across both serial and parallel implementations 
 * of Congruent Gradient method
 * 
 * @author Muhammad Rowaha
 */

#ifndef _MATRIX_OPS_H
#define _MATRIX_OPS_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

// A: matrix of size n_k x n (row-major)
// x: vector of size n
// y: output vector of size n_k (must be pre-allocated)
// n_k: number of rows of A
// n: number of columns of A (also the length of x)
void matrix_vector_multiply(const double *A, const double *x, double *y, int n_k, int n) {
    for (int i = 0; i < n_k; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}
/**
 * performs a simple inner dot product between two matrices
 * an example usage can be the following:
 * π = ⟨p, q⟩ or ρ = ⟨r, r⟩
 */
double dot(double *a, double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * updates the vector x in place
 * an  example usage can be the following:
 * x = x + α * p
 */
void add_scaled_vector_to(double *x, double alpha, double *p, int n) {
    for (int i = 0; i < n; i++) {
        x[i] += alpha * p[i];
    }
}

/**
 * updates the vector x in place
 * an  example usage can be the following:
 * x = x + α * p
 */
void subtract_scaled_vector_from(double *x, double alpha, double *q, int n) {
    for (int i = 0; i < n; i++) {
        x[i] -= alpha * q[i];
    }
}

double* concatenate_vectors(const double* a, int n, const double* b, int m) {
    double* result = malloc((n + m) * sizeof(double));
    if (!result) return NULL;
    memcpy(result, a, n * sizeof(double));
    memcpy(result + n, b, m * sizeof(double));
    return result;
}

void write_vector_to_file(double* vec, int size, char* filename) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < size; ++i) {
        fprintf(f, "%.6f\n", vec[i]);
    }

    fclose(f);
}

void write_matrix_to_file(double* matrix, int n, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(f, "%.6f", matrix[i * n + j]);
            if (j < n - 1) {
                fprintf(f, " ");
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

#endif // _MATRIX_OPS_H