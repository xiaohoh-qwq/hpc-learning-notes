#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

void matvec_basic(double *A, double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += A[i*n + j] * x[j];
        y[i] = sum;
    }
}

void matvec_unroll(double *A, double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        int j;
        for (j = 0; j <= n - 4; j += 4) {
            sum += A[i*n + j] * x[j] + A[i*n + j + 1] * x[j + 1]
                 + A[i*n + j + 2] * x[j + 2] + A[i*n + j + 3] * x[j + 3];
        }
        for (; j < n; j++) {
            sum += A[i*n + j] * x[j];
        }
        y[i] = sum;
    }
}

void init_matrix(double *M, int n) {
    for (int i = 0; i < n*n; i++)
        M[i] = (double)(rand() % 10);
}

void init_vector(double *v, int n) {
    for (int i = 0; i < n; i++)
        v[i] = (double)(rand() % 10);
}

int main() {
    double *A = malloc(sizeof(double)*N*N);
    double *x = malloc(sizeof(double)*N);
    double *y = malloc(sizeof(double)*N);
    double *y2 = malloc(sizeof(double)*N);

    srand(time(NULL));
    init_matrix(A, N);
    init_vector(x, N);

    clock_t start = clock();
    matvec_basic(A, x, y, N);
    clock_t end = clock();
    printf("Basic matvec time: %f seconds\n", (double)(end - start)/CLOCKS_PER_SEC);

    start = clock();
    matvec_unroll(A, x, y2, N);
    end = clock();
    printf("Unrolled matvec time: %f seconds\n", (double)(end - start)/CLOCKS_PER_SEC);

    // 验证结果是否相同
    for (int i = 0; i < N; i++) {
        if (abs(y[i] - y2[i]) > 1e-6) {
            printf("Mismatch at %d\n", i);
            break;
        }
    }

    free(A); free(x); free(y); free(y2);
    return 0;
}
