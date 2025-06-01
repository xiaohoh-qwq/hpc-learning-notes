#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

void matmul_unroll(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            int k;
            for (k = 0; k <= n - 4; k += 4) {
                sum += A[i*n + k] * B[k*n + j]
                     + A[i*n + k + 1] * B[(k + 1)*n + j]
                     + A[i*n + k + 2] * B[(k + 2)*n + j]
                     + A[i*n + k + 3] * B[(k + 3)*n + j];
            }
            for (; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
}

void init_matrix(double *M, int n) {
    for (int i = 0; i < n*n; i++)
        M[i] = (double)(rand() % 10);
}

int main() {
    double *A = malloc(sizeof(double)*N*N);
    double *B = malloc(sizeof(double)*N*N);
    double *C = malloc(sizeof(double)*N*N);

    srand(time(NULL));
    init_matrix(A, N);
    init_matrix(B, N);

    clock_t start = clock();
    matmul_unroll(A, B, C, N);
    clock_t end = clock();

    printf("Unrolled matmul time: %f seconds\n", (double)(end - start)/CLOCKS_PER_SEC);

    // 简单打印部分结果
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%8.2f ", C[i*N + j]);
        printf("\n");
    }

    free(A); free(B); free(C);
    return 0;
}
