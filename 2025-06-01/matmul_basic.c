#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

void matmul_basic(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
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
    matmul_basic(A, B, C, N);
    clock_t end = clock();

    printf("Elapsed time: %f seconds\n", (double)(end - start)/CLOCKS_PER_SEC);

    // 简单打印C矩阵左上角4x4元素
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%8.2f ", C[i*N + j]);
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
