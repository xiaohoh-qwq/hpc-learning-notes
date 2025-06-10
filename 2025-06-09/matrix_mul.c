#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// 定义矩阵维度
#define N 1000 // 矩阵A的行数，结果C的维度
#define M 800  // 矩阵A的列数

int main() {
    // 分配内存
    double* A = (double*)malloc(N * M * sizeof(double));
    double* C = (double*)malloc(N * N * sizeof(double));

    // 初始化矩阵A
    srand(time(NULL));
    for (int i = 0; i < N * M; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }

    printf("Starting parallel matrix multiplication C = A * A^T...\n");
    double start_time = omp_get_wtime();
    //循环的迭代次数随 i 的增加而减少
    // 使用 schedule(dynamic) 来处理不均衡的工作负载
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        // 只计算上三角部分 (j 从 i 开始)，因为C是对称矩阵
        for (int j = i; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * A[j * M + k];
            }
            C[i * N + j] = sum;
            C[j * N + i] = sum; // 利用对称性直接填充下三角部分
        }
    }

    double end_time = omp_get_wtime();
    printf("Calculation finished in %f seconds.\n", end_time - start_time);

    // 释放内存
    free(A);
    free(C);

    return 0;
}
