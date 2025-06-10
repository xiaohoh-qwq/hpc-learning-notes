#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 1024

void print_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f", A[i][j]);
        }
        printf("\n");
    }
}
int main() {
    double **A = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
    }
    double *b = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX * 10.0;
        }
        b[i] = (double)rand() / RAND_MAX * 10.0;
    }

    printf("Starting GEPP for a %d x %d matrix...\n", N, N);
    double start_time = omp_get_wtime();

    for (int k = 0; k < N - 1; k++) {
        int max_row_idx = k; //第k行为主元，然后遍历k行下面的所有行
        for (int i = k + 1; i < N; i++) {
            if (fabs(A[i][k]) > fabs(A[max_row_idx][k])) {
                max_row_idx = i;
            }
        }

        if (max_row_idx != k) {
            // 交换矩阵A的两行
            double *temp_row = A[k];
            A[k] = A[max_row_idx];
            A[max_row_idx] = temp_row;

            // 交换向量b的对应元素
            double temp_b = b[k];
            b[k] = b[max_row_idx];
            b[max_row_idx] = temp_b;
        }

        #pragma omp parallel for
        for (int i = k + 1; i < N; i++) {
            double multiplier = A[i][k] / A[k][k];
            // 更新一整行
            for (int j = k; j < N; j++) {
                A[i][j] = A[i][j] - multiplier * A[k][j];
            }
            // 同时更新b向量
            b[i] = b[i] - multiplier * b[k];
        }
    }

    double end_time = omp_get_wtime();
    printf("Forward elimination phase finished in %f seconds.\n", end_time - start_time);
    //回代求解
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < N; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
    
    printf("Solution vector x (first 5 elements):\n");
    for(int i = 0; i < 5 && i < N; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(x);

    return 0;
}
