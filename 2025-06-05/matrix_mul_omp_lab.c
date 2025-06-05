#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // OpenMP 头文件，也包含 omp_get_wtime()
#include <math.h>  // For fabs in verification

// 辅助函数：分配矩阵内存
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        perror("内存分配失败 (行)");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            perror("内存分配失败 (列)");
            for(int k=0; k<i; ++k) free(matrix[k]);
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

// 辅助函数：释放矩阵内存
void free_matrix(double** matrix, int rows) {
    if (matrix == NULL) return;
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 辅助函数：初始化矩阵（简单填充）
void initialize_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)(i + j + 1.0);
        }
    }
}

// 辅助函数：打印矩阵（用于小矩阵验证）
void print_matrix(const char* name, double** matrix, int rows, int cols) {
    printf("矩阵 %s (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i][j]);
            if (cols > 10 && j >= 9) {
                printf("...");
                break;
            }
        }
        printf("\n");
        if (rows > 10 && i >= 9) {
            printf("...\n");
            break;
        }
    }
    printf("\n");
}

// OpenMP 并行矩阵乘法 C = A * B
void matrix_multiply_parallel(double** A, double** B, double** C, int N_A_rows, int N_A_cols_B_rows, int N_B_cols) {
    int j, k; 
    #pragma omp parallel for private(j, k) shared(A, B, C, N_A_rows, N_A_cols_B_rows, N_B_cols)
    for (int i = 0; i < N_A_rows; i++) {
        for (j = 0; j < N_B_cols; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < N_A_cols_B_rows; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


int main(int argc, char *argv[]) {
    int N = 1000; // 默认矩阵维度 (N x N) * (N x N)
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("矩阵维度必须为正数。\n");
            return 1;
        }
    }
    printf("矩阵维度: %d x %d\n", N, N);

    int r1 = N, c1_r2 = N, c2 = N;

    double** A = allocate_matrix(r1, c1_r2);
    double** B = allocate_matrix(c1_r2, c2);
    double** C_parallel = allocate_matrix(r1, c2);

    initialize_matrix(A, r1, c1_r2);
    initialize_matrix(B, c1_r2, c2);

    if (N <= 10) {
        print_matrix("A", A, r1, c1_r2);
        print_matrix("B", B, c1_r2, c2);
    }

    // --- OpenMP 并行版本测试 ---
    printf("OpenMP 最大可用线程数: %d\n", omp_get_max_threads());

    double start_time_parallel = omp_get_wtime();
    matrix_multiply_parallel(A, B, C_parallel, r1, c1_r2, c2);
    double end_time_parallel = omp_get_wtime();
    double time_parallel = end_time_parallel - start_time_parallel;
    printf("OpenMP 并行矩阵乘法时间: %f 秒\n", time_parallel);

    if (N <= 10) {
        print_matrix("C (并行)", C_parallel, r1, c2);
    }

    free_matrix(A, r1);
    free_matrix(B, c1_r2);
    free_matrix(C_parallel, r1);

    return 0;
}
