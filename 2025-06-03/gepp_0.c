#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h> 

// 函数：打印矩阵
void print_matrix(double **A, int n) {
    printf("Matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.3f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 函数：初始化一个测试矩阵 
void init_matrix(double **A, int n) {
    // 创建一个可预测的矩阵，方便验证
    // 例如：
    // A = [ 2  1  1 ]
    //     [ 4 -6  0 ]
    //     [-2  7  2 ]
    if (n == 3) {
        A[0][0] = 2; A[0][1] = 1;  A[0][2] = 1;
        A[1][0] = 4; A[1][1] = -6; A[1][2] = 0;
        A[2][0] = -2;A[2][1] = 7;  A[2][2] = 2;
    } else { // 对于其他大小，随机初始化或简单填充
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = (double)(rand() % 100) + 1.0; // 1 到 100 的随机数
            }
        }
        // 确保对角线元素不为0，以增加非奇异性（尽管部分主元会处理）
        for (int i = 0; i < n; i++) {
            if (A[i][i] == 0.0) A[i][i] = 1.0;
        }
    }
}

// 函数：执行带部分主元的高斯消元法
void gaussian_elimination_partial_pivoting(double **A, int n) {
    for (int i = 0; i < n - 1; i++) {
        // 1. 查找主元
        int pivot_row = i;
        double max_val = fabs(A[i][i]);
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k][i]) > max_val) {
                max_val = fabs(A[k][i]);
                pivot_row = k;
            }
        }

        // 2. 奇异性检查 
        if (max_val < 1e-9) { // 用一个很小的值判断是否接近0
            printf("Warning: Matrix might be singular or nearly singular at step %d.\n", i);
            // 在实际应用中可能需要更复杂的处理或退出
        }

        // 3. 行交换
        if (pivot_row != i) {
            for (int k = 0; k < n; k++) { // 从第0列开始交换整行
                double temp = A[i][k];
                A[i][k] = A[pivot_row][k];
                A[pivot_row][k] = temp;
            }
        }

        // 4. 消元 (计算乘数并更新行)
        // 我们将其分解为更传统的循环形式
        for (int j = i + 1; j < n; j++) { // 对于主元下方的每一行 j
            if (A[i][i] == 0.0) { // 再次检查，尽管部分主元应避免此情况
                 printf("Error: Division by zero at A[%d][%d]. Should have been handled by pivoting.\n", i, i);
                 return;
            }
            double factor = A[j][i] / A[i][i];
            A[j][i] = factor; 

            // 更新行 j 的剩余元素
            // 这是我们要进行循环展开的目标循环
            for (int k = i + 1; k < n; k++) {
                A[j][k] = A[j][k] - factor * A[i][k];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int n = 5; // 默认矩阵大小
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            printf("Matrix size must be positive.\n");
            return 1;
        }
    }
    printf("Matrix size n = %d\n", n);

    // 动态分配矩阵内存
    double **A = (double **)malloc(n * sizeof(double *));
    if (A == NULL) {
        perror("Failed to allocate memory for matrix rows");
        return 1;
    }
    for (int i = 0; i < n; i++) {
        A[i] = (double *)malloc(n * sizeof(double));
        if (A[i] == NULL) {
            perror("Failed to allocate memory for matrix columns");
            // 释放已分配的内存
            for (int j = 0; j < i; j++) free(A[j]);
            free(A);
            return 1;
        }
    }

    srand(time(NULL)); // 初始化随机数种子
    init_matrix(A, n);

    printf("Original Matrix (gepp_0.c):\n");
    print_matrix(A, n);

    // 记录开始时间
     clock_t start_time = clock();

    gaussian_elimination_partial_pivoting(A, n);

    // 记录结束时间 
     clock_t end_time = clock();
     double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Matrix after GEPP (gepp_0.c):\n");
    print_matrix(A, n);
     printf("CPU time used: %f seconds\n", cpu_time_used);

    // 释放内存
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);

    return 0;
}
