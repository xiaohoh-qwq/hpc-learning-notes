#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 

#define N 100000000

int main() {
    // 使用动态内存分配，以防栈溢出
    double *a = malloc(N * sizeof(double));
    if (a == NULL) {
        printf("内存分配失败!\n");
        return 1;
    }

    double sum_reduction = 0.0;
    double sum_manual = 0.0;
    double start_time, end_time;

    // 初始化数组，让每个元素为1.0，这样正确的总和就是 N
    for (int i = 0; i < N; i++) {
        a[i] = 1.0;
    }

    // --- 方法一：使用 OpenMP reduction 子句 ---
    printf("开始计算 (方法一：使用 reduction 子句)...\n");
    start_time = omp_get_wtime(); // 获取开始时间

    #pragma omp parallel for reduction(+:sum_reduction)
    for (int i = 0; i < N; i++) {
        sum_reduction += a[i];
    }

    end_time = omp_get_wtime(); // 获取结束时间
    printf("结果: sum_reduction = %f\n", sum_reduction);
    printf("耗时: %f 秒\n\n", end_time - start_time);


    // --- 方法二：手动实现归约 ---
    // 每个线程计算一个局部和，然后用原子操作更新全局和
    printf("开始计算 (方法二：手动实现归约)...\n");
    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        double local_sum = 0.0; // 每个线程都有自己的局部和变量 

        #pragma omp for
        for (int i = 0; i < N; i++) {
            local_sum += a[i];
        }

        // 使用原子操作将局部和安全地加到全局和上 
        #pragma omp atomic
        sum_manual += local_sum;
    }

    end_time = omp_get_wtime();
    printf("结果: sum_manual = %f\n", sum_manual);
    printf("耗时: %f 秒\n\n", end_time - start_time);

    // 释放内存
    free(a);

    return 0;
}
