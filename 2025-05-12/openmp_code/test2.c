#include <stdio.h>
#include <omp.h>
#include <time.h>

int main() {
    long long sum = 0; // 使用 long long 类型
    int N = 1000000;

    // 获取当前时间（开始时间）
    double start_time = omp_get_wtime();

    // 使用 OpenMP 加速
    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= N; i++) {
        sum += i;
    }

    // 获取当前时间（结束时间）
    double end_time = omp_get_wtime();

    // 打印结果和执行时间
    printf("Sum = %lld\n", sum);  // 输出 long long 类型的值
    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}
