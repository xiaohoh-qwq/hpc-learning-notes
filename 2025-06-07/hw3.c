#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>

// 定义数组大小
#define N 20000000

// 串行扫描函数，用于基准测试和局部扫描
void sequential_scan(double* in, double* out, int len) {
    if (len > 0) {
        out[0] = in[0];
        for (int i = 1; i < len; i++) {
            out[i] = out[i - 1] + in[i];
        }
    }
}

int main() {
    // --- 1. 初始化 ---
    double *a, *c_seq, *c_par;
    double start_time, end_time;

    // 分配内存
    a = (double*)malloc(N * sizeof(double));
    c_seq = (double*)malloc(N * sizeof(double)); // 存放串行结果
    c_par = (double*)malloc(N * sizeof(double)); // 存放并行结果

    // 初始化输入数组 a
    for (int i = 0; i < N; i++) {
        a[i] = 1.0;
    }

    printf("数组大小 N = %d\n\n", N);

    // --- 2. 串行算法作为基准 ---
    printf("开始计算 (串行算法)...\n");
    start_time = omp_get_wtime();
    sequential_scan(a, c_seq, N);
    end_time = omp_get_wtime();
    printf("串行算法耗时: %f 秒\n\n", end_time - start_time);


    // --- 3. 并行扫描算法 ---
    printf("开始计算 (并行扫描算法)...\n");
    start_time = omp_get_wtime();

    double *block_sums; // 这就是讲座中的辅助数组 W

    // 在一个大的并行区域内完成所有操作，以避免重复创建和销毁线程
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // 在主线程(tid=0)中分配辅助数组
        #pragma omp single
        {
            block_sums = (double*)malloc(num_threads * sizeof(double));
        }

        // 计算每个线程负责的数据块范围
        int chunk_size = N / num_threads;
        int start = tid * chunk_size;
        int end = (tid == num_threads - 1) ? N : start + chunk_size;

        // --- 阶段一: 局部扫描 ---
        // 每个线程对自己的块进行扫描，结果存入 c_par
        sequential_scan(a + start, c_par + start, end - start);

        // 将自己块的最后一个元素(块内总和)存入辅助数组
        block_sums[tid] = c_par[end - 1];

        // **同步点1**: 确保所有线程都完成了局部扫描，并且 block_sums 已被填满
        #pragma omp barrier

        // --- 阶段二: 中间扫描 ---
        // **由单个线程**对 block_sums 数组自身进行扫描
        #pragma omp single
        {
            for (int i = 1; i < num_threads; i++) {
                block_sums[i] += block_sums[i - 1];
            }
        }
        // `single` 区域结束时有一个隐式的屏障，作为 **同步点2**

        // --- 阶段三: 全局更新 ---
        // 每个线程(除了第一个)将前面所有块的总和加到自己的整个块上
        if (tid > 0) {
            for (int i = start; i < end; i++) {
                c_par[i] += block_sums[tid - 1];
            }
        }
    } // 并行区域结束

    end_time = omp_get_wtime();
    printf("并行算法耗时: %f 秒\n\n", end_time - start_time);


    // --- 4. 验证结果 ---
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (c_seq[i] != c_par[i]) {
            printf("验证失败! 索引 %d 处的值不匹配: 串行=%.2f, 并行=%.2f\n", i, c_seq[i], c_par[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("验证成功! 并行算法结果与串行算法结果完全一致。\n");
    }

    // --- 5. 清理内存 ---
    free(a);
    free(c_seq);
    free(c_par);
    free(block_sums);

    return 0;
}
