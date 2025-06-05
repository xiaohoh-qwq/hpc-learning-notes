#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int num_threads_to_set = 4; // 默认设置4个线程

    if (argc > 1) {
        num_threads_to_set = atoi(argv[1]);
        if (num_threads_to_set <= 0) {
            num_threads_to_set = 4; // 如果输入无效，则默认为4
        }
    }

    printf("程序将尝试使用 %d 个线程。\n", num_threads_to_set);

    // 使用 num_threads 子句来设置线程数量
    #pragma omp parallel num_threads(num_threads_to_set)
    {
        int tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        // 只有主线程 (线程ID为0) 打印线程总数
        if (tid == 0) {
            int nthreads = omp_get_num_threads();
            printf("Master thread (ID %d) reports: Number of threads in this parallel region = %d\n", tid, nthreads);
        }
    } // 并行区域结束

    return 0;
}
