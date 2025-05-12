#include <stdio.h>
#include <omp.h>

int main() {
    int tid, mcpu;

    // 获取当前线程号
    tid = omp_get_thread_num(); 
    // 获取当前并行区域中的线程数
    mcpu = omp_get_num_threads(); 
    printf("hello from thread %d in %d CPUs\n", tid, mcpu);
    printf("------before parallel\n");
    printf("\n");
    printf("------during parallel\n");
    // 开启并行区域，指定线程个数为3，并且tid和mcpu变量在每个线程中有私有副本
    #pragma omp parallel num_threads(3) private(tid,mcpu) 
    {
        // 在并行区域内重新获取当前线程号
        tid = omp_get_thread_num(); 
        // 在并行区域内重新获取当前并行区域中的线程数
        mcpu = omp_get_num_threads(); 
        printf("hello from thread %d in %d CPUs\n", tid, mcpu);
    }
    printf("\n");
    printf("------after parallel\n");
    printf("hello from thread %d in %d CPUs\n", tid, mcpu);
    return 0;
}
