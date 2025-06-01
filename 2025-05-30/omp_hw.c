#include<stdio.h>
#include<omp.h>
int main(){
	#pragma omp parallel
	{
		int tid=omp_get_thread_num();
		printf("hello world form the thread %d\n",tid);
	}
	return 0;
}
