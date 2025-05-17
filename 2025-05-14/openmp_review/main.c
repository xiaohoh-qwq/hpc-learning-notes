#include <stdio.h>
#include <omp.h>
#include "square.h"
//计算元素平方
int main() {
    int n = 8;
    int a[8] = {1,2,3,4,5,6,7,8};
    int result[8];

    square_array(a, result, n);

    for (int i = 0; i < n; i++)
        printf("%d^2 = %d\n", a[i], result[i]);

    return 0;
}
