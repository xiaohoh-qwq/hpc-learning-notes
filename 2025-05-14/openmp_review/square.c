#include <omp.h>

void square_array(int *a, int *result, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * a[i];
    }
}
