#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>  // SSE2
#include <time.h>

void simd_add(const float *a, const float *b, float *c, int n) {
    int i;
    for (i = 0; i < n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(&c[i], vc);
    }
}

void scalar_add(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1 << 20;
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    float *c = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    clock_t start, end;

    start = clock();
    scalar_add(a, b, c, n);
    end = clock();
    printf("标量加法时间: %f 秒\n", (end - start) / (double)CLOCKS_PER_SEC);

    start = clock();
    simd_add(a, b, c, n);
    end = clock();
    printf("SIMD 加法时间: %f 秒\n", (end - start) / (double)CLOCKS_PER_SEC);

    free(a);
    free(b);
    free(c);
    return 0;
}
