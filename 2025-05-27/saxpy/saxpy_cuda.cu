#include <iostream>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int n = 1 << 20;
    float *x, *y, *d_x, *d_y;
    float a = 2.0f;

    x = new float[n]; y = new float[n];
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<(n + 255)/256, 256>>>(n, a, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "y[0] = " << y[0] << std::endl;

    cudaFree(d_x); cudaFree(d_y);
    delete[] x; delete[] y;
    return 0;
}
