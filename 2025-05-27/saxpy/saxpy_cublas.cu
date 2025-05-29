#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    }

#define CHECK_CUBLAS(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    }

int main() {
    const int N = 1024;
    const float alpha = 2.0f;
    std::vector<float> x(N, 1.0f);
    std::vector<float> y(N, 2.0f);

    // ✅ 检查 GPU 是否可用
    int device_count = 0;
    cudaError_t devErr = cudaGetDeviceCount(&device_count);
    if (devErr != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device is detected." << std::endl;
        return -1;
    }

    // ✅ 设置设备
    CHECK_CUDA(cudaSetDevice(0));

    // 分配 GPU 内存
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 使用 cuBLAS 执行 saxpy
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1));

    CHECK_CUBLAS(cublasDestroy(handle));

    // 拷回主机并打印一个值以验证
    CHECK_CUDA(cudaMemcpy(y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "y[0] = " << y[0] << std::endl;  // 应该是 2 + 2 * 1 = 4

    // 释放 GPU 内存
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
