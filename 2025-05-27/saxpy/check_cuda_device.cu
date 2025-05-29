#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "CUDA device count: " << count << std::endl;
    return 0;
}
