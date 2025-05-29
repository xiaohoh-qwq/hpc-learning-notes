# GPU 与高性能计算概览

## 一、GPU 对高性能计算的意义

GPU（Graphics Processing Unit）最初用于图形渲染，但如今其强大的并行处理能力使其成为高性能计算的重要组成部分。

### 1.1 并行架构

GPU 拥有成百上千个核心，采用 SIMT（Single Instruction, Multiple Threads）执行模型，非常适合处理大规模的数据并行任务，如：

- 矩阵运算、向量处理
- 科学模拟
- 图像处理、机器学习训练

### 1.2 提升性能与吞吐量

在执行浮点密集型任务时，GPU 可提供远高于 CPU 的吞吐率。例如：

- 单精度浮点性能（FP32）可达数十 TFLOPS
- Tensor Core 支持混合精度，进一步加速深度学习任务

### 1.3 能效比高

GPU 在每瓦特计算能力上通常优于 CPU，这使其在绿色计算与超算集群中广受青睐。

---

## 二、CUDA 简介

CUDA（Compute Unified Device Architecture）是 NVIDIA 提供的并行编程平台，允许开发者使用 C/C++/Fortran 等语言在 GPU 上运行通用计算任务。

### 2.1 CUDA 特性

- 使用 `__global__` 定义 GPU kernel
- 提供线程块（block）和网格（grid）模型
- 支持显存管理（如 `cudaMalloc`, `cudaMemcpy`）
- 提供优化库（cuBLAS、cuFFT、cuDNN 等）

### 2.2 编程模型示意

```cpp
__global__ void kernel(...) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    ...
}

CUDA 通过将大量线程映射到 GPU 核心，显著加速可并行化的任务。
###  CPU vs GPU（对比表）

| 属性       | CPU（中央处理器）          | GPU（图形处理器）              |
|------------|----------------------------|-------------------------------|
| 核心数     | 少（4~32 个）              | 多（几百到上万个）             |
| 主频       | 高                         | 低                           |
| 并行模式   | 任务并行（Task Parallel）   | 数据并行（Data Parallel）     |
| 控制能力   | 强                         | 弱                           |
| 应用领域   | 系统控制、串行任务         | 数值密集型、大数据量处理       |

---

### 三、GPU 的重要周边技术

- **NVLink**：NVIDIA 高速互联总线，比 PCIe 更快，支持多个 GPU 间高速通信。
- **Infiniband**：用于节点间通信的高带宽低延迟网络，常见于超算集群。
- **GPUDirect**：允许 GPU 直接访问 RDMA 网络设备和其他 GPU 内存，减少 CPU 中转。
- **Tensor Core**：支持混合精度计算的专用核心，极大提升深度学习中矩阵乘法速度。
- **NVIDIA HPC SDK**：包括编译器（nvc、nvc++）、性能库、OpenACC 支持工具。

---
