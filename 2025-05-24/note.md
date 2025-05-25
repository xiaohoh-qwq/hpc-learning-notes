# HPC Class 矩阵乘法优化实验笔记

本笔记总结了多个矩阵乘法实现的不同优化策略，结合性能实验结果和对应的图片，帮助理解各类优化技术的效果与原理。

---

## 目录结构与对应实验结果

- **matmul**  
  - 最基础的矩阵乘法实现，采用三层循环的朴素写法，未做任何编译器优化或算法优化。
  - 该版本主要用于性能基线对比。
  - ![matmul](images/matmul.png)

- **matmul_baseline**  
  - 稍作代码规范和变量优化的版本，确保与基础版本语义相同。
  - 作为性能对比基准，便于量化后续优化带来的提升。
  - ![matmul_baseline](images/matmul_baseline.png)

- **matmul_Ofast**  
  - 使用 GCC 的 `-Ofast` 编译选项，该选项启用所有 `-O3` 优化和额外的激进优化（如浮点数学上的不严格规范）。
  - 优点是显著提升代码运行速度，缺点可能是数值精度或行为微小偏差。
  - 这一级别的编译优化主要靠编译器的自动向量化、循环展开等技术。
  - ![matmul_Ofast](images/matmul_Ofast.png)

- **matmul_Ofast_openmp**  
  - 基于 `-Ofast` 编译基础上，加入 OpenMP 多线程并行编程接口。
  - 利用多核 CPU 并行计算，将矩阵乘法中的循环划分为多线程执行，理论上能将速度提升接近核数倍。
  - 线程同步与负载均衡是关键难点，合理分配循环迭代可减少线程开销。
  - ![matmul_Ofast_openmp](images/matmul_Ofast_openmp.png)

- **matmul_Ofast_openmp_block**  
  - 在多线程基础上，进一步引入缓存阻塞（Cache Blocking）优化。
  - 矩阵乘法时访问的数据远大于缓存容量，直接按行列访问会导致频繁的缓存未命中，影响性能。
  - 缓存阻塞将矩阵划分成适合缓存大小的小块，最大化缓存重用率，降低内存带宽压力。
  - 这是一种经典的内存层次优化技术，显著提高程序对缓存的友好度。
  - ![matmul_Ofast_openmp_block](images/matmul_Ofast_openmp_block.png)

- **matmul_Ofast_openmp_block_avx**  
  - 结合 AVX 指令集的 SIMD 向量化优化。
  - AVX（Advanced Vector Extensions）允许单条指令对多个浮点数进行并行计算，极大提高单指令处理吞吐量。
  - 结合缓存阻塞与多线程，充分发挥现代 CPU 的硬件并行能力。
  - 需要编译器自动向量化支持，或手动用内嵌汇编/编译器指令实现。
  - ![matmul_Ofast_openmp_block_avx](images/matmul_Ofast_openmp_block_avx.png)

- **matmul_Ofast_openmp_block_mkl**  
  - 利用 Intel Math Kernel Library (MKL)，这是一个工业级高度优化的数学函数库。
  - MKL 内部集成多线程、SIMD指令、缓存优化和算法层面优化，性能接近处理器极限。
  - 使用库函数替代自写代码，大幅减少开发难度且性能优异。
  - 适合实际工程项目快速调用高效数学计算。
  - ![matmul_Ofast_openmp_block_mkl](images/matmul_Ofast_openmp_block_mkl.png)

- **matmul_Ofast_openmp_block_openblas**  
  - OpenBLAS 是开源的高性能 BLAS（基础线性代数子程序）库，支持多平台和多线程。
  - 类似 MKL，集成底层汇编优化和多线程。
  - 性能稍逊于 MKL，但灵活免费，广泛用于科研和开源项目。
  - ![matmul_Ofast_openmp_block_openblas](images/matmul_Ofast_openmp_block_openblas.png)

---

## 详细优化知识点解析

### 1. 编译器优化等级 `-Ofast`

- `-Ofast` 选项开启激进的优化策略，允许编译器忽略部分 IEEE 浮点标准和严格别名规则，换取更高效的代码。
- 例如循环展开（loop unrolling）、函数内联（inlining）、自动矢量化（auto vectorization）等手段。
- 适合对性能要求极高且能容忍少量精度误差的场景。

### 2. OpenMP 多线程并行

- OpenMP 提供简单的并行编程模型，使用 `#pragma omp parallel for` 语句指示编译器并行化循环。
- 通过线程池利用多核 CPU，减少串行执行时间。
- 需注意线程调度、共享变量保护（如加锁或原子操作）、负载均衡等问题。

### 3. 缓存阻塞优化（Cache Blocking）

- 现代处理器内存层次深，访问主内存代价高。
- 缓存阻塞技术把矩阵分成块，保证每块数据在处理期间尽可能保留在高速缓存中，减少缓存未命中。
- 以空间换时间，提升内存访问效率。

### 4. SIMD 向量化（AVX）

- SIMD 可以同时处理多个数据点，适合大量重复的数值计算。
- AVX指令集有256位（AVX2）和512位（AVX-512）版本，能同时处理8或16个单精度浮点数。
- 提升计算密集型代码的吞吐量。

### 5. 高性能数学库（MKL & OpenBLAS）

- 库函数是工业界优化矩阵运算的结晶，适合快速集成。
- 内部细节如流水线优化、预取指令、汇编代码手写等保证了极限性能。
- 方便开发者直接调用，省去自行调优成本。

---


