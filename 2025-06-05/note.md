# Lecture 4学习笔记

## 一、共享内存平台基础

### 1.1 基本概念
- **并发执行**：多个线程同时创建并运行。
- **共享与私有变量**：
    - **全局变量**：由所有线程共享。
    - **局部私有变量**：每个线程拥有自己的副本。
- **通信与协调**：
    - **隐式通信**：通过共享变量。
    - **显式协调**：通过对共享变量的同步操作。
- **平台规模与并行粒度**：
    - 共享内存机器通常规模较小（例如，大型机器约100核）。
    - 需要**粗粒度并行 (coarse-grained parallelism)**。
- **编程技术**：
    - Pthread：显式创建线程和同步线程执行。
    - **OpenMP**：使用编译器指令和子句（本课程重点）。

### 1.2 线程基础 (Thread Basics)
- **模型**：一个进程包含多个线程。
- **共享资源**：线程在同一进程中共享相同的地址空间和进程状态，这显著降低了上下文切换的成本。
- **交互方式**：线程并发运行，通过读/写共享地址空间中的相同位置进行交互。
- **执行顺序**：**不能假设任何执行顺序**。任何特定的执行顺序都必须通过同步机制来建立。
- **潜在问题**：需要关注死锁 (deadlock)、饥饿 (starvation) 和性能问题。
- **系统支持**：所有通用现代操作系统都支持线程。

### 1.3 共享内存平台编程要点
- **数据共享与设计**：全局变量共享看似简化了数据移动，但**数据划分 (data partitioning)** 和 **数据局部性 (data locality)** 仍然非常重要。设计时必须认真考虑指令级并行 (ILP)、内存层次结构和缓存效应。数据到线程的分布通常是隐式的。
- **显式同步 (Explicit Synchronization)**：
    - **目的1：保护共享数据**，防止**竞争条件 (race condition)**。
    - **目的2：强制执行顺序约束**。
    - **挑战**：需要防止应用同步时可能出现的**死锁 (deadlock)**。

### 1.4 竞争条件 (Race Condition)
- **定义**：
    - 多个线程同时读写共享数据项。
    - 最终结果变得不可预测，取决于哪个线程“最后完成比赛”（即最后完成操作）。
- **示例**：两个线程，一个对共享计数器执行自增操作，另一个执行自减操作。若初始值为5，无同步情况下，最终结果可能是4、5或6。
- **原因**：高级语言中的简单操作（如 `count++` 或 `count--`）在机器层面会分解为多个指令（典型的“读-修改-写”序列）：
    - `count++`:
        1. `register1 = count` (读)
        2. `register1 = register1 + 1` (修改)
        3. `count = register1` (写)
    - `count--`:
        1. `register2 = count` (读)
        2. `register2 = register2 - 1` (修改)
        3. `count = register2` (写)
    - 这些机器级指令在执行时可能发生交错，导致最终结果不确定。
- **解决方案**：需要确保一个线程的原子操作（如自增）完成后，另一个线程才能进行相关操作，即实现**互斥 (mutual exclusion)**。

## 二、示例：矩阵乘法的并行化

### 2.1 细粒度 vs. 粗粒度并行
- **任务依赖图**：可以为矩阵乘法绘制任务依赖图。
- **细粒度并行**：从图中可以看出，所有输出元素 $c_{ij}$ 都可以并发计算。
- **共享内存机器的需求**：对于共享内存机器，我们通常需要更**粗粒度的并行**。

### 2.2 粗粒度并行策略：划分输出矩阵 C
- **数据依赖性**：计算输出矩阵 C 的不同元素之间没有数据依赖，因此无需同步，属于**易并行 (embarrassingly parallel)**。
- **数据与工作量特性**：规则的数据结构和已知的工作量，适合进行**静态划分 (static partitioning)**。
- **划分方式示例**：
    - 1D 循环 (1D Cyclic)
    - 1D 块 (1D Block)
    - 1D 块循环 (1D Block Cyclic)

### 2.3 1D 划分方式图示
- **1D 循环划分 (1D Cyclic Partitioning)**：将输出矩阵的行（或列）逐个、循环地分配给线程，直到所有行（或列）都被分配完毕。
- **1D 块划分 (1D Block Partitioning)**：将输出矩阵的行（或列）首先分组为连续的块，然后每个块分配给一个线程。
- **1D 块循环划分 (1D Block Cyclic Partitioning)**：首先将行（或列）分组为块，确保块的数量远多于线程的数量。然后，这些块以循环的方式逐个分配给线程。

### 2.4 矩阵乘法在共享内存机器上的关注点
- **并行特性**：没有数据依赖，易并行，理论上可以实现线性加速比。
- **性能优化焦点**：应重点关注**每个核心上的串行计算效率**，例如：
    - 缓存效应 (cache effect)
    - 内存层次结构 (memory hierarchy)
    - 循环展开 (loop unrolling)
    - 指令级并行 (ILP)

## 三、使用 OpenMP 进行共享内存编程

### 3.1 OpenMP 简介
- **全称**：Open Multi-Processing。
- **定义**：一个基于**编译器指令 (directive-based)** 的应用程序编程接口 (API)，用于在共享内存架构上开发并行程序。
- **语言支持**：C/C++ 和 Fortran。
- **核心组成**：
    - **编译器指令 (Compiler directives)**：以 `#pragma omp` 开头，指导编译器生成并行代码。
    - **库函数调用 (Library Calls)**：需要包含头文件 `<omp.h>`。
    - **环境变量 (Environment Variables)**：例如 `OMP_NUM_THREADS`，用于控制运行时行为。

### 3.2 OpenMP 的动机
- **简化并行编程**：直接使用线程库（如 Pthreads）通常复杂且容易出错（需要处理线程创建、同步、条件变量等）。
- **弥补自动并行化编译器的不足**：尽管尝试多年，全自动将串行程序转换为高效并行程序的编译器进展有限。
- **显式并行化**：OpenMP 允许程序员通过相对较少的代码注解来指定并行性和数据独立性，用简单的指令隐藏了繁琐的线程调用。

### 3.3 OpenMP 程序员视角
- **特性**：可移植、线程化、共享内存编程规范，语法“轻量”。需要编译器支持。
- **OpenMP 可以做什么**：
    - 允许程序员将程序划分为串行区域和并行区域，而不是显式创建并发执行的线程。
    - 隐藏堆栈管理。
    - 提供一些同步构件。
- **OpenMP 不能做什么**：
    - 不会自动并行化。
    - 不保证一定能获得加速比。
    - 不能完全免除数据竞争的风险（程序员仍需负责保证数据访问的正确性）。

### 3.4 OpenMP 执行模型：Fork-Join 并行
- **模型描述**：
    - 程序开始时只有一个主线程 (Master thread)。
    - 当遇到并行区域时，主线程派生 (fork) 一个线程团队 (team of threads)。主线程也是团队的一员。
    - 并行区域内的代码由团队中的所有线程并行执行。
    - 并行区域结束后，团队线程汇合 (join) 到主线程，团队解散（或挂起），只有主线程继续执行后续的串行代码。
- **增量并行化**：并行性可以逐步添加到串行程序中，直到达到性能目标。

### 3.5 C/C++ 通用代码结构示例
```c
#include <omp.h> // OpenMP 头文件
#include <stdio.h>

int main() {
    int var1, var2, var3;

    // 串行代码部分
    // ...

    // 并行区域开始。派生线程团队。
    // 指定变量作用域
    #pragma omp parallel private(var1, var2) shared(var3)
    {
        // 此并行段由所有线程执行
        // ...

        // 其他 OpenMP 指令
        // ...

        // 运行时库函数调用
        var1 = omp_get_thread_num(); // 示例：获取线程ID
        // ...
    } // 所有线程在此汇合到主线程，团队解散

    // 恢复串行代码执行
    // ...
    return 0;
}
### 3.6 OpenMP 并行区域构件 (`#pragma omp parallel`)
- **基本语法**: `#pragma omp parallel [clause ...]` 后紧跟一个结构化代码块 `{ ... }`。
- **行为**:
    - 当一个线程遇到 `parallel` 指令，它会创建一个线程团队，并成为该团队的主线程（线程ID为0）。
    - 从并行区域开始，团队中的所有线程都会执行该区域内的相同代码。
    - 并行区域末尾有一个**隐式的壁垒 (barrier)**。所有线程必须到达此壁垒后，只有主线程才会继续执行并行区域之后的代码。
- **线程数量控制**:
    - **默认**: 通常由实现定义，一般是节点上的CPU核心数。
    - **库函数**: `omp_set_num_threads(int num_threads)`。
    - **子句 (Clause)**: `num_threads(integer_expression)`，例如 `#pragma omp parallel num_threads(4)`。
    - **获取信息函数**:
        - `omp_get_num_procs()`: 获取可用的处理器核心数。
        - `omp_get_num_threads()`: 在并行区域内调用，获取当前团队中的线程数。
        - `omp_get_thread_num()`: 在并行区域内调用，获取当前线程的ID (0 到 N-1)。
- **共享与私有变量**:
    - **默认共享**: 在并行区域之外声明的变量，默认情况下在团队中的所有线程之间是共享的。所有线程访问的是同一块内存，需要小心处理以避免竞争条件。
    - **默认私有**: 在并行区域之内声明的变量（不包括静态变量），默认情况下对于每个线程是私有的。每个线程拥有该变量的一个独立副本。
    - **子句 (Clauses) 用于显式控制**:
        - `private(list)`: 列表中的变量对于每个线程都是私有的。每个线程会创建一个同类型的新对象。
        - `shared(list)`: 列表中的变量在团队中的所有线程之间共享。

### 3.7 OpenMP "Hello World" 示例
**串行版本:**
```c
#include <stdio.h>

int main() {
    printf("Hello world\n");
    return 0;
}


**OpenMP 并行版本 (基本):**
```c
#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel // 开始并行区域
    {
        // 这段代码会被多个线程并行执行
        printf("Hello world from thread = %d\n", omp_get_thread_num());
    } // 并行区域结束，线程汇合
    return 0;
}
// 编译: gcc -fopenmp hello_omp.c -o hello_omp
**OpenMP 并行版本 (使用 private 子句):**
```c
#include <omp.h>
#include <stdio.h>

int main() {
    int tid; // 在并行区域外声明
    #pragma omp parallel private(tid) // tid 对每个线程私有
    {
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);
    }
    return 0;
}
### 3.8 OpenMP 工作共享构件：`for` 指令 (`#pragma omp for`)
- **基本语法**: `#pragma omp for [clause ...]` 后紧跟一个 `for` 循环。
- **行为**:
    - `for` 指令指定其后的循环迭代必须由当前线程团队并行执行。
    - **必须位于一个并行区域 (`#pragma omp parallel`) 内部**。
    - 循环的迭代会被分配给团队中的不同线程。
    - **循环控制变量 (如 `i`) 默认是私有的**。
    - 循环末尾有一个**隐式的壁垒**：所有线程必须完成它们分配到的迭代份额后，才能有线程继续执行循环之后的代码。
- **示例结构**:
```c
#pragma omp parallel
{
    // 其他并行代码...
    int i, N; // N 和其他变量根据上下文定义
    // int a[N], b[N], c[N]; // 假设数组已定义和初始化
    #pragma omp for
    for (i = 0; i < N; i++) {
        // c[i] = a[i] + b[i]; // 示例操作
    }
    // 其他并行代码...
}
- **快捷方式**: `#pragma omp parallel for`
    - 将并行区域的创建和 `for` 循环的工作共享合并到一条指令中。
    ```c
    // int i, MAX; // MAX 和其他变量根据上下文定义
    // double res[MAX]; // 假设数组已定义
    #pragma omp parallel for
    for (i = 0; i < MAX; i++) {
        // res[i] = some_function(); // 示例操作
    }
    ```
    这等同于先写 `#pragma omp parallel` 再在其内部写 `#pragma omp for`。
- **工作划分方式**:
    - `#pragma omp parallel for` **默认情况下通常采用块划分 (block partition)** 或类似的静态调度方式将迭代分配给线程。
- **使用循环的要点**:
    1.  **找到计算密集型循环**。
    2.  **确保循环迭代是独立的 (independent)**：即一次迭代的计算不依赖于其他迭代的结果（没有循环携带依赖, loop-carried dependencies）。这样才能安全地以任意顺序并行执行迭代。
    3.  放置合适的 OpenMP 指令并测试。
- **消除循环携带依赖示例**:
    ```c
    // 存在循环依赖 (j 的值依赖于前一次迭代)
    // int i, j, A[MAX];
    // j = 5;
    // for (i = 0; i < MAX; i++) {
    //     j += 2;
    //     A[i] = big(j);
    // }

    // 消除依赖后的可并行版本
    // int i, A[MAX]; // MAX 和 big() 根据上下文定义
    #pragma omp parallel for
    for (i = 0; i < MAX; i++) {
        int j_local = 5 + 2 * (i + 1); // 每个线程计算自己的 j
        // A[i] = big(j_local); // 示例操作
    }
    ```

### 3.9 OpenMP 示例：矩阵向量乘法 (`b = A*x + b`)
- **串行代码 (假设 b 已初始化为0或初始值)**:
```c
// int i, k, m, n;
// double A[m][n], x[n], b[m]; // 假设已定义和初始化
for (i = 0; i < m; i++) {       // 外层循环，遍历结果向量 b 的每个元素
    for (k = 0; k < n; k++) {   // 内层循环，计算点积
        b[i] += A[i][k] * x[k];
    }
}
- **并行化哪个循环？**
    - **内层 `k` 循环**: 如果并行化内层 `k` 循环，多个线程会同时尝试更新同一个 `b[i]` 元素，这将导致竞争条件和循环携带依赖（除非使用 `reduction` 或原子操作，这会更复杂）。
    - **外层 `i` 循环**: 外层 `i` 循环的每次迭代计算一个独立的 `b[i]` 元素，不同迭代之间没有数据依赖。这是理想的并行化目标。
- **OpenMP 并行化外层 `i` 循环**:
```c
// int i, k, m, n;
// double A[m][n], x[n], b[m]; // 假设已定义和初始化
#pragma omp parallel for shared(A, x, b, m, n) private(i, k)
for (i = 0; i < m; i++) {
    for (k = 0; k < n; k++) {
        b[i] += A[i][k] * x[k];
    }
}
- `shared(A, x, b, m, n)`: 矩阵A、向量x、结果向量b以及维度m, n是所有线程共享的。
- `private(i, k)`: 循环变量i和k对于每个线程是私有的。`i` 作为外层并行循环的索引，OpenMP 会自动处理其私有化和迭代分配。`k` 在每个外层循环迭代内部使用，也应设为私有。

### 3.10 OpenMP 计时函数 `omp_get_wtime()`

- **功能**: 一个可移植的墙上时钟 (wall clock) 计时函数。
- **返回值**: 返回一个双精度浮点数，表示从过去某个任意时间点到当前调用时刻所经过的秒数。
- **使用方法**: 通常成对使用，通过两次调用的差值来获取一个代码块的执行时间。
```c
double start_time, end_time, elapsed_time;
start_time = omp_get_wtime();
// ... 需要计时的代码块 ...
end_time = omp_get_wtime();
elapsed_time = end_time - start_time;
printf("Time taken: %f seconds\n", elapsed_time);
