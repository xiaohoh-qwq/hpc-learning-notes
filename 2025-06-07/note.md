#lecture 5 学习笔记: OpenMP 同步、数据依赖与并行算法

---

## 1. 同步 (Synchronization)

在共享内存模型中，多个线程会访问共享的全局数据，因此必须使用同步机制来**保护数据访问**，防止出现“竞争条件”(race conditions)，并确保线程间的执行顺序 。

### 1.1 Critical Section (临界区)

-  **作用**: 确保一次只有一个线程能执行特定的代码块，实现互斥 。
-  **指令**: `#pragma omp critical` 。
-  **示例**: 多个线程安全地对一个共享计数器 `count` 进行自增操作 。
    ```c
    #pragma omp parallel shared(count)
    {
        #pragma omp critical
        count++;
    }
    ```

### 1.2 Atomic (原子操作)

-  **作用**: 同样提供互斥，但仅限于**对单个内存位置的更新操作**，比 `critical` 更轻量级 。
-  **指令**: `#pragma omp atomic` 。
-  **示例**: 将每个线程的局部和 `lsum` 安全地累加到全局和 `sum` 中 。
    ```c
    #pragma omp atomic
    sum += lsum;
    ```

### 1.3 Barrier (屏障)

-  **作用**: 一个同步点。所有线程都必须到达这个点之后，才能继续执行后面的代码 。
-  **指令**: `#pragma omp barrier` 。
-  **注意**: OpenMP 的很多并行构造（如 `for`）在末尾都有一个隐含的屏障。可以使用 `nowait` 子句来移除这个隐含的屏障 。

---

## 2. 数据依赖 (Data Dependency)

数据依赖是限制并行化的关键因素。在 OpenMP 中，我们尤其需要关注**循环携带的数据依赖 (loop-carried data dependency)** 。

### 2.1 指令级数据依赖

-  **流依赖 (Flow Dependency, RAW)**: 写后读，是真正的数据依赖 。
-  **反依赖 (Anti-dependency, WAR)**: 读后写，是伪依赖，可通过重命名变量消除 。
-  **输出依赖 (Output Dependency, WAW)**: 写后写，也是伪依赖，可通过重命名变量消除 。

### 2.2 循环携带的数据依赖

-  **定义**: 指的是循环中某次迭代的计算依赖于之前迭代的结果 。
-  **问题**: 如果存在循环携带依赖，直接用 `#pragma omp for` 进行并行化会导致错误的结果，因为无法保证迭代的执行顺序 。
-   **示例与解决方案**:
    -   **存在依赖的循环**:
        ```c
        for (i=0; i<n; i++) {
            a[i] = b[i] + c[i];
            d[i] = e * a[i+1]; // a[i+1] 的值在下一次迭代中才被计算，产生依赖
        }
        ```
    -   **解决方案**: 将循环拆分为两个独立的循环。第一个循环完成后，所有 `a[i]` 的旧值都已被读取，第二个循环可以安全地更新 `a[i]` 的值。两个 `parallel for` 之间隐含的屏障确保了执行顺序 。
        ```c
        #pragma omp parallel for
        for (i=0; i<n; i++)
            d[i] = e * a[i+1];

        // 此处有一个隐含的屏障

        #pragma omp parallel for
        for (i=0; i<n; i++)
            a[i] = b[i] + c[i];
        ```

---

## 3. 归约操作 (Reduction Operation)

-   **定义**: 将一个数据集合通过一个关联运算符（如 `+`, `*`, `max`, `min`）合并成一个单一值的过程 。
-   **问题**: 一个简单的求和循环 `S = S + a[i]` 本质上是串行的，因为 `S` 的值在每次迭代中都发生变化，存在循环携带依赖 。
-   **并行思路**: 利用运算符的结合律（如加法），让每个线程先计算一个**局部和 (partial sum)**，最后再将所有局部和汇总 。
-   **OpenMP 方案**: 使用 `reduction` 子句是实现归约的标准方法 。
    -   **语法**: `reduction(operator:list)` 。
    -   **工作原理**: OpenMP 会为每个线程创建一个在 `list` 中的变量的私有副本，并根据 `operator` 对其进行初始化（如 `+` 对应 0，`*` 对应 1）。所有线程在自己的私有副本上完成计算后，OpenMP 会自动将这些私有副本的值合并到原始的共享变量中 。
    -   **示例**:
        ```c
        double S = 0;
        #pragma omp parallel for reduction(+:S)
        for (int i=0; i<n; i++) {
            S += a[i];
        }
        ```

---

## 4. 扫描操作 (Scan / Prefix Sum)

-   **定义**: 扫描操作（也称前缀和）会生成一个新数组，其中每个元素都是原数组中从开始到当前位置所有元素的累积运算结果 。例如，对 `[1,2,3,4]` 进行加法扫描，结果是 `[1,3,6,10]` 。
-   **挑战**: 简单的串行算法 `c[i] = c[i-1] + a[i]` 存在真实的循环携带依赖，无法直接并行化 。

### 4.1 并行扫描算法

为了在共享内存系统上高效并行化，可以采用一个三阶段算法 ：

1.  **阶段一：局部扫描**
    -   将数据分块，每个线程负责一个块 。
    -   每个线程对自己的数据块执行一次独立的串行扫描 。
    -   每个线程（除了最后一个）将自己块的最后一个元素（即块内总和）保存到一个辅助数组 `W` 中 。

2.  **阶段二：中间扫描**
    -   由**单个线程**对辅助数组 `W` 执行一次扫描操作 。这一步是串行的，但因为 `W` 的大小等于线程数，所以开销不大 。

3.  **阶段三：全局更新**
    -   每个线程将阶段二得到的对应值（即前面所有块的总和）加到自己块内的所有元素上 。

-   **性能**: 此并行算法的总操作数约为 `2n-1`，是串行算法 (`n-1`) 的两倍 。但由于是并行执行，当线程数大于 2 时，其并行时间 (`2n/t + t`) 通常会少于串行时间 。
-   **同步**: 每个阶段之间都需要同步（例如，使用 `barrier`）来确保前一阶段的所有计算都已完成 。
-   **实现**: 阶段二可以使用 `#pragma omp single` 指令来确保只有一个线程执行 。
