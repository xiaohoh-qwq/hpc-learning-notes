# CMU 15-418 第五讲学习笔记：并行任务调度与Cilk Plus运行时模型

本讲深入讨论了并行编程中如何调度任务、平衡负载以及减少通信开销，并引入了 Cilk Plus 语言及其运行时系统作为实际并行任务调度的实现案例。重点在于理解静态/动态/半静态分配的权衡，掌握任务队列的调度策略，并理解背后的运行时调度原理。

---

## 1. 工作负载分配策略

- **静态任务分配**  
  - 在程序开始前预先将任务平均分配到各处理器。  
  - 优点：调度开销小，执行简单。  
  - 缺点：不适合任务执行时间不均或数据依赖动态变化的情况，可能导致负载不均。

- **动态任务分配**  
  - 程序运行时按需分配任务，各处理器可以动态获取更多任务。  
  - 适合任务耗时难以预测的场景，如不规则计算（图遍历、稀疏矩阵）。  
  - 缺点：调度开销变大，可能引入锁争用、通信延迟。

- **半静态任务分配**  
  - 折中策略：先静态划分，再定期重新评估负载，调整分配。  
  - 优点：结合静态分配的高效和动态分配的弹性。

---

## 2. 并行编程的优化与度量

- 并行编程不是一次性完成的过程，而是一个迭代的优化流程。  
- 必须通过实际性能测量（如任务吞吐率、线程利用率）不断调整策略。

- **任务粒度选择的重要性**  
  - 粒度太细：调度开销高，线程切换频繁。  
  - 粒度太粗：并行性不够，核心资源浪费。  
  - 应视任务特性灵活调整。

- **动态调度中的任务划分策略**  
  - 可通过调整“每次抓取的任务大小”来控制调度粒度。  
  - 分布式任务队列可减少通信和锁争用开销。

- **任务独立性与可并行性**  
  - 若任务之间没有数据依赖（如多个质数检查），可以并发执行，提升速度。

---

## 3. Cilk Plus：并行语言模型

Cilk Plus 是对 C/C++ 的扩展，支持**细粒度并行性**和**任务级并行编程**。

- **核心关键词**  
  - `cilk_spawn`：创建可以并行执行的子任务。  
  - `cilk_sync`：等待所有 spawn 出来的子任务完成。

- **语义示例**  
  ```c
  cilk_spawn f();  // f 可以与后续代码并行执行
  ...
  cilk_sync;       // 等待所有并行子任务完成

### 语义说明

- 每一个 spawn 生成的任务会被放入一个任务队列中。
- sync 的作用类似于 barrier，确保在继续前所有子任务均已完成。
- 所有并行任务并不真正创建线程，而是由运行时管理的轻量线程池调度执行。

## 4. Cilk 运行时系统与任务队列

- 每个处理器维护一个双端任务队列（deque）：
  - 本地线程从队尾取任务（LIFO方式）执行。
  - 若本地任务队列为空，会从其他线程的队头窃取任务（FIFO方式）。

- 调度策略：Work Stealing
  - 每个线程优先执行自己的工作；当空闲时，从其他线程偷任务。
  - 被偷的是延续任务（continuations），即 spawn 后留下的部分。
  - 避免偷子任务，因为偷延续任务具有更强的局部性。

## 5. 延续窃取与调度策略

### 延续窃取（Continuation Stealing）

- 与其偷 spawn 出来的子任务，不如偷主线程剩下的后续任务。

#### 优势

- 主线程继续执行自己 spawn 的子任务（缓存友好）。
- 其他线程执行主线程的后续代码（可平衡负载）。

### 贪婪策略（Greedy Work-First）

- 优先执行产生更多并行任务的分支。
- 提高系统整体的工作产生速率。

### 滚动死沉（Greedy Work-First + Lazy Stealing）

- 本地线程保持忙碌（优先本地任务）。
- 空闲线程懒惰地从其他线程任务队列中偷任务。

## 6. 多线程并行执行的调度策略实现

### 子任务执行策略

- 子任务可以由当前线程执行，主线程的后续由其他线程执行（推荐方式）。
- 或反过来由其他线程执行子任务（不推荐，缓存不友好）。

### 任务队列调度模型

- 所有任务存放在任务队列中，由多个线程从中抓取执行。
- 队列调度方式决定了任务的局部性和负载平衡。

### 调度策略选择

- 贪婪调度策略：主动扩展工作负载。
- 阻塞策略：减少上下文切换，保持局部执行。

## 7. 总结

- 本讲内容强调任务调度策略对并行性能的核心影响。
- 静态、动态、半静态任务分配各有适用场景，编程中应根据任务类型与系统资源灵活调整。
- Cilk Plus 的运行时系统通过轻量任务队列 + 延续窃取 + 贪婪调度 实现了高效的并行执行模型。
- 并行编程需要不断测量、分析、优化，才能真正提升程序性能。
