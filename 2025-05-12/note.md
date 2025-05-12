# OpenMP 学习笔记

## 一、课程主题与核心概念

如何使用 OpenMP 进行共享内存并行编程，内容涵盖并行计算基础、OpenMP 的编程接口、变量作用域以及常见问题处理等。

### 并发与并行
- **并发（Concurrency）**：多个事件在同一时间段交替发生。
- **并行（Parallelism）**：多个事件在同一时刻同时发生。

### 并行计算
- 利用多个处理器或计算核心同时进行计算任务，以提高程序运行效率。
- OpenMP 支持在共享内存模型下进行并行计算，适用于多核 CPU 系统。

### 进程与线程
- **进程（Process）**：正在运行的程序实例。
- **线程（Thread）**：进程中的最小执行单元，多个线程可共享同一进程的资源。

## 二、OpenMP 简介

- OpenMP 是用于 C/C++ 和 Fortran 的共享内存并行编程接口。
- 通过简单的编译指导语句（`#pragma omp`）、库函数和环境变量控制并行行为。
- 避免了传统消息传递模型中的复杂性，适合快速实现多线程并行。

## 三、OpenMP 编译指导语句结构

OpenMP 使用 `#pragma omp` 开头的编译指导语句，其基本结构如下：

#pragma omp <directive> [clause[[,] clause] ...]

- **directive**：指令部分，指定并行结构。
- **clause**：子句部分，提供细化行为的参数。

## 四、Directive（指令）部分常见类型

- `parallel`：定义并行区域，多个线程同时执行。
- `for`：将循环迭代分配给多个线程。
- `sections` / `section`：划分多个代码段分别由线程执行。
- `single`：只由一个线程执行代码块，其他线程等待。
- `master`：只由主线程执行，不阻塞其他线程。
- `critical`：临界区，线程按顺序进入，避免并发冲突。
- `barrier`：所有线程同步，等待彼此完成。
- `task` / `taskwait`：显式任务并行，适用于不规则并行结构。

## 五、Clause（子句）部分常见类型

- `shared(varlist)`：指定变量为共享变量，可被所有线程访问。
- `private(varlist)`：每个线程独立拥有变量副本。
- `firstprivate(varlist)`：每个线程的私有变量初始化为主线程变量的值。
- `lastprivate(varlist)`：在并行块结束后将最后一次迭代的值复制回主线程。
- `default(shared|none|private)`：控制默认变量作用域。
- `schedule(static|dynamic|guided[, chunk])`：控制循环迭代的分配策略。
- `reduction(operator: varlist)`：规约操作，用于并行求和、乘积等。

## 六、关键结论与注意事项

- OpenMP 将任务自动划分到线程，减少显式线程管理的复杂性。
- 要特别注意变量作用域，错误使用会导致竞争条件或数据不一致。
- 并行计算程序需避免死锁、数据依赖和竞态条件。
- 规约操作必须满足交换律，否则结果不确定。
- 使用如 `#pragma omp parallel for` 这样的复合指令可以提升开发效率。
- OpenMP 的运行时库提供时间测量、锁管理等函数支持。
