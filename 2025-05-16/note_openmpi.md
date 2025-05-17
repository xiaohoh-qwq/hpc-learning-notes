# OpenMPI 初步概念学习笔记

---

## 一、什么是 OpenMPI？

OpenMPI（Open Message Passing Interface）是一个实现了 MPI（Message Passing Interface）标准的开源库，主要用于**分布式内存并行计算**，广泛应用于高性能计算（HPC）领域。

它允许多个运行在不同节点的进程**通过消息传递的方式协作计算**，是实现 **大规模并行算法**的基础工具之一。

---

## 二、MPI 与共享内存并行的区别

| 特性               | OpenMP（共享内存）      | MPI（分布式内存）           |
|--------------------|--------------------------|------------------------------|
| 内存结构           | 所有线程共享主内存       | 每个进程拥有独立内存         |
| 通信方式           | 隐式（读写同一变量）     | 显式（Send / Receive）       |
| 并行粒度           | 通常用于中小规模并行     | 擅长处理大规模并行           |
| 部署范围           | 同一台机器（多核CPU）    | 多台机器（集群、超算）       |

---

## 三、OpenMPI 核心概念

### 1. MPI 进程

OpenMPI 程序启动时会运行多个独立的**进程**（而不是线程），这些进程之间不会自动共享变量。

```bash
mpirun -np 4 ./my_mpi_program
表示启动 4 个 MPI 进程。

### 2. MPI 初始化与结束

所有 MPI 程序都必须以如下形式开始和结束：
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);      // 初始化MPI环境

    // 用户代码

    MPI_Finalize();              // 清理MPI环境
    return 0;
}
表示启动 4 个 MPI 进程。
### 3. 获取进程信息
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 当前进程编号（rank）
MPI_Comm_size(MPI_COMM_WORLD, &size);  // 总进程数
### 4.消息传递（点对点通信）
MPI_Send(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
MPI_Recv(&data, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
dest：目标进程编号

source：发送方进程编号

tag：消息标签，用于消息区分

MPI_COMM_WORLD：默认通信器，表示“所有进程”


### 5.集合通信（常见模式）
# OpenMPI 集合通信常见模式

| 通信模式      | 函数名示例                | 说明                             |
|---------------|---------------------------|----------------------------------|
| 广播（Broadcast） | `MPI_Bcast`               | 将一份数据广播给所有进程         |
| 收集（Gather）   | `MPI_Gather` / `MPI_Gatherv` | 汇总各进程数据到一个进程         |
| 分发（Scatter） | `MPI_Scatter`             | 将数据从一个进程分发到所有进程   |
| 全收集（Allgather） | `MPI_Allgather`          | 所有进程收集所有数据             |
| 规约（Reduce）   | `MPI_Reduce` / `MPI_Allreduce` | 并行版的 sum、max 等归约操作   |

