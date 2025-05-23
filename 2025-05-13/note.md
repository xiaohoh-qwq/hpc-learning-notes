# GDB 调试工具学习笔记

## 一、概述

GDB（GNU Debugger）是 GNU 项目下的调试器，广泛应用于 C、C++、Go、Rust、汇编等语言的程序调试中。它允许开发者在程序运行时设置断点、查看变量、跟踪执行流程、修改内存，从而定位程序中的错误或理解程序行为。

---

## 二、核心要点

1. GDB 是一款强大的调试工具，支持多种语言，适用于不同平台和架构。
2. 使用调试器的目的不仅是排错，也是深入理解程序执行过程的重要方式。
3. 学习任何新工具或框架时，优先阅读其官方文档，获取系统化、准确的信息。

---

## 三、关键结论

1. GDB 是程序开发中不可或缺的工具，特别是在处理段错误、内存越界、递归死循环等问题时极其有效。
2. 工具的学习应强调“概念—应用—原理”的完整路径，避免停留在浅层命令记忆。
3. GDB 的断点、堆栈回溯、变量查看、条件观察点等功能构成其调试能力的核心。
4. 多语言与跨平台特性增强了 GDB 的实用价值，特别适合系统级开发者使用。
5. 掌握调试器，有助于开发者反向理解程序底层机制，是从“写代码”迈向“理解程序”的关键步骤。

---

## 四、重要细节

1. 使用 `gcc -g` 编译程序可生成调试信息，供 GDB 使用。
2. GDB 支持设置断点（`break`）、单步执行（`next`、`step`）、查看调用栈（`backtrace`）、打印变量值（`print`）、设置观察点（`watch`）等功能。
3. 调试时可查看寄存器、内存地址、函数调用帧等底层状态。
4. 官方文档是获取 GDB 正确用法的首选资料：[https://www.gnu.org/software/gdb/documentation](https://www.gnu.org/software/gdb/documentation)

---

## 五、常用命令汇总

run # 运行程序
break 行号/函数名 # 设置断点
list # 查看源代码
info b # 查看断点的情况
next # 执行下一行，不进入函数
step # 执行下一行，若为函数调用则进入函数体
continue # 继续运行，直到下一个断点或程序结束
print 变量 # 打印变量的当前值
backtrace # 查看当前的函数调用栈
watch 变量 # 设置观察点，变量发生变化时暂停
quit # 退出 GDB
---

## 六、实践建议

- 编写故意引发段错误的程序进行调试练习
- 使用递归函数并观察栈的增长与函数调用链
- 实验观察点功能，调试某个变量在循环中的变化情况
- 逐步跟踪数组越界等内存错误
- 尝试调试包含汇编代码的程序，理解寄存器状态的变化

---
