# Makefile 学习笔记
---

#核心要点

- 在 Linux 软件开发中，makefile 是不可或缺的构建工具，用于管理复杂项目的编译流程。
- 使用多个源文件（如 `main.cpp`、`factorial.cpp` 和 `printhello.cpp`）可以清晰组织代码，通过头文件声明函数。
- 编译大型项目时，应降低编译时间，makefile 可减少冗余步骤并提高效率。
- 逐个编译源文件并生成 `.o` 目标文件可实现高效构建，避免重复编译。

---

#关键结论

- makefile 基本结构为：目标（target）、依赖（dependencies）和命令（commands）。
- 使用 makefile 可以实现源文件的自动化构建与清理，节省重复输入的时间。
- 每个升级版本的 makefile 都在结构清晰性、灵活性和可维护性上有显著改进。

---
#第一版本：手动写出所有文件的依赖关系
hello: main.cpp printhello.cpp factorial.cpp
	g++ -o hello main.cpp printhello.cpp factorial.cpp
#第二版本：基本功能
实现最基本的构建与清理流程，结构清楚但冗长、缺乏扩展性。

#makefile
#all: main

#main: main.o factorial.o printhello.o
	g++ -o main main.o factorial.o printhello.o

#main.o: main.cpp factorial.h printhello.h
	g++ -c main.cpp

#factorial.o: factorial.cpp factorial.h
	g++ -c factorial.cpp

#printhello.o: printhello.cpp printhello.h
	g++ -c printhello.cpp

#clean:
	rm -f *.o main
#第三版本：引入变量，增强可维护性
CXX = g++
CXXFLAGS = -Wall -g

OBJS = main.o factorial.o printhello.o
TARGET = main

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o $(TARGET)

#第四版本：自动化和目录结构优化
#CXX=g++
#TARGET=HELLO
#SRC=$(wildcard *.cpp)
#OBJ=$(patsubst %.cpp,%.o,$(SRC))
#CXXFLAGS= -c -Wall 
#$(TARGET): $(OBJ)
	$(CXX) -o $@ $^
#%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
#.PHONY: clean
#clean:
	rm -f *.o $(TARGET)
