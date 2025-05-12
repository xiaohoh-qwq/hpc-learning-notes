# 编译器
CC = gcc

# 编译选项
CFLAGS = -Wall -g

# 源文件和目标文件
SRC = $(wildcard *.c)
OBJ = $(patsubst %.c,%.o,$(SRC))

# 目标程序名
TARGET = test

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理目标
clean:
	rm -f $(OBJ) $(TARGET)

