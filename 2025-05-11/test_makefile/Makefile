# 编译器
CC = gcc

# 编译选项
CFLAGS = -Wall -g

# 目标文件
OBJS = main.o utils.o

# 目标程序名
TARGET = app

# 默认目标
all: $(TARGET)

# 链接目标程序
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# 编译每个源文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理目标
clean:
	rm -f $(OBJS) $(TARGET)
