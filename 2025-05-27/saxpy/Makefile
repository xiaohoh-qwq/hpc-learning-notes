# 编译器配置
CXX = g++
NVCC = nvcc
ACCFLAGS = -acc -ta=tesla:cc70 -Minfo=accel

# 编译选项
CXXFLAGS = -O2
NVCCFLAGS = -O2 -arch=sm_86
# 源文件与目标文件
TARGETS = saxpy_cpp saxpy_cuda saxpy_cublas saxpy_acc

all: $(TARGETS)

saxpy_cpp: saxpy_cpp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

saxpy_cuda: saxpy_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

saxpy_cublas: saxpy_cublas.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas

saxpy_acc: saxpy_acc.cpp
	nvc++ $(CXXFLAGS) $(ACCFLAGS) -o $@ $<

run_cpp:
	./saxpy_cpp

run_cuda:
	./saxpy_cuda

run_cublas:
	./saxpy_cublas

run_acc:
	./saxpy_acc

clean:
	rm -f $(TARGETS)
