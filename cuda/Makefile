# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

NVCC := $(HOME)/miniconda3/envs/cuda/bin/nvcc

# -Xcompiler takes the next parameter to the CPU compiler
NVCCFLAGS := -arch=sm_86 -Xcompiler -Wall,-Wextra -O0

INCLUDE := -Iinclude

SRCS := source/tensor.cu source/operations.cu source/utils.cu source/gpt2.cu source/main.cu
OBJS = $(SRCS:.cu=.o)

TARGET := main

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $(OBJS) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
