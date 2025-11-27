# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

NVCC := $(HOME)/miniconda3/envs/cuda/bin/nvcc

# -Xcompiler takes the next parameter to the CPU compiler
NVCCFLAGS := -arch=native -Xcompiler -Wall,-Wextra -O0

TARGET := main
SRCS := main.cu

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
