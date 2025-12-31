# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

NVCC := $(HOME)/miniconda3/envs/cuda/bin/nvcc

# -Xcompiler takes the next parameter to the CPU compiler
# -diag-suppress 177 suppresses unused declaration warnings
NVCCFLAGS := -arch=sm_86 -Xcompiler -Wall,-Wextra -O3 -diag-suppress 177

INCLUDE := -Iinclude

COMMON_SRCS := source/tensor.cu source/operations.cu source/utils.cu source/gpt2.cu source/llama3_8b.cu
COMMON_OBJS := $(COMMON_SRCS:.cu=.o)

MAIN_SRC := source/main.cu
MAIN_OBJ := $(MAIN_SRC:.cu=.o)

TIMER_SRC := source/timer.cu
TIMER_OBJ := $(TIMER_SRC:.cu=.o)

TARGETS := main timer

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

.PHONY: all clean

all: $(TARGETS)

main: $(COMMON_OBJS) $(MAIN_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $^ -o $@

timer: $(COMMON_OBJS) $(TIMER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f $(COMMON_OBJS) $(MAIN_OBJ) $(TIMER_OBJ) $(TARGETS)
