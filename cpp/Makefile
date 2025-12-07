# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

CXX := $(HOME)/miniconda3/envs/cuda/bin/x86_64-conda-linux-gnu-g++

# Enable warnings and set the highest optimization level
CXXFLAGS := -Wall -Wextra -O3 -march=native

INCLUDE := -Iinclude

COMMON_SRCS := source/tensor.cpp source/operations.cpp source/utils.cpp source/gpt2.cpp
COMMON_OBJS := $(COMMON_SRCS:.cpp=.o)

MAIN_SRC := source/main.cpp
MAIN_OBJ := $(MAIN_SRC:.cpp=.o)

TIMER_SRC := source/timer.cpp
TIMER_OBJ := $(TIMER_SRC:.cpp=.o)

TARGETS := main timer

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

.PHONY: all clean

all: $(TARGETS)

main: $(COMMON_OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

timer: $(COMMON_OBJS) $(TIMER_OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f $(COMMON_OBJS) $(MAIN_OBJ) $(TIMER_OBJ) $(TARGETS)
