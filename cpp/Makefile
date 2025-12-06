# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

CXX := $(HOME)/miniconda3/envs/cuda/bin/x86_64-conda-linux-gnu-g++

# Enable warnings and set optimization level 0
CXXFLAGS := -Wall -Wextra -O0

INCLUDE := -Iinclude

SRCS := source/tensor.cpp source/operations.cpp source/utils.cpp source/gpt2.cpp source/main.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET := main

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
