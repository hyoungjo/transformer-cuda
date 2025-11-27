# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

CXX := $(HOME)/miniconda3/envs/cuda/bin/x86_64-conda-linux-gnu-g++

# Enable warnings and set optimization level 0
CXXFLAGS := -Wall -Wextra -O0

TARGET := main
SRCS := main.cpp

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
