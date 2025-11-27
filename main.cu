#include <iostream>

__global__ void kernel() {}

int main() {
    std::cout << "Hello, CUDA!" << std::endl;
    kernel<<<1, 1>>>();
    return 0;
}
