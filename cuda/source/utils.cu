#include "tensor.hpp"
#include "utils.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>

namespace utils {

std::map<std::string, Tensor> load_data(const std::string &path) {
  std::cout << "[CUDA][INFO] Loading data from " << path << std::endl;

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[CUDA][ERROR] " << path << " is not opened." << std::endl;
    exit(1);
  }

  std::map<std::string, Tensor> tensors;
  int n = read_item<int>(file);
  for (int i = 0; i < n; ++i) {
    int len = read_item<int>(file);
    std::string name(len, '\0');
    file.read(&name[0], len);

    Tensor tensor;
    len = read_item<int>(file);
    for (int i = 0; i < len; ++i) {
      int64_t dim = static_cast<int64_t>(read_item<int>(file));
      tensor.shape.push_back(dim);
    }
    tensor.resize(tensor.shape);
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      tensor.data[i] = read_item<float>(file);
    }

    tensors[name] = tensor;
  }

  return tensors;
}

} // namespace utils
