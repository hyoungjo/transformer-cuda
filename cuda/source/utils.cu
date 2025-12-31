#include "gpt2.cuh"
#include "llama3_8b.cuh"
#include "tensor.cuh"
#include "utils.cuh"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace utils {

std::unique_ptr<Model> load_model(const std::string &model_name,
                                  const std::string &data_dir) {
  if (model_name == "gpt2") {
    return std::make_unique<GPT2>(data_dir + "weights.bin");
  } else if (model_name == "meta-llama/Meta-Llama-3-8B") {
    return std::make_unique<LLaMA3_8B>(data_dir + "weights.bin");
  } else {
    std::cerr << "[CPP][ERROR] Unknown model name: " << model_name << std::endl;
    exit(1);
  }
}

std::map<std::string, Tensor> load_data(const std::string &path,
                                        std::string device) {
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

    std::vector<int64_t> shape;
    len = read_item<int>(file);
    for (int i = 0; i < len; ++i) {
      int64_t dim = static_cast<int64_t>(read_item<int>(file));
      shape.push_back(dim);
    }
    Tensor tensor(shape, "cpu");
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      tensor.h_data[i] = read_item<float>(file);
    }
    if (device == "gpu") {
      tensor.to(device);
    }

    tensors[name] = tensor;
  }

  return tensors;
}

} // namespace utils
