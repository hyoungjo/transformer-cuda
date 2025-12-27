#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include <fstream>
#include <map>
#include <memory>
#include <string>

namespace utils {

template <typename T> T read_item(std::ifstream &file) {
  T val;
  file.read(reinterpret_cast<char *>(&val), sizeof(T));
  return val;
}

std::unique_ptr<Model> load_model(const std::string &model_name,
                                  const std::string &data_dir);
std::map<std::string, Tensor> load_data(const std::string &path);

} // namespace utils
