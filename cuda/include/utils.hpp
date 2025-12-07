#pragma once

#include "tensor.hpp"
#include <fstream>
#include <map>
#include <string>

namespace utils {

template <typename T> T read_item(std::ifstream &file) {
  T val;
  file.read(reinterpret_cast<char *>(&val), sizeof(T));
  return val;
}

std::map<std::string, Tensor> load_data(const std::string &path,
                                        std::string device);

} // namespace utils
